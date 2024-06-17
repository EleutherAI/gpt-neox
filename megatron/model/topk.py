"""
copied from https://github.com/openai/sparse_autoencoder/blob/main/sparse_autoencoder/kernels.py
"""

import torch

import triton
import triton.language as tl


## kernels


def triton_sparse_transpose_dense_matmul(
    sparse_indices: torch.Tensor,
    sparse_values: torch.Tensor,
    dense: torch.Tensor,
    N: int,
    BLOCK_SIZE_AK=128,
) -> torch.Tensor:
    """
    calculates sparse.T @ dense (i.e reducing along the collated dimension of sparse)
    dense must be contiguous along dim 0 (in other words, dense.T is contiguous)

    sparse_indices is shape (A, k)
    sparse_values is shape (A, k)
    dense is shape (A, B)

    output is shape (N, B)
    """

    assert sparse_indices.shape == sparse_values.shape
    assert sparse_indices.is_contiguous()
    assert sparse_values.is_contiguous()
    assert dense.is_contiguous()  # contiguous along B

    K = sparse_indices.shape[1]
    A = dense.shape[0]
    B = dense.shape[1]
    assert sparse_indices.shape[0] == A

    # COO-format and sorted
    sorted_indices = sparse_indices.view(-1).sort()
    coo_indices = torch.stack(
        [
            torch.arange(A, device=sparse_indices.device).repeat_interleave(K)[
                sorted_indices.indices
            ],
            sorted_indices.values,
        ]
    )  # shape (2, A * K)
    coo_values = sparse_values.view(-1)[sorted_indices.indices]  # shape (A * K,)
    return triton_coo_sparse_dense_matmul(coo_indices, coo_values, dense, N, BLOCK_SIZE_AK)


def triton_coo_sparse_dense_matmul(
    coo_indices: torch.Tensor,
    coo_values: torch.Tensor,
    dense: torch.Tensor,
    N: int,
    BLOCK_SIZE_AK=128,
) -> torch.Tensor:
    AK = coo_indices.shape[1]
    B = dense.shape[1]

    out = torch.zeros(N, B, device=dense.device, dtype=coo_values.dtype)

    grid = lambda META: (
        triton.cdiv(AK, META["BLOCK_SIZE_AK"]),
        1,
    )
    triton_sparse_transpose_dense_matmul_kernel[grid](
        coo_indices,
        coo_values,
        dense,
        out,
        stride_da=dense.stride(0),
        stride_db=dense.stride(1),
        B=B,
        N=N,
        AK=AK,
        BLOCK_SIZE_AK=BLOCK_SIZE_AK,
        BLOCK_SIZE_B=triton.next_power_of_2(B),
    )
    return out


@triton.jit
def triton_sparse_transpose_dense_matmul_kernel(
    coo_indices_ptr,
    coo_values_ptr,
    dense_ptr,
    out_ptr,
    stride_da,
    stride_db,
    B,
    N,
    AK,
    BLOCK_SIZE_AK: tl.constexpr,
    BLOCK_SIZE_B: tl.constexpr,
):
    """
    coo_indices is shape (2, AK)
    coo_values is shape (AK,)
    dense is shape (A, B), contiguous along B
    out is shape (N, B)
    """

    pid_ak = tl.program_id(0)
    pid_b = tl.program_id(1)

    coo_offsets = tl.arange(0, BLOCK_SIZE_AK)
    b_offsets = tl.arange(0, BLOCK_SIZE_B)

    A_coords = tl.load(
        coo_indices_ptr + pid_ak * BLOCK_SIZE_AK + coo_offsets,
        mask=pid_ak * BLOCK_SIZE_AK + coo_offsets < AK,
    )
    K_coords = tl.load(
        coo_indices_ptr + pid_ak * BLOCK_SIZE_AK + coo_offsets + AK,
        mask=pid_ak * BLOCK_SIZE_AK + coo_offsets < AK,
    )
    values = tl.load(
        coo_values_ptr + pid_ak * BLOCK_SIZE_AK + coo_offsets,
        mask=pid_ak * BLOCK_SIZE_AK + coo_offsets < AK,
    )

    last_k = tl.min(K_coords)
    accum = tl.zeros((BLOCK_SIZE_B,), dtype=tl.float32)

    for ind in range(BLOCK_SIZE_AK):
        if ind + pid_ak * BLOCK_SIZE_AK < AK:
            # workaround to do A_coords[ind]
            a = tl.sum(
                tl.where(
                    tl.arange(0, BLOCK_SIZE_AK) == ind,
                    A_coords,
                    tl.zeros((BLOCK_SIZE_AK,), dtype=tl.int64),
                )
            )

            k = tl.sum(
                tl.where(
                    tl.arange(0, BLOCK_SIZE_AK) == ind,
                    K_coords,
                    tl.zeros((BLOCK_SIZE_AK,), dtype=tl.int64),
                )
            )

            v = tl.sum(
                tl.where(
                    tl.arange(0, BLOCK_SIZE_AK) == ind,
                    values,
                    tl.zeros((BLOCK_SIZE_AK,), dtype=tl.float32),
                )
            )

            tl.device_assert(k < N)

            if k != last_k:
                tl.atomic_add(
                    out_ptr + last_k * B + BLOCK_SIZE_B * pid_b + b_offsets,
                    accum,
                    mask=BLOCK_SIZE_B * pid_b + b_offsets < B,
                )
                accum *= 0
                last_k = k

            if v != 0:
                accum += v * tl.load(dense_ptr + a * stride_da + b_offsets, mask=b_offsets < B)

    tl.atomic_add(
        out_ptr + last_k * B + BLOCK_SIZE_B * pid_b + b_offsets,
        accum,
        mask=BLOCK_SIZE_B * pid_b + b_offsets < B,
    )


def triton_sparse_dense_matmul(
    sparse_indices: torch.Tensor,
    sparse_values: torch.Tensor,
    dense: torch.Tensor,
) -> torch.Tensor:
    """
    calculates sparse @ dense (i.e reducing along the uncollated dimension of sparse)
    dense must be contiguous along dim 0 (in other words, dense.T is contiguous)

    sparse_indices is shape (A, k)
    sparse_values is shape (A, k)
    dense is shape (N, B)

    output is shape (A, B)
    """
    N = dense.shape[0]
    assert sparse_indices.shape == sparse_values.shape
    assert sparse_indices.is_contiguous()
    assert sparse_values.is_contiguous()
    assert dense.is_contiguous()  # contiguous along B

    A = sparse_indices.shape[0]
    K = sparse_indices.shape[1]
    B = dense.shape[1]

    out = torch.zeros(A, B, device=dense.device, dtype=sparse_values.dtype)

    triton_sparse_dense_matmul_kernel[(A,)](
        sparse_indices,
        sparse_values,
        dense,
        out,
        stride_dn=dense.stride(0),
        stride_db=dense.stride(1),
        A=A,
        B=B,
        N=N,
        K=K,
        BLOCK_SIZE_K=triton.next_power_of_2(K),
        BLOCK_SIZE_B=triton.next_power_of_2(B),
    )
    return out


@triton.jit
def triton_sparse_dense_matmul_kernel(
    sparse_indices_ptr,
    sparse_values_ptr,
    dense_ptr,
    out_ptr,
    stride_dn,
    stride_db,
    A,
    B,
    N,
    K,
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_B: tl.constexpr,
):
    """
    sparse_indices is shape (A, K)
    sparse_values is shape (A, K)
    dense is shape (N, B), contiguous along B
    out is shape (A, B)
    """

    pid = tl.program_id(0)

    offsets_k = tl.arange(0, BLOCK_SIZE_K)
    sparse_indices = tl.load(
        sparse_indices_ptr + pid * K + offsets_k, mask=offsets_k < K
    )  # shape (K,)
    sparse_values = tl.load(
        sparse_values_ptr + pid * K + offsets_k, mask=offsets_k < K
    )  # shape (K,)

    accum = tl.zeros((BLOCK_SIZE_B,), dtype=tl.float32)

    offsets_b = tl.arange(0, BLOCK_SIZE_B)

    for k in range(K):
        # workaround to do sparse_indices[k]
        i = tl.sum(
            tl.where(
                tl.arange(0, BLOCK_SIZE_K) == k,
                sparse_indices,
                tl.zeros((BLOCK_SIZE_K,), dtype=tl.int64),
            )
        )
        # workaround to do sparse_values[k]
        v = tl.sum(
            tl.where(
                tl.arange(0, BLOCK_SIZE_K) == k,
                sparse_values,
                tl.zeros((BLOCK_SIZE_K,), dtype=tl.float32),
            )
        )

        tl.device_assert(i < N)
        if v != 0:
            accum += v * tl.load(
                dense_ptr + i * stride_dn + offsets_b * stride_db, mask=offsets_b < B
            )

    tl.store(out_ptr + pid * B + offsets_b, accum.to(sparse_values.dtype), mask=offsets_b < B)


def triton_dense_dense_sparseout_matmul(
    dense1: torch.Tensor,
    dense2: torch.Tensor,
    at_indices: torch.Tensor,
) -> torch.Tensor:
    """
    dense1: shape (A, B)
    dense2: shape (B, N)
    at_indices: shape (A, K)
    out values: shape (A, K)
    calculates dense1 @ dense2 only for the indices in at_indices

    equivalent to (dense1 @ dense2).gather(1, at_indices)
    """
    A, B = dense1.shape
    N = dense2.shape[1]
    assert dense2.shape[0] == B
    assert at_indices.shape[0] == A
    K = at_indices.shape[1]
    assert at_indices.is_contiguous()

    assert dense1.stride(1) == 1, "dense1 must be contiguous along B"
    assert dense2.stride(0) == 1, "dense2 must be contiguous along B"

    if K > 512:
        # print("WARN - using naive matmul for large K")
        # naive is more efficient for large K
        return (dense1 @ dense2).gather(1, at_indices)

    out = torch.zeros(A, K, device=dense1.device, dtype=dense1.dtype)

    # grid = lambda META: (triton.cdiv(A, META['BLOCK_SIZE_A']),)

    triton_dense_dense_sparseout_matmul_kernel[(A,)](
        dense1,
        dense2,
        at_indices,
        out,
        stride_d1a=dense1.stride(0),
        stride_d1b=dense1.stride(1),
        stride_d2b=dense2.stride(0),
        stride_d2n=dense2.stride(1),
        A=A,
        B=B,
        N=N,
        K=K,
        BLOCK_SIZE_B=triton.next_power_of_2(B),
        BLOCK_SIZE_N=triton.next_power_of_2(N),
        BLOCK_SIZE_K=triton.next_power_of_2(K),
    )

    return out


@triton.jit
def triton_dense_dense_sparseout_matmul_kernel(
    dense1_ptr,
    dense2_ptr,
    at_indices_ptr,
    out_ptr,
    stride_d1a,
    stride_d1b,
    stride_d2b,
    stride_d2n,
    A,
    B,
    N,
    K,
    BLOCK_SIZE_B: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    """
    dense1: shape (A, B)
    dense2: shape (B, N)
    at_indices: shape (A, K)
    out values: shape (A, K)
    """

    pid = tl.program_id(0)

    offsets_k = tl.arange(0, BLOCK_SIZE_K)
    at_indices = tl.load(at_indices_ptr + pid * K + offsets_k, mask=offsets_k < K)  # shape (K,)

    offsets_b = tl.arange(0, BLOCK_SIZE_B)
    dense1 = tl.load(
        dense1_ptr + pid * stride_d1a + offsets_b * stride_d1b, mask=offsets_b < B
    )  # shape (B,)

    accum = tl.zeros((BLOCK_SIZE_K,), dtype=tl.float32)

    for k in range(K):
        # workaround to do at_indices[b]
        i = tl.sum(
            tl.where(
                tl.arange(0, BLOCK_SIZE_K) == k,
                at_indices,
                tl.zeros((BLOCK_SIZE_K,), dtype=tl.int64),
            )
        )
        tl.device_assert(i < N)

        dense2col = tl.load(
            dense2_ptr + offsets_b * stride_d2b + i * stride_d2n, mask=offsets_b < B
        )  # shape (B,)
        accum += tl.where(
            tl.arange(0, BLOCK_SIZE_K) == k,
            tl.sum(dense1 * dense2col),
            tl.zeros((BLOCK_SIZE_K,), dtype=tl.int64),
        )

    tl.store(out_ptr + pid * K + offsets_k, accum, mask=offsets_k < K)


class TritonDecoderAutograd(torch.autograd.Function):
    @staticmethod
    def forward(ctx, sparse_indices, sparse_values, decoder_weight):
        ctx.save_for_backward(sparse_indices, sparse_values, decoder_weight)
        return triton_sparse_dense_matmul(sparse_indices, sparse_values, decoder_weight.T)

    @staticmethod
    def backward(ctx, grad_output):
        sparse_indices, sparse_values, decoder_weight = ctx.saved_tensors

        assert grad_output.is_contiguous(), "grad_output must be contiguous; this is probably because the subsequent op was a .sum() or something like that, which returns a non contiguous gradient"

        decoder_grad = triton_sparse_transpose_dense_matmul(
            sparse_indices, sparse_values, grad_output, N=decoder_weight.shape[1]
        ).T

        return (
            None,
            triton_dense_dense_sparseout_matmul(grad_output, decoder_weight, sparse_indices),
            # decoder is contiguous when transposed so this is a matching layout
            decoder_grad,
            None,
        )


def triton_add_mul_(
    x: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    c: float,
):
    """
    does
    x += a * b * c

    x : [m, n]
    a : [m, n]
    b : [m, n]
    c : float
    """

    if len(a.shape) == 1:
        a = a[None, :].broadcast_to(x.shape)

    if len(b.shape) == 1:
        b = b[None, :].broadcast_to(x.shape)

    assert x.shape == a.shape == b.shape

    BLOCK_SIZE_M = 64
    BLOCK_SIZE_N = 64
    grid = lambda META: (
        triton.cdiv(x.shape[0], META["BLOCK_SIZE_M"]),
        triton.cdiv(x.shape[1], META["BLOCK_SIZE_N"]),
    )
    triton_add_mul_kernel[grid](
        x,
        a,
        b,
        c,
        x.stride(0),
        x.stride(1),
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        BLOCK_SIZE_M,
        BLOCK_SIZE_N,
        x.shape[0],
        x.shape[1],
    )


@triton.jit
def triton_add_mul_kernel(
    x_ptr,
    a_ptr,
    b_ptr,
    c,
    stride_x0,
    stride_x1,
    stride_a0,
    stride_a1,
    stride_b0,
    stride_b1,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    M: tl.constexpr,
    N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offsets_m = tl.arange(0, BLOCK_SIZE_M) + pid_m * BLOCK_SIZE_M
    offsets_n = tl.arange(0, BLOCK_SIZE_N) + pid_n * BLOCK_SIZE_N

    x = tl.load(
        x_ptr + offsets_m[:, None] * stride_x0 + offsets_n[None, :] * stride_x1,
        mask=(offsets_m[:, None] < M) & (offsets_n[None, :] < N),
    )
    a = tl.load(
        a_ptr + offsets_m[:, None] * stride_a0 + offsets_n[None, :] * stride_a1,
        mask=(offsets_m[:, None] < M) & (offsets_n[None, :] < N),
    )
    b = tl.load(
        b_ptr + offsets_m[:, None] * stride_b0 + offsets_n[None, :] * stride_b1,
        mask=(offsets_m[:, None] < M) & (offsets_n[None, :] < N),
    )

    x_dtype = x.dtype
    x = (x.to(tl.float32) + a.to(tl.float32) * b.to(tl.float32) * c).to(x_dtype)

    tl.store(
        x_ptr + offsets_m[:, None] * stride_x0 + offsets_n[None, :] * stride_x1,
        x,
        mask=(offsets_m[:, None] < M) & (offsets_n[None, :] < N),
    )



def triton_sum_dim0_in_fp32(xs):
    a, b = xs.shape

    assert xs.is_contiguous()
    assert xs.dtype == torch.float16

    BLOCK_SIZE_A = min(triton.next_power_of_2(a), 512)
    BLOCK_SIZE_B = 64  # cache line is 128 bytes

    out = torch.zeros(b, dtype=torch.float32, device=xs.device)

    grid = lambda META: (triton.cdiv(b, META["BLOCK_SIZE_B"]),)

    triton_sum_dim0_in_fp32_kernel[grid](
        xs,
        out,
        stride_a=xs.stride(0),
        a=a,
        b=b,
        BLOCK_SIZE_A=BLOCK_SIZE_A,
        BLOCK_SIZE_B=BLOCK_SIZE_B,
    )

    return out


@triton.jit
def triton_sum_dim0_in_fp32_kernel(
    xs_ptr,
    out_ptr,
    stride_a,
    a,
    b,
    BLOCK_SIZE_A: tl.constexpr,
    BLOCK_SIZE_B: tl.constexpr,
):
    # each program handles 64 columns of xs
    pid = tl.program_id(0)
    offsets_b = tl.arange(0, BLOCK_SIZE_B) + pid * BLOCK_SIZE_B

    all_out = tl.zeros((BLOCK_SIZE_B,), dtype=tl.float32)

    for i in range(0, a, BLOCK_SIZE_A):
        offsets_a = tl.arange(0, BLOCK_SIZE_A) + i
        xs = tl.load(
            xs_ptr + offsets_a[:, None] * stride_a + offsets_b[None, :],
            mask=(offsets_a < a)[:, None] & (offsets_b < b)[None, :],
            other=0,
        )
        xs = xs.to(tl.float32)
        out = tl.sum(xs, axis=0)
        all_out += out

    tl.store(out_ptr + offsets_b, all_out, mask=offsets_b < b)


def mse(
    output,
    target,
):  # fusing fp32 cast and MSE to save memory
    assert output.shape == target.shape
    assert len(output.shape) == 2
    assert output.stride(1) == 1
    assert target.stride(1) == 1

    a, b = output.shape

    BLOCK_SIZE_B = triton.next_power_of_2(b)

    class _MSE(torch.autograd.Function):
        @staticmethod
        def forward(ctx, output, target):
            ctx.save_for_backward(output, target)
            out = torch.zeros(a, dtype=torch.float32, device=output.device)

            triton_mse_loss_fp16_kernel[(a,)](
                output,
                target,
                out,
                stride_a_output=output.stride(0),
                stride_a_target=target.stride(0),
                a=a,
                b=b,
                BLOCK_SIZE_B=BLOCK_SIZE_B,
            )

            return out

        @staticmethod
        def backward(ctx, grad_output):
            output, target = ctx.saved_tensors
            res = (output - target).float()
            res *= grad_output[:, None] * 2 / b
            return res, None

    return _MSE.apply(output, target).mean()


def normalized_mse(recon: torch.Tensor, xs: torch.Tensor) -> torch.Tensor:
    # only used for auxk
    xs_mu = (
        triton_sum_dim0_in_fp32(xs) / xs.shape[0]
        if xs.dtype == torch.float16
        else xs.mean(dim=0)
    )

    loss = mse(recon, xs) / mse(
        xs_mu[None, :].broadcast_to(xs.shape), xs
    )

    return loss


@triton.jit
def triton_mse_loss_fp16_kernel(
    output_ptr,
    target_ptr,
    out_ptr,
    stride_a_output,
    stride_a_target,
    a,
    b,
    BLOCK_SIZE_B: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets_b = tl.arange(0, BLOCK_SIZE_B)

    output = tl.load(
        output_ptr + pid * stride_a_output + offsets_b,
        mask=offsets_b < b,
    )
    target = tl.load(
        target_ptr + pid * stride_a_target + offsets_b,
        mask=offsets_b < b,
    )

    output = output.to(tl.float32)
    target = target.to(tl.float32)

    mse = tl.sum((output - target) * (output - target)) / b

    tl.store(out_ptr + pid, mse)


def triton_add_mul_(
    x: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    c: float,
):
    """
    does
    x += a * b * c

    x : [m, n]
    a : [m, n]
    b : [m, n]
    c : float
    """

    if len(a.shape) == 1:
        a = a[None, :].broadcast_to(x.shape)

    if len(b.shape) == 1:
        b = b[None, :].broadcast_to(x.shape)

    assert x.shape == a.shape == b.shape

    BLOCK_SIZE_M = 64
    BLOCK_SIZE_N = 64
    grid = lambda META: (
        triton.cdiv(x.shape[0], META["BLOCK_SIZE_M"]),
        triton.cdiv(x.shape[1], META["BLOCK_SIZE_N"]),
    )
    triton_add_mul_kernel[grid](
        x,
        a,
        b,
        c,
        x.stride(0),
        x.stride(1),
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        BLOCK_SIZE_M,
        BLOCK_SIZE_N,
        x.shape[0],
        x.shape[1],
    )


### Our addition: down-proj autograd operator

class TritonDownProjAutograd(torch.autograd.Function):
    @staticmethod
    def forward(ctx, sparse_indices, sparse_values, decoder_weight):
        ctx.save_for_backward(sparse_indices, sparse_values, decoder_weight)
        return triton_sparse_dense_matmul(sparse_indices, sparse_values, decoder_weight.T)

    @staticmethod
    def backward(ctx, grad_output):
        sparse_indices, sparse_values, decoder_weight = ctx.saved_tensors

        assert grad_output.is_contiguous(), "grad_output must be contiguous; this is probably because the subsequent op was a .sum() or something like that, which returns a non contiguous gradient"

        decoder_grad = triton_sparse_transpose_dense_matmul(
            sparse_indices, sparse_values, grad_output, N=decoder_weight.shape[1]
        ).T

        return (
            None,
            triton_dense_dense_sparseout_matmul(grad_output, decoder_weight, sparse_indices),
            # decoder is contiguous when transposed so this is a matching layout
            decoder_grad,
            None,
        )

topk_down_proj = TritonDownProjAutograd.apply