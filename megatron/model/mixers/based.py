### TODO: top-of-file imports


### TODO: implement a registry and get_mixer() fn

### TODO: Based LinAttn module


class TaylorExp(nn.Module):
    # TODO: cite zoology based mixer for source, though modified
    
    # want to compute Taylor approx of exp(q^T k / sqrt(d)) 
    # ~= 1 + (q^T k / sqrt(d)) + (q^T k / sqrt(d))^2 / 2

    # or rather, want something s.t. 
    # when we mult feature_map(q) and feature_map(k)
    # together, we get the above taylor approx.


    # so we'll compute 1, x / sqrt(sqrt(d)), and (x^2 /sqrt(d)) / sqrt(2)
    # for x = q and x = k.

    def __init__(
        self,
        head_dim,
        head_dim_idx,
    ):
        super().__init__()

        self.root_2 = math.sqrt(2)
        self.root_d = math.sqrt(head_dim)
        self.rroot_d = math.sqrt(head_dim)

        self.head_dim_idx = head_dim_idx


    def forward(self, x):
        # x has dims [sq, b, np, hn] or (seqlen, bs, n_heads, head_dim)
        # Based code assumes (batch_size, n_heads, seq_len, head_dim)
        # (^ STALE: we're using same shapes as Based zoology mixer now)


        # second-order term
        x2 = (x.unsqueeze(-1) * x.unsqueeze(-2)).flatten(-2) / (self.root_2 * self.root_d)
        return torch.cat([
            torch.ones(x[:1].shape, device=x.device),
            x / self.root_d,
            x2
        ], dim=self.head_dim_idx)

class ParallelBasedLinAttention(nn.Module):

    """
    The linear attention block from Based, using Taylor approximation of attention
    """

    def __init__(
        self,
        neox_args,
    ):
        super().__init__()

        self.fp16 = neox_args.precision == "fp16"
        self.bf16 = neox_args.precision == "bfloat16"

        world_size = mpu.get_model_parallel_world_size()
        self.hidden_size_per_partition = mpu.divide(neox_args.hidden_size, world_size)
        self.hidden_size_per_attention_head = mpu.divide(
            neox_args.hidden_size, neox_args.num_attention_heads
        )
        self.num_attention_heads_per_partition = mpu.divide(
            neox_args.num_attention_heads, world_size
        )

        # Strided linear layer.
        self.query_key_value = mpu.ColumnParallelLinear(
            neox_args=neox_args,
            input_size=neox_args.hidden_size,
            output_size=3 * neox_args.hidden_size,
            gather_output=False,
            init_method=init_method,
            bias=neox_args.use_bias_in_attn_linear,
        )

        # TODO: norm_factor stuff, muP?

        # pos emb stuff doesn't go here

        # TODO: configure backend type, identity map type somehow

        # TODO: dropout??

        # Output.
        self.dense = mpu.RowParallelLinear(
            neox_args=neox_args,
            input_size=neox_args.hidden_size,
            output_size=neox_args.hidden_size,
            input_is_parallel=True,
            init_method=output_layer_init_method,
            skip_bias_add=True,
            parallel_output=parallel_output,
            bias=neox_args.use_bias_in_attn_linear,
        )
        # neox boilerplate ^

        self.feature_map = TaylorExp(
            head_dim=self.hidden_size_per_attention_head,
            head_dim_idx=-1,
        )

    def forward(
        self, 
        hidden_states, 
        attention_mask, # this will be ignored for now?
        layer_past=None, # TODO: assert that this is None
    ):

        # hidden_states: [sq, b, h]

        # =====================
        # Query, Key, and Value
        # =====================

        # Attention heads [sq, b, h] --> [sq, b, (np * 3 * hn)]
        mixed_x_layer, _ = self.query_key_value(hidden_states)

        # [sq, b, (np * 3 * hn)] --> [sq, b, np, 3 * hn]
        new_tensor_shape = mixed_x_layer.size()[:-1] + (
            self.num_attention_heads_per_partition,
            3 * self.hidden_size_per_attention_head,
        )
        mixed_x_layer = mixed_x_layer.view(*new_tensor_shape)

        # [sq, b, np, 3 * hn] --> 3 [sq, b, np, hn]
        (query_layer, key_layer, value_layer) = mpu.split_tensor_along_last_dim(
            mixed_x_layer, 3
        )

        # TODO: based uses a separate feature_dim and head_dim, separate n_kv_heads and n_heads


        # TODO: eliminate some reshapes here
        sq, b, np, hn = query_layer.size()
        query_layer = query_layer.view(b, sq, np, hn).transpose(1, 2)
        key_layer = key_layer.view(b, sq, np, hn).transpose(1, 2)
        value_layer = value_layer.view(b, sq, np, hn).transpose(1, 2)

        # TODO: will this be compatible w/ decoding or w/ sk != sq ?

        query_layer, key_layer = self.feature_map(query_layer), self.feature_map(key_layer)
        query_layer, key_layer, value_layer = query_layer.unsqueeze(-2), key_layer.unsqueeze(-2), value_layer.unsqueeze(-1)

        # perform lin attention
        attention_output = self.naive_attention(query_layer, key_layer, value_layer)

        # reshape to sq, b, hp (seqlen, batch, hidden) from [b, np, sq, hn] (batch, n heads, seqlen, head dim)
        # TODO: it appears we can reuse reshapes from attention (transformer.py#L726-737)
        # [b, np, sq, hn] --> [sq, b, np, hn]
        context_layer = context_layer.permute(2, 0, 1, 3).contiguous()

        # [sq, b, np, hn] --> [sq, b, hp]
        new_context_layer_shape = context_layer.size()[:-2] + (
            self.hidden_size_per_partition,
        )
        context_layer = context_layer.view(*new_context_layer_shape)

        # =================
        # Output. [sq, b, h]
        # =================

        output, bias = self.dense(attention_output)

        return output, bias

    def naive_attention(self, q, k, v): # TODO: based attention (quadratic mem. complexity)
        # Causal attn
        y = ((q * (k * v).cumsum(dim=2)).sum(dim=-1) / 
                 ((q * k.cumsum(dim=2)).sum(dim=-1) + self.eps))
        
        return y

    def flash_based(self): #TODO: change fn name #TODO: create a triton kernel
        raise NotImplementedError

class ParallelBasedGatedConv(nn.Module):

    """
    Gated short-convolution sub-block of Based.
    """

    def __init__(
        self,
        neox_args,
    ):

    def forward():

    def 

### TODO: add ability to provide a list of mixernames for both state and sequence mixers.

### TODO: MambaBlock

### TODO: GLA Lin Attn 

### TODO: Hyena block


### TODO: S4 block?


### TODO: RWKV4 block

### TODO: RWKV5 block

### TODO: RWKV6 block?

### TODO: H3 block?