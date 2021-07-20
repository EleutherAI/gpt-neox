import torch
from megatron import print_rank_0
from .initialize import get_model_parallel_group
from .initialize import get_model_parallel_rank
from .initialize import get_model_parallel_world_size
from .utils import VocabUtility

class _VocabParallelCrossEntropy(torch.autograd.Function):

    @staticmethod
    def forward(ctx, vocab_parallel_logits, target):

        # Maximum value along vocab dimension across all GPUs.
        logits_max = torch.max(vocab_parallel_logits, dim=-1)[0]
        torch.distributed.all_reduce(logits_max,
                                     op=torch.distributed.ReduceOp.MAX,
                                     group=get_model_parallel_group())
        # Subtract the maximum value.
        vocab_parallel_logits.sub_(logits_max.unsqueeze(dim=-1))

        # Get the partition's vocab indecies
        get_vocab_range = VocabUtility.vocab_range_from_per_partition_vocab_size
        partition_vocab_size = vocab_parallel_logits.size()[-1]
        rank = get_model_parallel_rank()
        world_size = get_model_parallel_world_size()
        vocab_start_index, vocab_end_index = get_vocab_range(
            partition_vocab_size, rank, world_size)

        # Create a mask of valid vocab ids (1 means it needs to be masked).
        target_mask = (target < vocab_start_index) | (target >= vocab_end_index)
        masked_target = target.clone() - vocab_start_index
        masked_target[target_mask] = 0

        # Get predicted-logits = logits[target].
        # For Simplicity, we convert logits to a 2-D tensor with size
        # [*, partition-vocab-size] and target to a 1-D tensor of size [*].
        logits_2d = vocab_parallel_logits.view(-1, partition_vocab_size)
        masked_target_1d = masked_target.view(-1)
        arange_1d = torch.arange(start=0, end=logits_2d.size()[0],
                                 device=logits_2d.device)
        predicted_logits_1d = logits_2d[arange_1d, masked_target_1d]
        predicted_logits_1d = predicted_logits_1d.clone().contiguous()
        predicted_logits = predicted_logits_1d.view_as(target)
        predicted_logits[target_mask] = 0.0

        # All reduce is needed to get the chunks from other GPUs.
        torch.distributed.all_reduce(predicted_logits,
                                     op=torch.distributed.ReduceOp.SUM,
                                     group=get_model_parallel_group())

        # Sum of exponential of logits along vocab dimension across all GPUs.
        exp_logits = vocab_parallel_logits
        torch.exp(vocab_parallel_logits, out=exp_logits)
        sum_exp_logits = exp_logits.sum(dim=-1)
        torch.distributed.all_reduce(sum_exp_logits,
                                     op=torch.distributed.ReduceOp.SUM,
                                     group=get_model_parallel_group())

        # Loss = log(sum(exp(logits))) - predicted-logit.
        loss = torch.log(sum_exp_logits) - predicted_logits

        # Store softmax, target-mask and masked-target for backward pass.
        exp_logits.div_(sum_exp_logits.unsqueeze(dim=-1))
        ctx.save_for_backward(exp_logits, target_mask, masked_target_1d)
        #print_rank_0("CELOSS "*10, loss.shape, loss.sum())

        return loss

    @staticmethod
    def backward(ctx, grad_output):

        # Retreive tensors from the forward path.
        softmax, target_mask, masked_target_1d = ctx.saved_tensors

        # All the inputs have softmax as thier gradient.
        grad_input = softmax
        # For simplicity, work with the 2D gradient.
        partition_vocab_size = softmax.size()[-1]
        grad_2d = grad_input.view(-1, partition_vocab_size)

        # Add the gradient from matching classes.
        arange_1d = torch.arange(start=0, end=grad_2d.size()[0],
                                 device=grad_2d.device)
        grad_2d[arange_1d, masked_target_1d] -= (
            1.0 - target_mask.view(-1).float())

        # Finally elementwise multiplication with the output gradients.
        grad_input.mul_(grad_output.unsqueeze(dim=-1))

        return grad_input, None

def vocab_parallel_cross_entropy(vocab_parallel_logits, target):
    """Helper function for the cross entropy."""
    return _VocabParallelCrossEntropy.apply(vocab_parallel_logits, target)

class _VocabParallelKLDivLoss(torch.autograd.Function):

    @staticmethod
    def forward(ctx, vocab_parallel_s_logits, 
                vocab_parallel_t_logits, 
                epsilon=1e-12):

        vocab_parallel_s_logits.add_(epsilon)
        vocab_parallel_t_logits.add_(epsilon)

        # Get the partition's vocab indecies
        get_vocab_range = VocabUtility.vocab_range_from_per_partition_vocab_size
        batch_size, seq_len, partition_vocab_size = \
                                        vocab_parallel_s_logits.size()
        rank = get_model_parallel_rank()
        world_size = get_model_parallel_world_size()
        vocab_start_index, vocab_end_index = get_vocab_range(
            partition_vocab_size, rank, world_size)

        # For Simplicity, we convert logits to a 2-D tensor with size
        # [*, partition-vocab-size]
        s_logits_2d = vocab_parallel_s_logits.view(batch_size*seq_len, -1)
        t_logits_2d = vocab_parallel_t_logits.view(batch_size*seq_len, -1) 

        target_mask = torch.zeros_like(s_logits_2d)
        mask = torch.arange(start=vocab_start_index, end=vocab_end_index, 
                            device=s_logits_2d.device)
        
        target_mask[:,mask] = 1.0
        s_logits_2d.mul_(target_mask)
        t_logits_2d.mul_(target_mask)

        del mask
        del target_mask

        # All reduce is needed to get the chunks from other GPUs.
        torch.distributed.all_reduce(s_logits_2d,
                                     op=torch.distributed.ReduceOp.SUM,
                                     group=get_model_parallel_group())
        torch.distributed.all_reduce(t_logits_2d,
                                    op=torch.distributed.ReduceOp.SUM,
                                    group=get_model_parallel_group())

        # Sum of exponential of logits along vocab dimension across all GPUs.
        exp_s_logits = s_logits_2d
        exp_t_logits = t_logits_2d

        torch.exp(s_logits_2d, out=exp_s_logits)
        torch.exp(t_logits_2d, out=exp_t_logits)

        sum_exp_s_logits = exp_s_logits.sum(dim=-1)
        sum_exp_t_logits = exp_t_logits.sum(dim=-1)

        torch.distributed.all_reduce(sum_exp_s_logits,
                                     op=torch.distributed.ReduceOp.SUM,
                                     group=get_model_parallel_group())
        torch.distributed.all_reduce(sum_exp_t_logits,
                                     op=torch.distributed.ReduceOp.SUM,
                                     group=get_model_parallel_group())

        # Softmax
        exp_s_logits.div_(sum_exp_s_logits.unsqueeze(dim=-1))
        exp_t_logits.div_(sum_exp_t_logits.unsqueeze(dim=-1))

        # Store softmax of student and teacher logits for backward pass.
        ctx.save_for_backward(exp_s_logits - exp_t_logits)

        # loss = p log(p/q)
        # loss = (exp_t_logits * (exp_t_logits/exp_s_logits).log())
        loss = exp_t_logits.mul_(exp_s_logits.div_(exp_t_logits).log_()).mul_(-1)
        loss = loss.sum(dim=-1).view(-1, seq_len).clone()

        del exp_s_logits
        del exp_t_logits
        #print_rank_0("KDLOSS "*10, loss.shape, loss.sum())
        return loss

    @staticmethod
    def backward(ctx, grad_output):
        grad_output_shape = grad_output.shape
        _reshape = (grad_output_shape[0], grad_output_shape[1], -1)

        grad_input, = ctx.saved_tensors
        grad_input = grad_input.view(_reshape)
        grad_input.mul_(grad_output.unsqueeze(dim=-1))

        return grad_input, None, None

def vocab_parallel_KLDivLoss(vocab_parallel_s_logits, vocab_parallel_t_logits, epsilon=1e-12):
    """Helper function for the cross entropy."""
    return _VocabParallelKLDivLoss.apply(vocab_parallel_s_logits, vocab_parallel_t_logits, epsilon)


class _VocabParallelMSELoss(torch.autograd.Function):

    @staticmethod
    def forward(ctx, vocab_parallel_s_logits, vocab_parallel_t_logits):
        _logit_shape = vocab_parallel_s_logits.shape

        # Get the partition's vocab indecies
        get_vocab_range = VocabUtility.vocab_range_from_per_partition_vocab_size
        batch_size, seq_len, partition_vocab_size = vocab_parallel_s_logits.size()
        rank = get_model_parallel_rank()
        world_size = get_model_parallel_world_size()
        vocab_start_index, vocab_end_index = get_vocab_range(
            partition_vocab_size, rank, world_size)

        # For Simplicity, we convert logits to a 2-D tensor with size
        # [*, partition-vocab-size]
        s_logits_2d = vocab_parallel_s_logits.view(batch_size*seq_len, partition_vocab_size)
        t_logits_2d = vocab_parallel_t_logits.view(batch_size*seq_len, partition_vocab_size)

        target_mask = torch.zeros_like(s_logits_2d)
        mask = torch.arange(start= vocab_start_index, end= vocab_end_index)
        
        target_mask[:,mask] = 1.0
        s_logits_2d.mul_(target_mask)
        t_logits_2d.mul_(target_mask)

        del target_mask
        del mask

        # All reduce is needed to get the chunks from other GPUs.
        torch.distributed.all_reduce(s_logits_2d,
                                     op=torch.distributed.ReduceOp.SUM,
                                     group=get_model_parallel_group())
        torch.distributed.all_reduce(t_logits_2d,
                                    op=torch.distributed.ReduceOp.SUM,
                                    group=get_model_parallel_group())

        logits_diff = t_logits_2d.sub_(s_logits_2d)
        #ctx.save_for_backward(logits_diff.div(t_logits_2d.shape[0]))
        ctx.save_for_backward(logits_diff.div(torch.numel(t_logits_2d)))
        loss = logits_diff.square_().mean(dim=-1).view(-1, seq_len).clone()
        #loss = logits_diff.square_().sum(dim=-1).view(-1, seq_len).clone()

        del t_logits_2d
        del s_logits_2d
        return loss

    @staticmethod
    def backward(ctx, grad_output):
        grad_output_shape = grad_output.shape
        # Retreive tensors from the forward path.
        logits_diff, = ctx.saved_tensors
        logits_diff = logits_diff.view(grad_output_shape[0], grad_output_shape[1], -1)
        grad_input = logits_diff.mul_(-2)
        # Finally elementwise multiplication with the output gradients.
        grad_input.mul_(grad_output.unsqueeze(dim=-1))
        return grad_input, None

def vocab_parallel_MSELoss(vocab_parallel_s_logits, vocab_parallel_t_logits):
    """Helper function for the cross entropy."""
    return _VocabParallelMSELoss.apply(vocab_parallel_s_logits, vocab_parallel_t_logits)
