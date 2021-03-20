import torch
from torch.optim import Optimizer


class SM3(Optimizer):
    """Implements SM3 algorithm.
    It has been proposed in `Memory-Efficient Adaptive Optimization`_.
    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): coefficient that scale delta before it is applied
            to the parameters (default: 0.1)
        momentum (float, optional): coefficient used to scale prior updates
            before adding. This drastically increases memory usage if
            `momentum > 0.0`. This is ignored if the parameter's gradient
            is sparse. (default: 0.0)
        beta (float, optional): coefficient used for exponential moving
            averages (default: 0.0)
        eps (float, optional): Term added to square-root in denominator to
            improve numerical stability (default: 1e-30)
    .. _Memory-Efficient Adaptive Optimization:
        https://arxiv.org/abs/1901.11150
    """

    def __init__(self, params, lr=0.1, momentum=0.0, beta=0.0, eps=1e-30):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {0}".format(lr))
        if not 0.0 <= momentum < 1.0:
            raise ValueError("Invalid momentum: {0}".format(momentum))
        if not 0.0 <= beta < 1.0:
            raise ValueError("Invalid beta: {0}".format(beta))
        if not 0.0 <= eps:
            raise ValueError("Invalid eps: {0}".format(eps))

        defaults = {'lr': lr, 'momentum': momentum, 'beta': beta, 'eps': eps}
        super(SM3, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            momentum = group['momentum']
            beta = group['beta']
            eps = group['eps']
            for p in group['params']:
                if p is None:
                    continue
                grad = p.grad

                state = self.state[p]
                shape = grad.shape
                rank = len(shape)

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['momentum_buffer'] = 0.
                    _add_initial_accumulators(state, grad)

                if grad.is_sparse:
                    # the update is non-linear so indices must be unique
                    grad.coalesce()
                    grad_indices = grad._indices()
                    grad_values = grad._values()

                    # Transform update_values into sparse tensor
                    def make_sparse(values):
                        constructor = grad.new
                        if grad_indices.dim() == 0 or values.dim() == 0:
                            return constructor().resize_as_(grad)
                        return constructor(grad_indices, values, grad.size())

                    acc = state[_key(0)]
                    update_values = _compute_sparse_update(beta, acc, grad_values, grad_indices)

                    self._update_sparse_accumulator(beta, acc, make_sparse(update_values))

                    # Add small amount for numerical stability
                    update_values.add_(eps).rsqrt_().mul_(grad_values)

                    update = make_sparse(update_values)
                else:
                    # Get previous accumulators mu_{t-1}
                    if rank > 1:
                        acc_list = [state[_key(i)] for i in range(rank)]
                    else:
                        acc_list = [state[_key(0)]]

                    # Get update from accumulators and gradients
                    update = _compute_update(beta, acc_list, grad)

                    # Update accumulators.
                    self._update_accumulator(beta, acc_list, update)

                    # Add small amount for numerical stability
                    update.add_(eps).rsqrt_().mul_(grad)

                    if momentum > 0.:
                        m = state['momentum_buffer']
                        update.mul_(1. - momentum).add_(m, alpha=momentum)
                        state['momentum_buffer'] = update.detach()

                p.sub_(update, alpha=group['lr'])
                state['step'] += 1
        return loss

    @staticmethod
    def _update_accumulator(beta, acc_list, update):
        for i, acc in enumerate(acc_list):
            nu_max = _max_reduce_except_dim(update, i)
            if beta > 0.:
                torch.max(acc, nu_max, out=acc)
            else:
                # No need to compare - nu_max is bigger because of grad ** 2
                acc.copy_(nu_max)

    @staticmethod
    def _update_sparse_accumulator(beta, acc, update):
        nu_max = _max_reduce_except_dim(update.to_dense(), 0).squeeze()
        if beta > 0.:
            torch.max(acc, nu_max, out=acc)
        else:
            # No need to compare - nu_max is bigger because of grad ** 2
            acc.copy_(nu_max)


def _compute_sparse_update(beta, acc, grad_values, grad_indices):
    # In the sparse case, a single accumulator is used.
    update_values = torch.gather(acc, 0, grad_indices[0])
    if beta > 0.:
        update_values.mul_(beta)
    update_values.addcmul_(grad_values, grad_values, value=1. - beta)
    return update_values


def _compute_update(beta, acc_list, grad):
    rank = len(acc_list)
    update = acc_list[0].clone()
    for i in range(1, rank):
        # We rely on broadcasting to get the proper end shape.
        update = torch.min(update, acc_list[i])
    if beta > 0.:
        update.mul_(beta)
    update.addcmul_(grad, grad, value=1. - beta)

    return update


def _key(i):
    # Returns key used for accessing accumulators
    return 'accumulator_' + str(i)


def _add_initial_accumulators(state, grad):
    # Creates initial accumulators. For a dense tensor of shape (n1, n2, n3),
    # then our initial accumulators are of shape (n1, 1, 1), (1, n2, 1) and
    # (1, 1, n3). For a sparse tensor of shape (n, *), we use a single
    # accumulator of shape (n,).
    shape = grad.shape
    rank = len(shape)
    defaults = {'device': grad.device, 'dtype': grad.dtype}
    acc = {}

    if grad.is_sparse:
        acc[_key(0)] = torch.zeros(shape[0], **defaults)
    elif rank == 0:
        # The scalar case is handled separately
        acc[_key(0)] = torch.zeros(shape, **defaults)
    else:
        for i in range(rank):
            acc_shape = [1] * i + [shape[i]] + [1] * (rank - 1 - i)
            acc[_key(i)] = torch.zeros(acc_shape, **defaults)

    state.update(acc)


def _max_reduce_except_dim(tensor, dim):
    # Computes max along all dimensions except the given dim.
    # If tensor is a scalar, it returns tensor.
    rank = len(tensor.shape)
    result = tensor
    if rank > 0:
        assert dim < rank
        for d in range(rank):
            if d != dim:
                result = result.max(dim=d, keepdim=True).values
    return result
