###############
# pseudo code #
###############

# scenario:
# you have a grid of 24 gpus, and three types of communication operation
# mp - 4x all reduces per layer (2 in forward, 2 in backward) (*probably* slowest?)
# pp - 2x communications per layer (*probably* fastest)
# dp - 1x all reduce per batch (probably slow, but can be compressed)
# 3 types of connection, 1 fast, 1 slow, one slowest, all with different numbers of connections
# you need to lay out all gpus to schedule group the nodes in groups, and allocate connection types to that group.

from itertools import product, combinations
import numpy as np


def divisors(n):
    # get factors and their counts
    factors = {}
    nn = n
    i = 2
    while i * i <= nn:
        while nn % i == 0:
            factors[i] = factors.get(i, 0) + 1
            nn //= i
        i += 1
    if nn > 1:
        factors[nn] = factors.get(nn, 0) + 1

    primes = list(factors.keys())

    # generates factors from primes[k:] subset
    def generate(k):
        if k == len(primes):
            yield 1
        else:
            rest = generate(k + 1)
            prime = primes[k]
            for factor in rest:
                prime_to_i = 1
                # prime_to_i iterates prime**i values, i being all possible exponents
                for _ in range(factors[prime] + 1):
                    yield factor * prime_to_i
                    prime_to_i *= prime

    # in python3, `yield from generate(0)` would also work
    for factor in generate(0):
        yield factor


class Topology:

    def __init__(self, num_nodes, num_axes=3):
        self.num_nodes = num_nodes
        self.num_axes = num_axes
        self.mappings = self._all_possible_mappings(num_nodes, num_axes)

    @staticmethod
    def _all_possible_mappings(nodes, axes):
        for i in product(divisors(nodes), repeat=axes):
            if np.prod(i) == nodes:
                yield i

    @staticmethod
    def is_possible_mapping(iterable, n):
        return np.prod(iterable) == n and all([m > 0 for m in iterable])

    def fix_axis(self, n):
        # fixes an axis to size n - at least one in the grid must have size n
        assert self.num_nodes % n == 0  # num nodes must be divisible by n
        _mappings = self._all_possible_mappings(self.num_nodes // n, self.num_axes - 1)
        self.mappings = [list(i) + [n] for i in _mappings]


# t = Topology(24, 3)
# t.fix_axis(6)
# print(t.mappings)

# pipeline is the only sequential parallelism operation
# both mp and pp are all doing an allreduce
# the nvlink pairs are not really fit for all reduce ops because if you have a group size larger than 2,
# the slowest communicator will be the bottleneck for the whole group
# we can try
# a) scheduling the pipeline to be across the fast connections, and using onebit adam across the slow ones
# b) using zero 3 and hoping

# so now we have a sub-problem
from deepspeed.runtime.pipe.topology import PipeModelDataParallelTopology, PipelineParallelGrid


p = PipeModelDataParallelTopology(6, 2, 2)
print(p.get_axis_comm_lists('model'))
# class CustomTopology(ProcessTopology):
#     """ A topology to schedule pipeline parallelism across fast, intranode connections, and data/model parallel
#     (ideally compressed) across slow, inter/intra node connections.
#
#     Recommended that you *don't* use model parallel with this topology, although it is possible."""
#
#     def __init__(self, num_dp, num_pp, gpus_per_node, num_mp=1):
#         assert num_pp == gpus_per_node
#         super().__init__(axes=['data', 'pipe', 'model'], dims=[num_dp, num_pp, num_mp])

#
# p = CustomTopology(4, 6, 6)
#
# # print(p.get_axis_comm_lists('pipe'))
#
#
# class CustomGrid(PipelineParallelGrid):
#     """
#     A custom grid that overwrites DeepSpeed's PipelineParallelGrid._build_p2p_groups() to be able to take a custom
#     DAG rather than a strictly sequential p2p group.
#     """
#
#     def __init__(self, topology=None, world_size=5, process_group=None):
#         # TODO use process_group if provided
#         self.global_rank = 0
#         self.world_size = world_size
#         if topology is not None:
#             if self.global_rank == 0:
#                 print('Using topology:', topology)
#             self._topo = topology
#         self.data_parallel_size = max(self._topo.get_dim('data'), 1)
#         self.pipe_parallel_size = max(self._topo.get_dim('pipe'), 1)
#         self.model_parallel_size = max(self._topo.get_dim('model'), 1)
#         # assert self._is_grid_valid(), "Invalid Grid"
#
#         self.stage_id = self.get_stage_id()
#         self.data_parallel_id = self.get_data_parallel_id()
#
#         # Create new ProcessGroups for all model parallelism. DeepSpeedLight uses these
#         # to detect overflow, etc.
#         self.ds_model_proc_group = None
#         self.ds_model_rank = -1
#         for dp in range(self.data_parallel_size):
#             ranks = sorted(self._topo.get_axis_list(axis='data', idx=dp))
#             if self.global_rank == 0:
#                 # print(f'RANK={self.global_rank} building DeepSpeed model group: {ranks}')
#                 pass
#             # proc_group = dist.new_group(ranks=ranks)
#             # if self.global_rank in ranks:
#             #     self.ds_model_proc_group = proc_group
#             #     self.ds_model_world_size = len(ranks)
#             #     self.ds_model_rank = ranks.index(self.global_rank)
#         # assert self.ds_model_rank > -1
#         # assert self.ds_model_proc_group is not None
#
#         # Create new ProcessGroup for gradient all-reduces - these are the data parallel groups
#         self.dp_group = []
#         self.dp_groups = self._topo.get_axis_comm_lists('data')
#         # for g in self.dp_groups:
#         #     proc_group = dist.new_group(ranks=g)
#         #     if self.global_rank in g:
#         #         self.dp_group = g
#         #         self.dp_proc_group = proc_group
#
#         self.is_first_stage = (self.stage_id == 0)
#         self.is_last_stage = (self.stage_id == (self.pipe_parallel_size - 1))
#
#         self.p2p_groups = self._build_p2p_groups()
#
#         # Create new ProcessGroup for pipeline collectives - these are pipe parallel groups
#         self.pp_group = []
#         # self.pp_proc_group = None
#         self.pipe_groups = self._topo.get_axis_comm_lists('pipe')
#         # for ranks in self.pipe_groups:
#         #     if self.global_rank == 0:
#         #         # print(f'RANK={self.global_rank} building pipeline group: {ranks}')
#         #         pass
#         #     proc_group = dist.new_group(ranks=ranks)
#         #     if self.global_rank in ranks:
#         #         self.pp_group = ranks
#         #         self.pp_proc_group = proc_group
#         # assert self.pp_proc_group is not None
#
#         # Create new ProcessGroup for model (tensor-slicing) collectives
#
#         # Short circuit case without model parallelism.
#         # TODO: it would be nice if topology had bcast semantics to avoid this branching
#         # case?
#         # if self.model_parallel_size == 1:
#         #     for group_rank in range(self.world_size):
#         #         group_rank = [group_rank]
#         #         group = dist.new_group(ranks=group_rank)
#         #         if group_rank[0] == self.global_rank:
#         #             self.slice_group = group_rank
#         #             self.slice_proc_group = group
#         #     return
#         # else:
#         #     self.mp_group = []
#         #     self.model_groups = self._topo.get_axis_comm_lists('model')
#         #     for g in self.model_groups:
#         #         proc_group = dist.new_group(ranks=g)
#         #         if self.global_rank in g:
#         #             self.slice_group = g
#         #             self.slice_proc_group = proc_group
#
#     def _build_p2p_groups(self, custom_order=None):
#         """Groups for sending and receiving activations and gradients across model
#         parallel stages.
#
#         """
#         if custom_order is not None:
#             assert len(custom_order) == self.world_size
#             # assert every node has been visited twice, aside from the first and last
#             flattened = [item for sublist in custom_order for item in sublist]
#             flat_set = set(flattened[1:-1])
#             assert len(flat_set) == (len(flattened[1:-1]) / 2)
#             assert all([flattened[0] not in flat_set, flattened[-1] not in flat_set])
#             return custom_order
#
#         comm_lists = self._topo.get_axis_comm_lists('pipe')
#         p2p_lists = []
#         for rank in range(self.world_size):
#             for l in comm_lists:
#                 assert len(l) == self.pipe_parallel_size
#                 if rank in l:
#                     idx = l.index(rank)
#                     buddy_rank = l[(idx + 1) % self.pipe_parallel_size]
#                     p2p_lists.append([rank, buddy_rank])
#                     break  # next global rank
#         assert len(p2p_lists) == self.world_size
#         return p2p_lists
#
#
# g = CustomGrid(p)
#
# print(g._build_p2p_groups(custom_order=[[0, 1], [1, 2], [2, 4], [4, 3], [3, 5]]))

