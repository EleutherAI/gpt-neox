# Copyright (c) 2026, EleutherAI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import multiprocessing
import os
import socket
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy

import pytest
import torch


def _logit_change(
    topology, parameterization, hidden_size, num_layers, steps=3, lr=1e-2
):
    from megatron.model import GPT2ModelPipe
    from megatron.model.parameterization import get_param_groups
    from megatron.neox_arguments import NeoXArgs
    from tests.common import BASE_CONFIG

    config = deepcopy(BASE_CONFIG)
    config.update(
        global_num_gpus=1,
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_attention_heads=4,
        seq_length=32,
        max_position_embeddings=32,
        pipe_parallel_size=0,
        use_cpu_initialization=True,
        precision="fp32",
        checkpoint_activations=False,
        partition_activations=False,
        tokenizer_type="CharLevelTokenizer",
        attention_config=[[["global"], "all"]],
        scaled_upper_triang_masked_softmax_fusion=False,
        bias_gelu_fusion=False,
        rope_fusion=False,
        layernorm_fusion=False,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        init_method="normal",
        output_layer_init_method="normal",
        weight_decay=0.0,
        optimizer={"type": "adam", "params": {"lr": lr}},
        parameterization=parameterization,
        mup_base_hidden_size=64,
        mup_base_num_layers=2,
    )
    args = NeoXArgs.from_dict(config)
    args.build_tokenizer()
    torch.manual_seed(0)
    model = GPT2ModelPipe(neox_args=args, topology=topology).to_sequential()
    optimizer = torch.optim.AdamW(
        [dict(g, lr=lr * g.get("lr_scale", 1.0)) for g in get_param_groups(model, args)]
    )

    tokens = torch.randint(0, 256, (8, 33), generator=torch.Generator().manual_seed(1))
    inputs = (
        tokens[:, :-1],
        torch.arange(32).expand(8, -1),
        torch.ones(1, 1, 32, 32).tril() < 0.5,
    )
    before = model(inputs).detach()
    for _ in range(steps):
        logits = model(inputs)
        loss = torch.nn.functional.cross_entropy(
            logits.flatten(0, 1), tokens[:, 1:].flatten()
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return (model(inputs).detach() - before).abs().mean().item()


def _run(runs):
    import deepspeed
    from deepspeed.runtime.activation_checkpointing import checkpointing
    from deepspeed.runtime.pipe.topology import PipeModelDataParallelTopology
    from megatron import mpu

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    os.environ.update(
        MASTER_ADDR="127.0.0.1",
        MASTER_PORT=str(port),
        RANK="0",
        LOCAL_RANK="0",
        WORLD_SIZE="1",
    )
    deepspeed.init_distributed(dist_backend="gloo", auto_mpi_discovery=False)
    topology = PipeModelDataParallelTopology(num_pp=1, num_mp=1, num_dp=1)
    mpu.initialize_model_parallel(1, topology=topology)
    checkpointing.mpu = mpu
    mpu.model_parallel_cuda_manual_seed(1234)
    return [_logit_change(topology, *run) for run in runs]


@pytest.mark.cpu
def test_logit_change_transfers_across_width_and_depth():
    runs = [
        ("sp", 64, 2),
        ("sp", 512, 2),
        ("mup", 64, 2),
        ("mup", 512, 2),
        ("completep", 64, 2),
        ("completep", 64, 16),
    ]
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(1, mp_context=context) as pool:
        sp_64, sp_512, mup_64, mup_512, completep_2, completep_16 = pool.submit(
            _run, runs
        ).result()

    assert sp_512 / sp_64 > 3
    assert 0.5 < mup_512 / mup_64 < 2
    assert 0.5 < completep_16 / completep_2 < 2
