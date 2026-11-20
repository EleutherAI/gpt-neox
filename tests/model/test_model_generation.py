# Copyright (c) 2025, EleutherAI
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

"""
instantiate models, save checkpoints, load checkpoints, compare loaded parameters to saved parameters and compare forward pass outputs

This tests contain a relatively large number of functions. They are not split into separate tests because a lot of boilerplate (e.g. instantiate model) needs
to run in order to perform follow up tests. Joining in one test reduces runtime at the expense of decreased transparency of test results in case of failures.
"""


import os
import json
from datetime import timedelta
from itertools import product
from unittest.mock import patch

import pytest
import torch
from tests.common import DistributedTest, model_setup, parametrize

PARAMS_TO_TEST = {
    "pipe_parallel_size,model_parallel_size,world_size": [
        [0, 1, 1],
        [0, 1, 2],
        [1, 2, 2],
        [0, 2, 2],
        [2, 1, 2],
    ],
    "top_p,temperature,top_k,min_p,top_h,typical_p,epsilon_cutoff,eta_cutoff": [
        [0.0, 0.5, 0, 0.1, 0.0, 0.9, 0.0, 0.001],
        [0.5, 0.0, 100, 0.0, 0.4, 1.0, 0.001, 0.0],
        [0.5, 0.5, 0, 0.1, 0.4, 0.9, 0.001, 0.001],
    ],
    "prompt": ["", "hello world"],
    "fp16,fp32_allreduce": [
        [
            {
                "enabled": True,
                "type": "bfloat16",
                "loss_scale": 0,
                "loss_scale_window": 1000,
                "hysteresis": 2,
                "min_loss_scale": 1,
            },
            True,
        ],
        [
            {
                "enabled": True,
                "loss_scale": 0,
                "loss_scale_window": 1000,
                "hysteresis": 2,
                "min_loss_scale": 1,
            },
            False,
        ],
    ],
}

parameters, names = parametrize(
    PARAMS_TO_TEST, max_tests=int(os.getenv("MAX_TESTCASES", 50)), seed=None
)


@pytest.mark.skip
@pytest.mark.parametrize("param_dict", parameters, ids=names)
def test_train(param_dict):
    t1 = run_generate_test_class()
    t1.run_generate_test(param_dict, param_dict.pop("prompt"))


class run_generate_test_class(DistributedTest):
    world_size = 2

    def run_generate_test(param_dict, prompt):
        from megatron.text_generation_utils import generate_samples_from_prompt
        from megatron.utils import is_mp_rank_0

        fixed_params = {
            "num_samples": 3,
            "maximum_tokens": 50,
            "minimum_tokens": 10,
            "make_vocab_size_divisible_by": 2,
            "sample_output_file": "test_sample_output.txt",
            "checkpoint_activations": False,
            "partition_activations": False,
            "no_load_optim": True,
        }

        param_dict.update(fixed_params)
        # TODO: we don't need to reinstantiate the model every time if we're only changing sampling settings - should be a workaround for this
        model, _, _, args_loaded = model_setup(None, param_dict, clear_data=True)
        model.eval()

        prompts = [prompt for _ in range(args_loaded.num_samples)]
        output = generate_samples_from_prompt(
            neox_args=args_loaded,
            model=model,
            text=prompts,
            maximum_tokens=args_loaded.maximum_tokens,
            minimum_tokens=args_loaded.minimum_tokens,
            recompute=False,
            temperature=args_loaded.temperature,
            top_k=args_loaded.top_k,
            top_p=args_loaded.top_p,
            min_p=args_loaded.min_p,
            top_h=args_loaded.top_h,
            typical_p=args_loaded.typical_p,
            epsilon_cutoff=args_loaded.epsilon_cutoff,
            eta_cutoff=args_loaded.eta_cutoff,
        )

        # Only the prompt owner returns records.
        owner = (
            torch.distributed.get_rank() == 0
            if args_loaded.is_pipe_parallel
            else is_mp_rank_0()
        )
        if owner:
            assert len(output) == len(prompts)
            for prompt, out in zip(prompts, output):
                assert prompt == out["context"]
                assert len(out["text"]) > 0
                assert out["length"] >= args_loaded.minimum_tokens
        else:
            assert output == []


def _fixed_generation_model(tp_size, pp_size, tied):
    from megatron import mpu
    from megatron.initialize import initialize_megatron
    from megatron.model.transformer import NormPipe, ParallelLinearPipe
    from megatron.model.word_embeddings import EmbeddingPipe
    from megatron.neox_arguments import NeoXArgs
    from megatron.training import setup_model_and_optimizer

    args = NeoXArgs.from_dict(
        dict(
            num_gpus=torch.distributed.get_world_size(),
            global_num_gpus=torch.distributed.get_world_size(),
            pipe_parallel_size=pp_size,
            model_parallel_size=tp_size,
            num_layers=2,
            hidden_size=32,
            num_attention_heads=4,
            seq_length=8,
            max_position_embeddings=10,
            no_weight_tying=not tied,
            norm="layernorm",
            pos_emb="learned",
            opt_pos_emb_offset=2 if tied else 0,
            precision="fp32",
            tokenizer_type="CharLevelTokenizer",
            no_load_optim=True,
            train_micro_batch_size_per_gpu=1,
            train_iters=1,
        )
    )
    args.configure_distributed_args()
    args.build_tokenizer()
    initialize_megatron(args)
    model, *_ = setup_model_and_optimizer(args, use_cache=True)
    # Identical global weights for all layouts. The winner is always 263 while
    # EOS is masked, but its score reveals whether cached position IDs restart.
    with torch.no_grad():
        for param in model.parameters():
            param.zero_()
        for layer in model.module.modules():
            if isinstance(layer, NormPipe):
                layer.norm.bias[0] = layer.norm.weight[1] = 1
            if isinstance(layer, EmbeddingPipe):
                pos = layer.position_embeddings.weight
                pos[::2, 1], pos[1::2, 1] = -1, 1
                if not tied:
                    continue
                weight = layer.word_embeddings.weight
            elif isinstance(layer, ParallelLinearPipe):
                weight = layer.final_linear.weight
            else:
                continue
            start = mpu.get_model_parallel_rank() * weight.size(0)
            for token, score in [(5, 2), (263, 3), (400, 10)]:
                if start <= token < start + weight.size(0):
                    weight[token - start, 0] = score
                    if token == 263:
                        weight[token - start, 1] = 0.05
    return args, model


def _check_pipeline_io(args, model, rank, tmpdir):
    from megatron.text_generation_utils import (
        generate_samples_input_from_file,
        generate_samples_interactive,
        generate_samples_unconditional,
    )

    source, dest = tmpdir / "prompts.txt", tmpdir / "out.jsonl"
    if rank == 0:
        source.write_text("AB\nABCDEFGH\nCD\n")
    torch.distributed.barrier()
    args.return_logits = True
    model.module.inference_mode(use_cache=True)
    for generate, opts in (
        (generate_samples_input_from_file, dict(input_file=source)),
        (generate_samples_unconditional, dict(number_of_samples=2)),
    ):
        # Count actual opens for writing, including opens that truncate a file.
        with patch("builtins.open", wraps=open) as opened:
            records = generate(
                args,
                model,
                output_file=dest,
                eos_token_id=400,
                maximum_tokens=4,
                **opts,
            )
        writers = [c for c in opened.call_args_list if c.args[:2] == (dest, "w")]
        assert len(writers) == (1 if rank == 0 else 0)
        if rank == 0:
            assert len(records) == (
                3 if generate is generate_samples_input_from_file else 2
            )
            assert [
                json.loads(line) for line in dest.read_text().splitlines()
            ] == records
            for record in records:
                if record["context"] == "ABCDEFGH":
                    assert record["logits"] == [] and not record["finished"]
                    continue
                assert record["finished"] and record["length"] == 0
                assert (
                    len(record["logits"]) == 1
                )  # only the EOS decision, not the budget
                assert record["logits"][0][400] == 10
        else:
            assert records == []
    # Only rank zero reads stdin; EOF must release all stages.
    replies = ["AB", EOFError()] if rank == 0 else []
    with patch("builtins.input", side_effect=replies) as stdin, patch("os.system"):
        generate_samples_interactive(args, model, maximum_tokens=2, eos_token_id=400)
    assert stdin.call_count == (2 if rank == 0 else 0)


def _check_parallel_generation(rank, tmpdir, target_pp, target_tp, world_size):
    import deepspeed
    from megatron import mpu
    from megatron.text_generation_utils import stream_tokens

    os.environ.update(LOCAL_RANK=str(rank), RANK=str(rank), WORLD_SIZE=str(world_size))
    torch.cuda.set_device(rank)
    deepspeed.init_distributed(
        dist_backend="nccl",
        auto_mpi_discovery=False,
        init_method=(tmpdir / "rendezvous").as_uri(),
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=90),
    )
    reference = {}
    try:
        for (pp_size, tp_size), tied in product(
            ((0, 1), (target_pp, target_tp)), (False, True)
        ):
            reference_layout = (pp_size, tp_size) == (0, 1)
            args, model = _fixed_generation_model(tp_size, pp_size, tied)
            sampler = (
                model.grid.stage_to_global(pp_size - 1, data=0, model=0)
                if pp_size
                else 0
            )
            for recompute, sample, scores in product((False, True), repeat=3):
                model.module.inference_mode(use_cache=not recompute)
                args.return_logits = scores
                # A non-source rank must not draw from its own RNG.
                torch.cuda.manual_seed(
                    123 if reference_layout or rank == sampler else 456
                )
                rng_before = torch.cuda.get_rng_state()
                options = dict(temperature=1.0, top_k=2, top_p=0.95) if sample else {}
                steps = list(
                    stream_tokens(
                        args,
                        model,
                        [[81, 90], [65, 66]] if not pp_size or rank == 0 else [],
                        eos_token_id=400,
                        minimum_tokens=3,
                        maximum_tokens=4,
                        recompute=recompute,
                        **options,
                    )
                )
                tokens, _, _, logits, done = steps[-1]
                ids = tokens[:, 2:6].tolist()
                assert len(steps) == 4 and done.all()
                assert all(row[-1] == 400 for row in ids)
                if not sample:
                    assert ids == [[263, 263, 263, 400]] * 2
                if not reference_layout and rank != sampler:
                    assert torch.equal(torch.cuda.get_rng_state(), rng_before)
                key = (tied, recompute, sample, scores)
                if not scores:
                    assert logits is None
                if reference_layout:
                    reference[key] = (ids, logits.cpu() if scores else None)
                else:
                    expected_ids, expected_logits = reference[key]
                    assert ids == expected_ids
                    if scores:
                        assert logits.shape == (4, 512)
                        torch.testing.assert_close(logits.cpu(), expected_logits)
                        if not sample:
                            assert logits[0, 263] != logits[1, 263]
            if pp_size and not tied:
                _check_pipeline_io(args, model, rank, tmpdir)
            mpu.destroy_model_parallel()
    finally:
        torch.distributed.destroy_process_group()


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two CUDA devices")
def test_non_pipeline_tp_generation_matches_single_rank(tmp_path):
    torch.multiprocessing.spawn(
        _check_parallel_generation, args=(tmp_path, 0, 2, 2), nprocs=2
    )


@pytest.mark.parametrize("pp,tp,dp", [(2, 1, 1), (1, 2, 1), (1, 1, 2), (2, 2, 1)])
def test_pipeline_generation_matches_single_rank(tmp_path, pp, tp, dp):
    world_size = pp * tp * dp
    if torch.cuda.device_count() < world_size:
        pytest.skip(f"requires {world_size} CUDA devices")
    torch.multiprocessing.spawn(
        _check_parallel_generation,
        args=(tmp_path, pp, tp, world_size),
        nprocs=world_size,
    )
