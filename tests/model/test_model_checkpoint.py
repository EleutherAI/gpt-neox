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
import shutil
import torch

import pytest
from tests.common import (
    DistributedTest,
    clear_test_dirs,
    model_setup,
    binary,
    parametrize,
)
import torch

PARAMS_TO_TEST = {
    "pipe_parallel_size,model_parallel_size": [[0, 1], [1, 2], [0, 2], [2, 1]],
    "checkpoint_validation_with_forward_pass": [True],
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
    import tempfile

    d = tempfile.mkdtemp()
    param_dict["save"] = d

    t1 = test_run_checkpoint_test_class()
    t1.run_checkpoint_test(param_dict=param_dict)


class test_run_checkpoint_test_class(DistributedTest):
    def run_checkpoint_test(yaml_list=None, param_dict=None):

        from megatron.checkpointing import load_checkpoint
        from megatron.checkpointing import save_checkpoint

        model, optimizer, lr_scheduler, args_loaded = model_setup(
            yaml_list, param_dict, clear_data=True
        )

        # save model checkpoint
        save_checkpoint(
            neox_args=args_loaded,
            iteration=42,
            model=model,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
        )

        # reload model from checkpoint
        (
            reloaded_model,
            reloaded_optimizer,
            reloaded_lr_scheduler,
            args_reloaded,
        ) = model_setup(yaml_list, param_dict, clear_data=False)
        iteration = load_checkpoint(
            neox_args=args_reloaded,
            model=reloaded_model,
            optimizer=reloaded_optimizer,
            lr_scheduler=reloaded_lr_scheduler,
        )

        # ensure same checkpoint is loaded
        assert (
            iteration == 42
        ), "run_checkpoint_test() iteration loaded from checkpoint correct"

        # check all weight groups are the same
        for idx, ((n1, p1), (n2, p2)) in enumerate(
            zip(
                list(model.module.named_parameters()),
                list(reloaded_model.module.named_parameters()),
            )
        ):
            assert n1 == n2
            params_equal = (p1 == p2).all().item()
            assert params_equal, "run_checkpoint_test() params equal: " + str(n1)


if __name__ == "__main__":
    params = list(
        parametrize(
            PARAMS_TO_TEST, max_tests=int(os.getenv("MAX_TESTCASES", 50)), seed=None
        )
    )
    test_train(params[0])


class TestEmbeddingLoadStateDictVocabResize:
    """A checkpoint's word_embeddings.weight vocab size can differ from the model's
    padded vocab size (see issue #645: a slim/unpadded 25216-row checkpoint loaded
    into a model padded to 50304 rows raised a hard state_dict size mismatch).
    Embedding._load_from_state_dict must resize instead of raising.
    """

    def setup_method(self):
        import torch.distributed as dist
        from megatron import mpu

        if not dist.is_initialized():
            dist.init_process_group(
                backend="gloo", init_method="tcp://127.0.0.1:29505", rank=0, world_size=1
            )
        mpu.destroy_model_parallel()
        mpu.initialize_model_parallel(1)

    def teardown_method(self):
        from megatron import mpu

        mpu.destroy_model_parallel()

    def _make_embedding(self, vocab_size, hidden_size=8):
        import types
        from megatron.model.word_embeddings import Embedding
        from megatron.model.init_functions import init_method_normal

        neox_args = types.SimpleNamespace(
            sequence_parallel=False,
            use_mup=False,
            mup_embedding_mult=1.0,
            mup_rp_embedding_mult=1.0,
            use_bnb_optimizer=False,
            use_cpu_initialization=True,
            params_dtype=torch.float32,
            model_parallel_size=1,
        )
        return Embedding(
            neox_args=neox_args,
            hidden_size=hidden_size,
            vocab_size=vocab_size,
            max_sequence_length=16,
            embedding_dropout_prob=0.0,
            init_method=init_method_normal(0.02),
            use_pos_emb=False,
        )

    def test_loads_smaller_checkpoint_vocab_into_larger_padded_model(self):
        checkpoint_vocab_size = 25216
        padded_vocab_size = 50304

        checkpoint_embedding = self._make_embedding(checkpoint_vocab_size)
        checkpoint_state = checkpoint_embedding.state_dict()

        model_embedding = self._make_embedding(padded_vocab_size)
        original_padding_rows = model_embedding.word_embeddings.weight.data[
            checkpoint_vocab_size:
        ].clone()

        # Must not raise a state_dict size mismatch.
        model_embedding.load_state_dict(checkpoint_state, strict=True)

        loaded = model_embedding.word_embeddings.weight.data
        assert torch.equal(
            loaded[:checkpoint_vocab_size],
            checkpoint_state["word_embeddings.weight"],
        ), "checkpoint rows must load exactly"
        assert torch.equal(
            loaded[checkpoint_vocab_size:], original_padding_rows
        ), "rows beyond the checkpoint's vocab must keep the model's own initialized values"

    def test_loads_larger_checkpoint_vocab_into_smaller_model_by_truncating(self):
        checkpoint_vocab_size = 50304
        smaller_vocab_size = 25216

        checkpoint_embedding = self._make_embedding(checkpoint_vocab_size)
        checkpoint_state = checkpoint_embedding.state_dict()

        model_embedding = self._make_embedding(smaller_vocab_size)
        model_embedding.load_state_dict(checkpoint_state, strict=True)

        loaded = model_embedding.word_embeddings.weight.data
        assert torch.equal(
            loaded, checkpoint_state["word_embeddings.weight"][:smaller_vocab_size]
        )
