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

from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from megatron.data import data_utils, gpt2_dataset, indexed_dataset
from megatron.neox_arguments.neox_args import NeoXArgsTraining


pytestmark = pytest.mark.cpu


@pytest.fixture
def single_process(monkeypatch):
    """Exercise real mmap data and C++ indices without a CUDA process group."""
    monkeypatch.setattr(gpt2_dataset.torch.distributed, "get_rank", lambda: 0)
    monkeypatch.setattr(
        gpt2_dataset.torch.distributed, "get_world_size", lambda **kwargs: 1
    )
    monkeypatch.setattr(
        gpt2_dataset.torch.distributed, "all_reduce", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(
        gpt2_dataset.torch.cuda, "LongTensor", lambda values: np.array(values)
    )
    monkeypatch.setattr(gpt2_dataset.mpu, "get_io_parallel_group", lambda: None)


@pytest.fixture(scope="module")
def index_helpers():
    try:
        from megatron.data import helpers  # noqa: F401
    except ImportError:
        data_utils.compile_helper()


@pytest.fixture
def ordered_corpus(tmp_path):
    prefix = str(tmp_path / "ordered_text_document")
    documents = [
        np.arange(10, 15, dtype=np.int32),
        np.arange(20, 29, dtype=np.int32),
        np.arange(30, 34, dtype=np.int32),
        np.arange(40, 51, dtype=np.int32),
    ]
    builder = indexed_dataset.make_builder(prefix + ".bin", "mmap")
    for document in documents:
        builder.add_item(document)
        builder.end_document()
    builder.finalize(prefix + ".idx")
    return prefix, documents


def make_dataset(prefix, document_ids, preserve_data_order, seed=123, num_epochs=2):
    return gpt2_dataset.GPT2Dataset(
        name="train",
        data_prefix=prefix,
        documents=np.array(document_ids, dtype=np.int32),
        indexed_dataset=indexed_dataset.make_dataset(prefix, "mmap", skip_warmup=True),
        num_samples=8,
        num_epochs=num_epochs,
        seq_length=4,
        seed=seed,
        preserve_data_order=preserve_data_order,
    )


@pytest.mark.parametrize("seed", [0, 123])
@pytest.mark.parametrize("num_epochs", [None, 1, 2])
def test_ordered_samples_follow_source_and_repeat_epochs(
    ordered_corpus, single_process, index_helpers, seed, num_epochs
):
    prefix, documents = ordered_corpus
    # Nonzero IDs also cover the document subsets used by train/valid/test splits.
    document_ids = [1, 3]
    dataset = make_dataset(prefix, document_ids, True, seed, num_epochs)
    expected_ids = document_ids * (num_epochs if num_epochs is not None else 2)
    np.testing.assert_array_equal(dataset.doc_idx, expected_ids)
    np.testing.assert_array_equal(
        dataset.shuffle_idx, np.arange(len(dataset.shuffle_idx))
    )
    stream = np.concatenate([documents[i] for i in expected_ids])
    for i in range(len(dataset)):
        np.testing.assert_array_equal(dataset[i]["text"], stream[4 * i : 4 * i + 5])


def test_ordered_and_shuffled_caches_are_independent(
    ordered_corpus, single_process, index_helpers, tmp_path
):
    prefix, _ = ordered_corpus
    shuffled = make_dataset(prefix, [0, 1, 2, 3], False)
    ordered = make_dataset(prefix, [0, 1, 2, 3], True)
    assert not np.array_equal(shuffled.doc_idx, ordered.doc_idx)
    assert len(list(tmp_path.glob("*_ordered_*.npy"))) == 3
    assert len(list(tmp_path.glob("*.npy"))) == 6

    # Both modes should load their own cached maps on subsequent runs.
    with patch.object(gpt2_dataset, "_build_doc_idx", side_effect=AssertionError):
        for preserve, expected in [(True, ordered), (False, shuffled)]:
            actual = make_dataset(prefix, [0, 1, 2, 3], preserve)
            for name in ["doc_idx", "sample_idx", "shuffle_idx"]:
                np.testing.assert_array_equal(
                    getattr(actual, name), getattr(expected, name)
                )


def test_default_shuffle_matches_legacy_rng_sequence():
    documents = np.arange(12, dtype=np.int32)
    expected_rng = np.random.RandomState(42)
    expected_docs = np.tile(documents, 3)
    expected_rng.shuffle(expected_docs)
    expected_samples = np.arange(25, dtype=np.uint32)
    expected_rng.shuffle(expected_samples)

    actual_rng = np.random.RandomState(42)
    np.testing.assert_array_equal(
        gpt2_dataset._build_doc_idx(documents, 3, actual_rng), expected_docs
    )
    np.testing.assert_array_equal(
        gpt2_dataset._build_shuffle_idx(25, actual_rng), expected_samples
    )


def test_split_builder_only_preserves_training_order(ordered_corpus):
    prefix, _ = ordered_corpus
    with patch.object(data_utils, "GPT2Dataset") as dataset_class:
        data_utils.build_train_valid_test_datasets(
            data_prefix=prefix,
            use_shared_fs=True,
            data_impl="mmap",
            pack_impl="packed",
            allow_chopped=True,
            splits_string="2,1,1",
            train_valid_test_num_samples=[4, 2, 2],
            train_valid_test_epochs=[None, None, None],
            seq_length=4,
            seed=123,
            skip_warmup=True,
            preserve_data_order=True,
        )
    calls = dataset_class.call_args_list
    assert [call.kwargs["preserve_data_order"] for call in calls] == [
        True,
        False,
        False,
    ]
    for call, expected_ids in zip(calls, [[0, 1], [2], [3]]):
        np.testing.assert_array_equal(call.args[2], expected_ids)


def test_explicit_paths_only_preserve_training_order():
    args = NeoXArgsTraining()
    args.preserve_data_order = True
    args.train_data_paths = ["train"]
    args.valid_data_paths = ["valid"]
    args.test_data_paths = ["test"]
    args.seed = 123
    args.seq_length = 4
    args.precompute_model_name = None
    with patch.object(data_utils, "build_the_dataset") as build:
        data_utils.build_weighted_datasets(args, [8], [2], [2], None, None, None)
    calls = build.call_args_list
    assert [call.kwargs.get("preserve_data_order", False) for call in calls] == [
        True,
        False,
        False,
    ]


def test_single_dataset_builder_passes_order_flag(ordered_corpus):
    prefix, _ = ordered_corpus
    with patch.object(data_utils, "GPT2Dataset") as dataset_class:
        data_utils.build_the_dataset(
            data_prefix=prefix,
            pos_data_prefix=None,
            neg_data_prefix=None,
            name="train_0",
            data_impl="mmap",
            pack_impl="packed",
            dataset_impl="gpt2",
            allow_chopped=True,
            num_samples=8,
            num_epochs=None,
            seq_length=4,
            seed=123,
            skip_warmup=True,
            preserve_data_order=True,
        )
    assert dataset_class.call_args.kwargs["preserve_data_order"] is True


@pytest.mark.parametrize("train_paths", [None, ["train"]])
def test_preserve_order_accepts_single_training_stream(train_paths):
    args = NeoXArgsTraining()
    assert args.preserve_data_order is False
    args.preserve_data_order = True
    args.train_data_paths = train_paths
    data_utils.validate_data_order(args)


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"dataset_impl": "pairwise"}, "dataset_impl='gpt2'"),
        ({"dataset_impl": "online"}, "dataset_impl='gpt2'"),
        ({"pack_impl": "unpacked"}, "pack_impl='packed'"),
        ({"pack_impl": "pack_until_overflow"}, "pack_impl='packed'"),
        ({"train_data_paths": ["a", "b"]}, "single training dataset"),
    ],
)
def test_preserve_order_rejects_incompatible_configs(overrides, message):
    args = NeoXArgsTraining()
    args.preserve_data_order = True
    for name, value in overrides.items():
        setattr(args, name, value)
    with pytest.raises(ValueError, match=message):
        data_utils.validate_data_order(args)


def test_disabled_flag_keeps_existing_configs_valid():
    data_utils.validate_data_order(
        SimpleNamespace(
            preserve_data_order=False,
            dataset_impl="pairwise",
            pack_impl="unpacked",
            train_data_paths=["a", "b"],
        )
    )


def test_order_validation_runs_before_loading_data():
    args = NeoXArgsTraining()
    args.train_iters = 1
    args.preserve_data_order = True
    args.train_data_paths = ["a", "b"]
    with pytest.raises(ValueError, match="single training dataset"):
        data_utils.build_train_valid_test_data_loaders(args)


@pytest.mark.parametrize("packing_impl", ["unpacked", "pack_until_overflow"])
def test_index_builder_rejects_unsupported_ordered_packing(packing_impl):
    with pytest.raises(ValueError, match="pack_impl='packed'"):
        gpt2_dataset._build_index_mappings(
            "train",
            "unused",
            None,
            None,
            None,
            1,
            1,
            4,
            123,
            packing_impl,
            preserve_data_order=True,
        )
