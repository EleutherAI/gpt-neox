from oslo.pytorch.model_parallelism.utils.mappings import (
    Column,
    Row,
    Update,
)
from oslo.pytorch.model_parallelism.utils.mappings import (
    TensorParallelismMapping as _TensorParallelismMapping,
)
import copy
import importlib


class TensorParallelismMapping(_TensorParallelismMapping):
    __MAPPING__ = dict(
        GPT2ModelPipe=[
            Column("mlp.dense_h_to_4h", "final_linear"),
            Columne("query_key_value", combined_qkv=True)
            Row("mlp.dense_4h_to_h"),
            Update("num_attention_heads", "all_head_size"),
        ],
    )

    @staticmethod
    def _load_class_by_model_name(model_name):
        """
        Load base class obj by class name
        Args:
            model_name (str): model name (e.g. Bert, GPT2, T5, ...)
        Returns:
            class: GPT2ModelPipe
        """
        assert (
            model_name == "GPT2ModelPipe"
        ), "Currently, only GPT2ModelPipe is supported"
        from .gpt2_model import GPT2ModelPipe

        return GPT2ModelPipe
