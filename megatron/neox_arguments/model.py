from dataclasses import dataclass


@dataclass
class NeoXArgsModel:
    num_layers: int = None
    """
    Number of transformer layers.
    """

    hidden_size: int = None
    """
    Transformer hidden size.
    """

    num_attention_heads: int = None
    """
    Number of transformer attention heads.
    """

    seq_length: int = None
    """
    Maximum sequence length to process.
    """

    max_position_embeddings: int = None
    """
    Maximum number of position embeddings to use. This is the size of position embedding.
    """

    norm: str = "layernorm"
    """
    Normalization layer to use. Choose from "layernorm", "rmsnorm" and "scalenorm".
    """

    layernorm_epsilon: float = 1e-05
    """
    Layer norm epsilon.
    """

    rms_norm_epsilon: float = 1e-8
    """
    Root mean squared norm epsilon
    """

    scalenorm_epsilon: float = 1e-8
    """
    Scalenorm epsilon
    """

    pos_emb: str = "learned"
    """
    Type of positional embedding to use - choose from 'learned', 'sinusoidal', 'rpe', 'none'
    """

    rpe_num_buckets: int = 32
    """
    T5 relative positional encoding number of buckets, default 32.
    """

    rpe_max_distance: int = 128
    """
    T5 relative positional encoding max distance, default 128.
    """

    no_weight_tying: bool = False
    """
    Disables weight tying between embedding weights and final Linear layer
    """

    geglu: bool = False
    """
    Enable geglu activation function (WARNING: will increase memory usage, adjust embd dims accordingly)
    """

    sparsity: str = "none"
    """
    Sparse attention layer configuration: none = all regular attn, all = all sparse attn, interspersed = sparse on odd layers, dense on even.
    """

    num_unique_layers: int = None
    """
    Number of unique transformer layers. num-layers should be divisible by this value. Currently only has an effect when pipe_parallel_size=0.
    """

    param_sharing_style: str = "grouped"
    """
    Ordering of the shared parameters. For example, for a num-layers=4 and --num-unique-layers=2, we will have the following ordering for two unique layers 1 and 2-: grouped: [1, 2, 1, 2] and spaced: [1, 1, 2, 2].
    """

    make_vocab_size_divisible_by: int = 128
    """
    Pad the vocab size to be divisible by this value. This is added for computational efficiency reasons.
    """

    apply_residual_connection_post_layernorm: bool = False
    """
    If set, use original BERT residual connection ordering.
    """

    openai_gelu: bool = False
    """
    Use OpenAIs GeLU implementation. This option should not be used unless for backward compatibility reasons.
    """

    scaled_upper_triang_masked_softmax_fusion: bool = False
    """
    Enable fusion of query_key_value_scaling time (upper diagonal) masking and softmax.
    """

    scaled_masked_softmax_fusion: bool = False
    """
    Enable fusion of query_key_value_scaling general masking and softmax.
    """

    bias_gelu_fusion: bool = False
    """
    Enable bias and gelu fusion.
    """

    bias_dropout_fusion: bool = False
    """
    Enable bias and dropout fusion.
    """

    fp16_lm_cross_entropy: bool = False
    """
    Move the cross entropy unreduced loss calculation for lm head to fp16.
    """

    init_method_std: float = 0.02
    """
    Standard deviation of the zero mean normal distribution used for weight initialization.
    """

    apply_query_key_layer_scaling: bool = False
    """
    Scale Q * K^T by 1 / layer-number. If this flag is set, then it will automatically set attention-softmax-in-fp32 to true
    """

    use_cpu_initialization: bool = False
    """
    If set, affine parallel weights initialization uses CPU
    """

    attention_softmax_in_fp32: bool = False
    """
    Run attention masking and softmax in fp32.
    """

    fp32_allreduce: bool = False
    """
    All-reduce in fp32
    """
    