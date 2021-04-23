from dataclasses import dataclass

@dataclass
class NeoXArgsTextgen:

    text_gen_type: str = None
    """
    How to generate text/sample the model.
    Options: `unconditional`, `input-file`, `interactive`
    """

    temperature: float = 1.0
    """
    Sampling temperature.
    """

    greedy: bool = False
    """
    Use greedy sampling.
    """

    top_p: float = 0.0
    """
    Top p sampling.
    """

    top_k: int = 0
    """
    Top k sampling.
    """

    out_seq_length: int = 1024
    """
    Size of the output generated text.'
    """

    sample_input_file: str = None
    """
    Get input from file instead of interactive mode, each line is an input.
    """

    sample_output_file: str = None
    """
    Output file got from --sample-input-file
    """

    num_samples: int = 0
    """
    Number of samples to generate unconditionally, defaults to 0 and interactive conditional sampling
    """

    genfile: str = None
    """
    Output file when generating unconditionally
    """

    recompute: bool = False
    """
    During generation recompute all attention instead of using previously computed keys/values.
    """
