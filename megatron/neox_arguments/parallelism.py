from dataclasses import dataclass

@dataclass
class NeoXArgsParallelism:
    
    pipe_parallel_size: int = 0
    """
    Number of pipeline parallel stages. Disable with 0.
    """
    
    model_parallel_size: int = 1
    """
    Size of the model parallelism.
    """

    pipe_partition_method: str = "type:transformer"
    """
    method used to distribute model layers across pipeline stages. Choose from "parameters", which balances the number of parameters on each pipeline stage, "uniform", which naively balances the number of layers per stage, or "type:[regex]" (in our case this will basically only be "type:transformer"), which balances layers whose class names match [regex]
    """