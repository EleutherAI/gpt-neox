from dataclasses import dataclass

@dataclass
class NeoXArgsDeepspeedConfig:
    train_batch_size: int = None
