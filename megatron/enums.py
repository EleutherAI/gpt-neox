import enum

class AttnMaskType(enum.Enum):
    padding = 1
    causal = 2
    prefix = 3