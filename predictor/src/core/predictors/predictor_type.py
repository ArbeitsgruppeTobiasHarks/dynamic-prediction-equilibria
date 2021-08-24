from enum import Enum


class PredictorType(Enum):
    ZERO = 0
    CONSTANT = 1
    LINEAR = 2
    REGULARIZED_LINEAR = 3
    MACHINE_LEARNING = 4
