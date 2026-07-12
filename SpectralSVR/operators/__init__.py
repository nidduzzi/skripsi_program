"""Operator baselines: compare SpectralSVR against neural operators (DeepONet,
and later FNO/SNO/NSM) on the same problems, scored with the same metrics."""

from .__base import Operator as Operator
from .benchmark import benchmark as benchmark
from .dataset import (
    OperatorDataset as OperatorDataset,
    sensor_grid as sensor_grid,
)
from .deeponet import DeepONetOperator as DeepONetOperator
from .spectralsvr import SpectralSVROperator as SpectralSVROperator
