import logging
from typing_extensions import Self, override

import torch
from torch import nn
import numpy as np
from torch.utils.data import TensorDataset, DataLoader

from .__base import (
    MultiRegression,
)
from torchmetrics.functional import mean_squared_error

logger = logging.getLogger(__name__)


class FNN(MultiRegression):
    """A class GPU variation that implements the Feed-forward Neural Network for regression tasks


    # Parameters:

    # Attributes:
    - All hyperparameters of section "Parameters".
    """

    def __init__(
        self,
        MAX_EPOCH: int = 100,
        lr: float = 0.001,
        batch_size: int = 4,
        activation: type[nn.Module] = nn.Softplus,
        n_hidden: int = 3,  # number of hidden layers
        w_hidden: int = 100,  # width of hidden layers
        dtype: torch.dtype = torch.float32,
        device: torch.device | None = None,
        logger: logging.Logger | None = None,
    ):
        if device is None:
            device = torch.device("cpu")
        super().__init__(dtype, device, logger)

        # Hyperparameters
        self.batch_size: int = batch_size
        self.MAX_EPOCH: int = MAX_EPOCH
        self.lr: float = lr
        self.n_hidden: int = n_hidden
        self.w_hidden: int = w_hidden
        self.activation: type[nn.Module] = activation

        # Model parameters
        self.input: nn.Module | None = None
        self.hidden: nn.Module | None = None
        self.output: nn.Module | None = None
        self.params: nn.Module | None = None
        self._in_features: int | None = None
        self._out_features: int | None = None

    @property
    @override
    def trained(self) -> bool:
        return self.params is not None

    def _build_network(self, in_features: int, out_features: int) -> nn.Module:
        """Build (and store) the network for the given input/output widths.

        Shared by training and loading so the architecture is defined once.
        """
        self._in_features = in_features
        self._out_features = out_features
        self.input = nn.Sequential(
            nn.Linear(in_features, self.w_hidden), self.activation()
        )
        self.hidden = nn.Sequential(
            *sum(
                [
                    [nn.Linear(self.w_hidden, self.w_hidden), self.activation()]
                    for _ in range(self.n_hidden)
                ],
                [],
            )
        )
        self.output = nn.Linear(self.w_hidden, out_features)
        self.params = nn.Sequential(self.input, self.hidden, self.output).to(
            device=self.device,
            dtype=self.dtype,
        )
        return self.params

    @override
    def _optimize_parameters_and_set(self, X: torch.Tensor, y: torch.Tensor):
        self._build_network(X.shape[1], y.shape[1])
        assert self.params is not None

        optimizer = torch.optim.Adam(self.params.parameters(), self.lr)
        self.params.train()
        ds = TensorDataset(X, y)
        dl = DataLoader(ds, self.batch_size)
        for _ in range(self.MAX_EPOCH):
            for X_batch, y_batch in dl:
                optimizer.zero_grad()
                preds = self.params.forward(X_batch)
                loss = mean_squared_error(preds, y_batch)

                # backprop
                loss.backward()
                optimizer.step()
        optimizer.zero_grad()

        self.params.eval()
        self.params.requires_grad_(False)
        return (self.params,)

    def _predict(self, X_: torch.Tensor):
        assert self.params is not None, (
            "The model doesn't see to be fitted, try running .fit() method first"
        )
        self.params.eval()
        self.logger.debug(f"X:{X_.shape}")
        y_pred = self.params.forward(X_)
        self.logger.debug("y':")
        self.logger.debug(y_pred)
        return y_pred

    @staticmethod
    def _resolve_path(filepath: str) -> str:
        return filepath if filepath.endswith(".pt") else f"{filepath}.pt"

    def dump(self, filepath: str = "model", only_hyperparams: bool = False) -> None:
        """Save the model with ``torch.save`` (weights are not JSON-friendly).
        - filepath: string, default = 'model'
            File path to save the model's ``.pt`` file.
        - only_hyperparams: boolean, default = False
            To either save only the model's hyperparameters or not, it
            only affects trained/fitted models.
        """
        payload: dict[str, object] = {
            "type": "FNN",
            "hyperparameters": {
                "MAX_EPOCH": self.MAX_EPOCH,
                "lr": self.lr,
                "batch_size": self.batch_size,
                "activation": self.activation,
                "n_hidden": self.n_hidden,
                "w_hidden": self.w_hidden,
                "dtype": self.dtype,
            },
        }
        if (self.params is not None) and (not only_hyperparams):
            payload["parameters"] = {
                "in_features": self._in_features,
                "out_features": self._out_features,
                "state_dict": self.params.state_dict(),
            }
        torch.save(payload, self._resolve_path(filepath))

    @classmethod
    def load(cls, filepath: str, only_hyperparams: bool = False) -> Self:
        """Load a model saved by :meth:`dump`.
        - filepath: string
            The model's ``.pt`` file path.
        - only_hyperparams: boolean, default = False
            To either load only the model's hyperparameters or not, it
            only has effects when the dump of the model was done with the
            model's parameters.
        """
        payload = torch.load(cls._resolve_path(filepath), weights_only=False)
        if payload["type"] != "FNN":
            raise Exception(f"Model type '{payload['type']}' doesn't match 'FNN'")

        model = cls(**payload["hyperparameters"])

        params = payload.get("parameters")
        if (params is not None) and (not only_hyperparams):
            net = model._build_network(params["in_features"], params["out_features"])
            net.load_state_dict(params["state_dict"])
            net.eval()
            net.requires_grad_(False)
        return model


NumpyArrayorTensor = np.ndarray | torch.Tensor
