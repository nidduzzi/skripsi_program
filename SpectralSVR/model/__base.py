import abc
import json
from typing import Callable, overload, Any

import numpy as np
import torch
from typing_extensions import Self

NumpyArrayorTensor = np.ndarray | torch.Tensor


def torch_json_encoder(obj: Any):  # pyright: ignore[reportExplicitAny]
    if type(obj).__module__ == torch.__name__:
        if isinstance(obj, torch.Tensor):
            return obj.tolist()
        else:
            return obj.item()
    raise TypeError(f"""Unable to  "jsonify" object of type :', {type(obj)}""")


def dump_model(
    model_dict: dict[str, Any],  # pyright: ignore[reportExplicitAny]
    file_encoder: Callable[[Any], Any],  # pyright: ignore[reportExplicitAny]
    filepath: str = "model",
):
    with open(f"{filepath.replace('.json', '')}.json", "w") as fp:
        json.dump(model_dict, fp, default=file_encoder)


def load_model(filepath: str = "model") -> dict[str, Any]:
    helper_filepath = filepath if filepath.endswith(".json") else f"{filepath}.json"
    with open(helper_filepath, "r", encoding="utf-8") as fp:
        file_text = fp.read()
    model_json = json.loads(file_text)

    return model_json


class MultiRegression(abc.ABC):
    def __init__(
        self,
        verbose: bool = False,
        dtype: torch.dtype = torch.float32,
        device: torch.device | None = None,
    ):
        if device is None:
            device = torch.device("cpu")
        self._device: torch.device = device
        self.verbose: bool = verbose
        self.dtype: torch.dtype = dtype

    @property
    @abc.abstractmethod
    def trained(self) -> bool:
        pass

    @property
    def device(self):
        return self._device

    @device.setter
    def device(self, device: torch.device):
        self._device = device

    @abc.abstractmethod
    def _optimize_parameters_and_set(
        self, X: torch.Tensor, y: torch.Tensor
    ) -> tuple[torch.Tensor, ...]:
        pass

    def fit(
        self,
        X_arr: NumpyArrayorTensor,
        y_arr: NumpyArrayorTensor,
        update=False,
    ):
        """Fits the model given the set of X attribute vectors and y labels.
        - X: ndarray or tensor of shape (n_samples, n_attributes)
        - y: ndarray or tensor of shape (n_samples,) or (n_samples, n_outputs)
            If the label is represented by an array of n_outputs elements, the y
            parameter must have n_outputs columns.
        """
        # converting to tensors and passing to GPU
        x: torch.Tensor
        if isinstance(X_arr, torch.Tensor):
            x = X_arr.to(self.device, dtype=self.dtype)
        else:
            x = torch.from_numpy(X_arr).to(self.device, dtype=self.dtype)
        x = x.view(-1, 1) if x.ndim == 1 else x

        if isinstance(y_arr, torch.Tensor):
            y = y_arr.to(self.device, dtype=self.dtype)
        else:
            y = torch.from_numpy(y_arr).to(self.device, dtype=self.dtype)
        y = y.view(-1, 1) if y.ndim == 1 else y

        assert x.shape[0] == y.shape[0], (
            f"X_arr and y_arr does not have the same shape along the 0th dim: (X: {x.shape}, y: {y.shape})"
        )

        _ = self._optimize_parameters_and_set(x, y)

        return self

    @abc.abstractmethod
    def _predict(self, X_: torch.Tensor) -> torch.Tensor:
        pass

    @overload
    def predict(self, X: np.ndarray) -> np.ndarray: ...

    @overload
    def predict(self, X: torch.Tensor) -> torch.Tensor: ...

    def predict(
        self,
        X: NumpyArrayorTensor,
    ) -> NumpyArrayorTensor:
        """Predicts the labels of data X given a trained model.
        - X: ndarray of shape (n_samples, n_attributes)
        """
        if isinstance(X, torch.Tensor):
            X_reshaped_torch: torch.Tensor = X.reshape(-1, 1) if X.ndim == 1 else X
            x = X_reshaped_torch.clone().to(self.device, dtype=self.dtype)
        else:
            X_reshaped_np = X.reshape(-1, 1) if X.ndim == 1 else X
            x = torch.from_numpy(X_reshaped_np).to(self.device, dtype=self.dtype)

        y_pred = self._predict(x)
        predictions: np.ndarray | torch.Tensor
        if isinstance(X, torch.Tensor):
            predictions = y_pred.to(X)
        else:
            predictions = y_pred.cpu().numpy()

        return predictions.reshape(-1) if X.ndim == 1 else predictions

    @abc.abstractmethod
    def dump(self, filepath: str = "model", only_hyperparams: bool = False):
        pass

    @classmethod
    @abc.abstractmethod
    def load(cls, filepath: str, only_hyperparams: bool = False) -> Self:
        pass

    def print(
        self,
        *values: object,
    ):
        if self.verbose:
            print(*values)
