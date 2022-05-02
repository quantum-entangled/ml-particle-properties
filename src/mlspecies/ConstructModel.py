from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from scipy.io import loadmat
from sklearn.metrics import mean_squared_error as MSE
from sklearn.model_selection import train_test_split


class ConstructModel:
    """Construct a model to predict particle properties, using ML."""

    def __init__(
        self,
        name: str = "Ar",
        property_name: str = "Z_int",
        temperature_data: npt.NDArray[Any] = np.empty(0),
        estimator: object = None,
        noise: bool = False,
    ) -> None:
        self.name = name
        self.property_name = property_name
        self.temperature_data = temperature_data
        self.estimator = estimator
        self.noise = noise

        self.property_data = self._load_data()

    def _load_data(self) -> npt.NDArray[Any]:
        """Load data from exicting *.mat files."""
        property_data = (
            loadmat("../data/" + self.name + "_" + self.property_name + "_data.mat")
            .get(self.name + "_" + self.property_name)
            .round(decimals=3)
        )

        if self.noise:
            noise = np.random.normal(0, 0.005, property_data.shape)
            property_data += noise

        return property_data

    def split_data(self) -> tuple[Any]:
        """Use train_test_split to split the data."""
        X_train, X_test, y_train, y_test = train_test_split(
            self.temperature_data, self.property_data, test_size=0.3, random_state=3
        )

        return X_train, X_test, y_train, y_test

    def train_model(self, X_train: npt.NDArray[Any], y_train: npt.NDArray[Any]) -> None:
        """Fit model to given data."""
        if y_train.shape[1] == 1:
            self.estimator.fit(X_train, y_train.ravel())
        else:
            self.estimator.fit(X_train, y_train)

    def make_predictions(self, X_test: npt.NDArray[Any]) -> npt.NDArray[Any]:
        """Calculate model predictions on given data."""
        return self.estimator.predict(X_test)

    def _sort_and_reshape(
        self, X_test: npt.NDArray[Any], y_pred: npt.NDArray[Any]
    ) -> tuple[Any]:
        """Sort and reshape given data."""
        sort_index = X_test.argsort(axis=0)
        X_test_sorted = X_test[sort_index].reshape(-1, 1)
        y_pred_sorted = y_pred[sort_index].reshape(-1, self.property_data.shape[1])

        return X_test_sorted, y_pred_sorted

    def plot_results(self, X_test: npt.NDArray[Any], y_pred: npt.NDArray[Any]) -> None:
        """Plot model prediction results."""
        X_test_sorted, y_pred_sorted = self._sort_and_reshape(X_test, y_pred)

        plt.figure(figsize=(8, 8))
        plt.plot(
            self.temperature_data,
            self.property_data,
            linestyle="-",
            label=("True"),
        )
        plt.plot(
            X_test_sorted,
            y_pred_sorted,
            linestyle="--",
            label=("Predicted"),
        )
        plt.xlim(0, self.temperature_data.max())
        plt.grid()
        plt.title("Comparison Between True And Predicted Values")
        plt.legend(loc="best")
        plt.savefig("../res/" + self.name + "_" + self.property_name + ".png", dpi=400)
        plt.show()

    @staticmethod
    def compute_score(y_test: npt.NDArray[Any], y_pred: npt.NDArray[Any]) -> None:
        """Evaluate model performance."""
        mse = MSE(y_test, y_pred)
        rmse = mse**0.5

        print(f"Test set RMSE: {rmse:.3f}")