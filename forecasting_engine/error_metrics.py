# coding=utf-8
import numpy as np


def rmse(actual, prediction):
    """
    Calculates the root mean square error of the actual and predicted
    :param actual: np.array of actual values
    :param prediction: np.array of prediction values
    :return: float, root mean square error
    """
    if isinstance(actual, np.ndarray) and isinstance(prediction, np.ndarray):
        return np.sqrt(np.mean((prediction - actual) ** 2))
    elif isinstance(actual, list) and isinstance(prediction, list):
        return np.sqrt(np.mean((np.array(prediction) - np.array(actual)) ** 2))
    else:
        raise TypeError("Types must be same and either list or numpy array.")


def mape(actual, prediction):
    """
    Calculates the mean absolute percentage error of the actual and predicted
    :param actual: np.array of actual values
    :param prediction: np.array of prediction values
    :return: float, mean absolute percentage error
    """
    if isinstance(actual, np.ndarray) and isinstance(prediction, np.ndarray):
        return np.mean(np.abs(((actual - prediction) / actual) * 100))
    elif isinstance(actual, list) and isinstance(prediction, list):
        return np.mean(
            np.abs(
                (((np.array(actual) - np.array(prediction)) / np.array(actual)) * 100)
            )
        )
    else:
        raise TypeError("Types must be same and either list or numpy array.")


def mae(actual, prediction):
    """
    Calculates the mean absolute error of the actual and predicted
    :param actual: np.array of actual values
    :param prediction: np.array of prediction values
    :return: float, mean absolute error
    """
    if isinstance(actual, np.ndarray) and isinstance(prediction, np.ndarray):
        return np.mean(np.abs((actual - prediction) / actual))
    elif isinstance(actual, list) and isinstance(prediction, list):
        return np.mean(np.abs((np.array(actual) - np.array(prediction))))
    else:
        raise TypeError("Types must be same and either list or numpy array.")


def bias(actual, prediction):
    """
    Calculates the bias of the actual and predicted
    :param actual: np.array of actual values
    :param prediction: np.array of prediction values
    :return: float, bias
    """
    if isinstance(actual, np.ndarray) and isinstance(prediction, np.ndarray):
        forecast_errors = actual - prediction
        return np.sum(forecast_errors) / np.float64(len(forecast_errors))
    elif isinstance(actual, list) and isinstance(prediction, list):
        forecast_errors = np.array(actual) - np.array(prediction)
        return np.sum(forecast_errors) / np.float64(len(forecast_errors))
    else:
        raise TypeError("Types must be same and either list or numpy array.")


def residual_by_actual(actual, prediction):
    """
    Calculates residuals by actuals for
    :param actual: np.array of actual values
    :param prediction: np.array of prediction values
    :return: float, residuals by actuals
    """
    if isinstance(actual, np.ndarray) and isinstance(prediction, np.ndarray):
        sum_errors = np.abs(actual - prediction).sum()
        sum_actuals = np.array(actual).sum()
        return sum_errors / sum_actuals
    elif isinstance(actual, list) and isinstance(prediction, list):
        sum_errors = np.abs(np.array(actual) - np.array(prediction)).sum()
        sum_actuals = np.array(actual).sum()
        return sum_errors / sum_actuals
    else:
        raise TypeError("Types must be same and either list or numpy array.")
