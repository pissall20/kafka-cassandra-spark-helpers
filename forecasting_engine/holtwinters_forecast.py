# coding=utf-8
from forecasting_engine.statsmodel_holtwinters import ExponentialSmoothing
from forecasting_engine.error_metrics import rmse, mape
from forecasting_engine.data_classes import forecasting_result
import math
import numpy as np
from logger import Logger


class HoltWinters(object):
    """
    HoltWinters Algorithm:
    Data must be strictly positive. Hence, the quality check is done.
    Only when the quality check passes, holtwinters will run.
    """

    def __init__(self, best_configuration_found=0, time_series=[], pred_steps=2):
        """

        Perform quality check on input data and if passed, save best forecasting results in the forecasting_results array.
        if best_configuration_found = 1,
            just object is created to call the required member functions.
        else,
            _best_holtwinters_model() function is called.

        """

        self.best_configuration_found = best_configuration_found
        self.time_series = time_series
        self.pred_steps = pred_steps
        self.train_ts = np.zeros(1)
        self.test_ts = np.zeros(1)

        self.logger = Logger(self.__class__.__name__).get()

        self.logger.info("Inside HoltWinters init")

    def apply_model(self, TrainTestData):
        if self.best_configuration_found == 0:
            # super(HoltWinters, self).__init__()    #super removed to implement mp
            self.train_ts = TrainTestData[0]
            self.test_ts = TrainTestData[1]
            self.time_series = TrainTestData[2]
            self._trend_component = ["add", "mul"]
            if self._quality_check():
                holtwinters_mape, holtwinters_rmse, best_trend = (
                    self._best_holtwinters_model()
                )
                # Save the forecasting results to be reflected back to the Manager
                TrainTestData[3]["HoltWinters"] = forecasting_result(
                    holtwinters_mape, holtwinters_rmse, best_trend, self
                )
            else:
                self.logger.info("QC failed for HoltWinters")
                pass

    def _quality_check(self):
        """
        Quality Check: Data must be strictly positive
        :return: True/False
        """
        # QC for Holtwinters
        qc = sum([1 if value <= 0 else 0 for value in self.time_series])
        return qc == 0

    def _apply_holtwinters(self, trend):
        """
        Apply holtwinters on training set, predict for test set one by one and get mape.
        :param trend: additive/multiplicative
        :return: A tuple of mape,rmse and trend component
        """

        train, test = self.train_ts, self.test_ts
        if len(train) < 2:
            return 969696.96, 969696.96, "add"

        model = ExponentialSmoothing(train, trend=trend)
        model_fit = model.fit()
        yhat = model_fit.forecast(len(test))
        mape_error = mape(self.test_ts, yhat)
        rmse_error = rmse(self.test_ts, yhat)
        return mape_error, rmse_error, trend

    def _best_holtwinters_model(self):
        """
        Get best holtwinters model
        :return: Tuple of best_mape, best_rmse and best_trend
        """
        # best_mape, best_rmse, best_trend = float("inf"), float("inf"), None
        errors_config = [
            self._apply_holtwinters(trend) for trend in self._trend_component
        ]
        errors_config.sort()  # By mape
        self.best_mape, self.best_rmse, self.best_cfg = errors_config[0]
        return self.best_mape, self.best_rmse, self.best_cfg

    def get_forecast(self, pred_steps, time_series, best_trend):
        """
        Get best holtwinters model forecasted values
        :return: Tuple of best_mape, best_rmse, predicted_values
        """

        # create a list to store forecasted values.
        time_series = list(time_series)
        yhat = []
        for i in range(pred_steps):
            model = ExponentialSmoothing(time_series, trend=best_trend)
            model_fit = model.fit()
            val = model_fit.forecast()[0]
            if math.isnan(val):
                val = np.mean(time_series[-5:])
            yhat.append(val)
            time_series.append(val)
        return yhat
