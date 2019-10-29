# coding=utf-8
import itertools
import warnings
from statsmodels.tsa.arima_model import ARIMA
from forecasting_engine.error_metrics import rmse, mape
from forecasting_engine.data_classes import forecasting_result
import numpy as np


class Arima(object):

    """
    ARIMA forecasting:
    ARIMA is trained by hyperoptimizing p,d, and q parameters.
    p = range(0,6,2)
    d = range(0,2)
    q = range(0,2)

    """

    def __init__(self, best_configuration_found=0, time_series=[], pred_steps=3):
        """
        Initialize p,d,q values and look for best configuration for arima model
        Super function should not be commented if we add more algorithms in independent and inherit after arima.
        Get the best arima forecast.
        if best_configuration_found = 1,
            just object is created to call the required member functions.
        else,
            best_arima_model() function is called.
        """
        self.best_configuration_found = best_configuration_found
        self.time_series = time_series
        self.pred_steps = pred_steps
        self.train_ts = np.zeros(1)
        self.test_ts = np.zeros(1)

        print("Inside Arima's init")
        # super(Arima, self).__init__()  #super removed to implement mp

    def apply_model(self, TrainTestData):
        if self.best_configuration_found == 0:
            # super(Arima,self).__init__()
            self.train_ts = TrainTestData[0]
            self.test_ts = TrainTestData[1]
            self.time_series = TrainTestData[2]
            self._p_values = range(0, 6, 2)
            self._d_values = range(0, 2)
            self._q_values = range(0, 2)
            arima_mape, arima_rmse, best_cfg, best_model = self._best_arima_model()
            # Save the forecasting results to be reflected back to the Manager
            TrainTestData[3]["Arima"] = forecasting_result(
                arima_mape, arima_rmse, best_cfg, best_model
            )
        else:
            print("in final forecasting init:obj  ")

    def _apply_arima(self, arima_order):
        """
        Apply ARIMA for a particular p,d,q and get the corresponding error metrics

        :param arima_order: Tuple (p,d,q)
        :return: mape_error, rmse_error, arima_order

        """

        train, test = self.train_ts, self.test_ts

        model = ARIMA(train, order=arima_order)

        if len(train) < 7:
            print("ValueError: Insufficient degrees of freedom to estimate")
            return 969696.96, 969696.96, arima_order, model

        try:
            model_fit = model.fit(disp=0)
        except:
            print("SVD did not converge in finding out best mape")
            return 969696.96, 969696.96, arima_order, model

        yhat = model_fit.forecast(len(test))[0]
        mape_error = mape(test, yhat)
        rmse_error = rmse(test, yhat)
        return mape_error, rmse_error, arima_order, model

    def _best_arima_model(self):
        """
        Get best ARIMA model configurations
        :return: best mape, best rmse and best configuration
        """

        s = [self._p_values, self._d_values, self._q_values]
        list_of_orders = list(itertools.product(*s))
        errors_config = [self._apply_arima(order) for order in list_of_orders]
        errors_config.sort()
        best_mape, best_rmse, best_cfg, best_model = errors_config[0]
        return best_mape, best_rmse, best_cfg, best_model

    def get_forecast(self, pred_steps, time_series, best_cfg):
        """
        Get best forecasts from best ARIMA model.
        :return: best mape, best rmse, and best predictions
        """
        warnings.filterwarnings("ignore")

        time_series = time_series.astype("float64")

        yhat = []
        for i in range(pred_steps):
            model = ARIMA(time_series, order=best_cfg)

            try:
                model_fit = model.fit(disp=0)
            except:
                print("SVD did not converge in arima while forecasting future")
                val = np.mean(time_series[-(len(time_series) - 5) :])
                yhat = [val] * pred_steps
                return yhat

            val = model_fit.forecast()[0][0]
            yhat.append(val)
            time_series = np.append(time_series, val)

        return yhat
