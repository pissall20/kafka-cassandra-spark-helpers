import numpy as np
from pmdarima.arima import auto_arima
from statsmodels.tsa.arima_model import ARIMA
from forecasting_engine.error_metrics import mape, rmse
from forecasting_engine.data_classes import forecasting_result


class Auto_Arima(object):
    def __init__(self, best_configuration_found=0, time_series=[], pred_steps=[]):
        print("Inside auto-arima init ")
        # super(Auto_Arima, self).__init__() #super removed to implement mp

        self.best_configuration_found = best_configuration_found
        self.time_series = time_series
        self.pred_steps = pred_steps
        self.train_ts = np.zeros(1)
        self.test_ts = np.zeros(1)

    def apply_model(self, TrainTestData):
        if self.best_configuration_found == 0:
            self.train_ts = TrainTestData[0]
            self.test_ts = TrainTestData[1]
            self.time_series = TrainTestData[2]
            best_mape, best_rmse, best_cfg = self._apply_auto_arima()
            # Save the forecasting results to be reflected back to the Manager
            TrainTestData[3]["Auto_Arima"] = forecasting_result(
                best_mape, best_rmse, best_cfg, self
            )

    def _apply_auto_arima(self):
        try:
            stepwise_fit = auto_arima(
                self.train_ts,
                start_p=1,
                start_q=1,
                max_p=10,
                max_q=10,
                seasonal=True,
                error_action="ignore",
                suppress_warnings=True,
                stepwise=True,
            )
        except:
            print(
                " raise ValueError('Could not successfully fit ARIMA to input data') : "
            )
            return 696969.69, 696969.69, [1, 1, 1]

        len_of_test_ts = len(self.test_ts)
        auto_arima_pred = stepwise_fit.predict(len_of_test_ts)
        best_mape = mape(self.test_ts, auto_arima_pred)
        best_rmse = rmse(self.test_ts, auto_arima_pred)
        best_cfg = stepwise_fit.order

        return best_mape, best_rmse, best_cfg

    def get_forecast(self, pred_steps, time_series, best_cfg):

        yhat = []
        for i in range(pred_steps):
            if len(time_series) < 7:
                print("ValueError: Insufficient degrees of freedom to estimate")
                return [np.mean(time_series[-5:]) * 5]

            # using ARIMA model(not auto_arima) because of specific (p,d,q) parameter.
            model = ARIMA(time_series, order=best_cfg)

            try:
                model_fit = model.fit(disp=0)
            except:
                print("SVD did not converge in auto_Arima while forecasting future")
                # forecast values is equal to mean of last 5 values of timeseries
                val = np.mean(time_series[-5:])
                yhat = [val] * pred_steps
                return yhat

            val = model_fit.forecast()[0][0]
            yhat.append(val)
            time_series = np.append(time_series, val)

        return yhat
