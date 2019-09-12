import numpy as np
from scipy.optimize import minimize
from sklearn.model_selection import TimeSeriesSplit

from forecasting_engine.data_classes import forecasting_result
from forecasting_engine.error_metrics import rmse, mape
from logger import Logger


class HoltWinters_new(object):
    """
    Holt-Winters model with the anomalies detection using Brutlag method
    # series - initial time series
    # slen - length of a season
    # alpha, beta, gamma - Holt-Winters model coefficients
    # n_preds - predictions horizon
    # scaling_factor - sets the width of the confidence interval by Brutlag (usually takes values from 2 to 3)
    """

    def __init__(self, best_configuration_found=0, time_series=None, pred_steps=2):

        self.logger = Logger(self.__class__.__name__).get()

        self.logger.info("Inside new hw")

        # super(HoltWinters_new, self).__init__()    #super removed to implement mp

        self.best_configuration_found = best_configuration_found
        self.time_series = [] if not time_series else time_series
        self.pred_steps = pred_steps
        self.train_ts = np.zeros(1)
        self.test_ts = np.zeros(1)

    def apply_model(self, TrainTestData):
        self.train_ts = TrainTestData[0]
        self.test_ts = TrainTestData[1]
        self.time_series = TrainTestData[2]
        if self.best_configuration_found == 1:
            self.train_ts = self.time_series

        elif self.best_configuration_found == 0:
            self.slen = 1
            self.alpha = 0
            self.beta = 0
            self.gamma = 0
            self.scaling_factor = 1.96

            if self._quality_check():
                self.alpha_final, self.beta_final, self.gamma_final = (
                    self.minimize_params()
                )
                best_cfg = [self.alpha_final, self.beta_final, self.gamma_final]
                self.scaling_factor = 3
                holtwinters_mape, holtwinters_rmse = self.triple_exponential_smoothing()
                # Save the forecasting results to be reflected back to the Manager
                TrainTestData[3]["HoltWinters_new"] = forecasting_result(
                    holtwinters_mape, holtwinters_rmse, best_cfg, self
                )

    def _quality_check(self):
        """
        Quality Check: Data must be strictly positive
        :return: True/False
        """
        # QC for Holtwinters
        qc = sum([1 if value <= 0 else 0 for value in self.time_series])
        return qc == 0

    # def __apply_holtwinters(self):

    def initial_trend(self):
        sum = 0.0
        for i in range(self.slen):
            sum += float(self.train[i + self.slen] - self.train[i]) / self.slen

        return sum / self.slen

    def initial_seasonal_components(self):
        seasonals = {}
        season_averages = []
        n_seasons = int(len(self.train) / self.slen)

        # let's calculate season averages
        for j in range(n_seasons):
            season_averages.append(
                sum(self.train[self.slen * j : self.slen * j + self.slen])
                / float(self.slen)
            )

        # let's calculate initial values
        for i in range(self.slen):
            sum_of_vals_over_avg = 0.0

            for j in range(n_seasons):
                sum_of_vals_over_avg += (
                    self.train[self.slen * j + i] - season_averages[j]
                )

            seasonals[i] = sum_of_vals_over_avg / n_seasons

        return seasonals

    def minimize_params(self, best_configuration_found=0):
        x = [0, 0, 0]
        # Minimizing the loss function
        opt = minimize(
            self.timeseriesCVscore,
            x0=x,
            # args=(self.train_ts, mape),
            method="TNC",
            bounds=((0, 1), (0, 1), (0, 1)),
        )

        alpha_final, beta_final, gamma_final = opt.x
        return alpha_final, beta_final, gamma_final

    def triple_exponential_smoothing(
        self,
        best_configuration_found=0,
        slen=1,
        scaling_factor=1.96,
        alpha=0,
        beta=0,
        gamma=0,
        pred_steps=2,
        timeseries=list(),
    ):
        self.result = []
        self.Smooth = []
        self.Season = []
        self.Trend = []
        self.PredictedDeviation = []
        self.UpperBond = []
        self.LowerBond = []

        if best_configuration_found == 1:
            self.alpha = alpha
            self.beta = beta
            self.gamma = gamma
            self.slen = slen
            self.train = timeseries
            self.n_steps = pred_steps
            self.scaling_factor = scaling_factor

        else:
            self.n_steps = len(self.test)

        seasonals = self.initial_seasonal_components()

        # print("I am in triple expo")
        for i in range(len(self.train) + self.n_steps):

            if i == 0:  # components initialization
                smooth = self.train[0]
                trend = self.initial_trend()
                self.result.append(self.train[0])
                self.Smooth.append(smooth)
                self.Trend.append(trend)
                self.Season.append(seasonals[i % self.slen])
                self.PredictedDeviation.append(0)
                self.UpperBond.append(
                    self.result[0] + self.scaling_factor * self.PredictedDeviation[0]
                )
                self.LowerBond.append(
                    self.result[0] - self.scaling_factor * self.PredictedDeviation[0]
                )

                continue

            if i >= len(self.train):  # predicting
                m = i - len(self.train) + 1
                self.result.append(
                    (smooth + (m + 1) * trend) + seasonals[i % self.slen]
                )
                # when predicting we increase uncertainty on each step
                self.PredictedDeviation.append(self.PredictedDeviation[-1] * 1.01)

            else:
                val = self.train[i]
                last_smooth, smooth = (
                    smooth,
                    self.alpha * (val - seasonals[i % self.slen])
                    + (1 - self.alpha) * (smooth + trend),
                )
                trend = self.beta * (smooth - last_smooth) + (1 - self.beta) * trend
                seasonals[i % self.slen] = (
                    self.gamma * (val - smooth)
                    + (1 - self.gamma) * seasonals[i % self.slen]
                )
                self.result.append(smooth + trend + seasonals[i % self.slen])
                # Deviation is calculated according to Brutlag algorithm.
                self.PredictedDeviation.append(
                    self.gamma * np.abs(self.train[i] - self.result[i])
                    + (1 - self.gamma) * self.PredictedDeviation[-1]
                )

            self.UpperBond.append(
                self.result[-1] + self.scaling_factor * self.PredictedDeviation[-1]
            )
            self.LowerBond.append(
                self.result[-1] - self.scaling_factor * self.PredictedDeviation[-1]
            )
            self.Smooth.append(smooth)
            self.Trend.append(trend)
            self.Season.append(seasonals[i % self.slen])
            # return self.series

        if best_configuration_found == 0:
            mape_error = mape(list(self.test), self.result[-len(self.test) :])
            rmse_error = rmse(list(self.test), self.result[-len(self.test) :])
            return mape_error, rmse_error
        else:
            return self.result[:]

    def timeseriesCVscore(self, params):
        """
        Returns error on CV
        params - vector of parameters for optimization
        series - dataset with timeseries
        slen - season length for Holt-Winters model
        """
        errors = []

        # values = series.values
        values = self.train_ts
        self.alpha, self.beta, self.gamma = params

        # set the number of folds for cross-validation
        tscv = TimeSeriesSplit(n_splits=3)

        # iterating over folds, train model on each, forecast and calculate error
        for train, test in tscv.split(values):

            self.train = values[train]
            self.test = values[test]
            self.triple_exponential_smoothing()
            predictions = self.result[-len(self.test) :]
            actual = values[test]
            error = mape(list(actual), predictions)
            errors.append(error)

        # print "error: "
        # print errors
        return np.mean(np.array(errors))

    def get_forecast(self, pred_steps, time_series, best_cfg):
        alpha = best_cfg[0]
        beta = best_cfg[1]
        gamma = best_cfg[2]
        forecasted_values = self.triple_exponential_smoothing(
            best_configuration_found=1,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            pred_steps=pred_steps,
            timeseries=time_series,
        )
        return forecasted_values[-pred_steps:]
