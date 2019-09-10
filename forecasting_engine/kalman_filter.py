# from __future__ import division, absolute_import, print_function
import numpy as np
from algorithms.forecasting_engine.error_metrics import mape, rmse
from algorithms.forecasting_engine.data_classes import forecasting_result
import statsmodels.api as sm


class LocalLinearTrend(sm.tsa.statespace.MLEModel):
    def __init__(self, best_configuration_found=0, time_series=[], pred_steps=2):

        print("inside kalman filter : ")

        self.best_configuration_found = best_configuration_found
        self.time_series = time_series
        self.pred_steps = pred_steps
        self.train_ts = np.zeros(1)
        self.test_ts = np.zeros(1)

    def apply_model(self, TrainTestData=[]):
        # Model order
        k_states = k_posdef = 2
        if self.best_configuration_found == 0:
            self.train_ts = TrainTestData[0]
            self.test_ts = TrainTestData[1]
            self.time_series = TrainTestData[2]
            time_series = self.train_ts
        else:
            time_series = self.time_series

        # Initialize the statespace
        super(LocalLinearTrend, self).__init__(
            time_series, k_states=k_states, k_posdef=k_posdef
        )  # super required to initialize state space

        # Initialize the matrices
        self["design"] = np.r_[1, 0]
        self["transition"] = np.array([[1, 1], [0, 1]])
        self["selection"] = np.eye(k_states)

        # Initialize the state space model as approximately diffuse
        self.initialize_approximate_diffuse()

        # Because of the diffuse initialization, burn first two
        # loglikelihoods
        self.loglikelihood_burn = 2

        # Cache some indices
        idx = np.diag_indices(k_posdef)
        self._state_cov_idx = ("state_cov", idx[0], idx[1])

        # Setup parameter names
        self._param_names = ["sigma2.measurement", "sigma2.level", "sigma2.trend"]

        if self.best_configuration_found == 0:
            mape_error, rmse_error = self._best_kalman_model()
            best_cfg = []
            # Save the forecasting results to be reflected back to the Manager
            TrainTestData[3]["LocalLinearTrend"] = forecasting_result(
                mape_error, rmse_error, best_cfg, self
            )

    @property
    def start_params(self):
        # Simple start parameters: just set as 0.1
        return np.r_[0.1, 0.1, 0.1]

    def transform_params(self, unconstrained):
        # Parameters must all be positive for likelihood evaluation.
        # This transforms parameters from unconstrained parameters
        # returned by the optimizer to ones that can be used in the model.
        return unconstrained ** 2

    def untransform_params(self, constrained):
        # This transforms parameters from constrained parameters used
        # in the model to those used by the optimizer
        return constrained ** 0.5

    def update(self, params, **kwargs):
        # The base Model class performs some nice things like
        # transforming the params and saving them
        params = super(LocalLinearTrend, self).update(params, **kwargs)

        # Extract the parameters
        measurement_variance = params[0]
        level_variance = params[1]
        trend_variance = params[2]

        # Observation covariance
        self["obs_cov", 0, 0] = measurement_variance

        # State covariance
        self[self._state_cov_idx] = [level_variance, trend_variance]

    def _best_kalman_model(self):
        res = self.fit()
        pred_vals = res.forecast(steps=len(self.test_ts))
        actual = self.test_ts
        mape_error = mape(actual, pred_vals)
        rmse_error = rmse(actual, pred_vals)
        return mape_error, rmse_error

    def get_forecast(self, pred_steps, time_series, best_cfg):

        """
        apply_model called to initialize state space model so as to enable the forecast
        """

        self.best_configuration_found = 1
        self.time_series = time_series
        self.pred_steps = pred_steps
        self.apply_model()

        res = self.fit()
        pred_val = res.forecast(pred_steps)
        return pred_val
