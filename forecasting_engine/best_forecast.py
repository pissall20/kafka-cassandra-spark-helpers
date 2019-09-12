# coding=utf-8
import sys
import time

from forecasting_engine.best_forecast_abc import BestForecastInterface
from forecasting_engine.data_process import DataProcessing
from forecasting_engine.independent_models import TimeSeriesIndependent
from logger import Logger

start = time.time()


class BestForecast(BestForecastInterface):
    """
        Calls inherited classes Abstract, Independent and Dependent
        Saves the best results from all the models.
    """

    def __init__(self):

        self.logger = Logger(self.__class__.__name__).get()

        self.logger.info("Inside best forecast init")

        self.forecasting_results = dict()
        self.best_model_instance = None

    def data_preprocessing(self, time_series, split_ratio):
        """
        :return: train and test timeseries
        """

        data_proc = DataProcessing(time_series)
        train_ts, test_ts = data_proc.train_test_split(split_ratio)

        return train_ts, test_ts

    def get_models():
        """
        :return:
            list of all the currently supported models in forecasting_engine
        """

        return list(TimeSeriesIndependent.models.keys())

    def get_forecasts_from_all_models(
        self, time_series, train_ts, test_ts, forecasting_models
    ):
        """
        return type: dictionary with model name in camelCase as key
        :return: forecasted values from all models
        """

        ind_models = TimeSeriesIndependent(time_series, train_ts, test_ts)
        forecasting_results = ind_models.run_models(forecasting_models)

        return forecasting_results

    def get_best_forecast(
        self,
        time_series,
        pred_steps,
        forecasting_models=[],
        split_ratio=0.1,
        sort_by_mape=True,
    ):
        """
        :param timeseries: Input time series
        :param pred_steps: For how many steps do we need to forecast?
        :param split_ratio: Train and test split ratio
        :param forecasting_models: List of models on which forecasting_engine
                                   is to be run

        :return:
            list of best forecast values among all the models
            best model name
            best model's mape
        """

        train_ts, test_ts = self.data_preprocessing(time_series, split_ratio)
        self.forecasting_results = self.get_forecasts_from_all_models(
            time_series, train_ts, test_ts, forecasting_models
        )

        # self.forecasting results has algorithm's name as key and [mape,rmse,forecasts] as value
        self.logger.info(f"forecasting_values_results : {self.forecasting_results}")

        # Getting best model from the key(algorithm name) w.r.t. to the lowest mape value
        # best_of_all = sorted(self.forecasting_results.items(), key=operator.itemgetter(1))
        if sort_by_mape:
            best_of_all = sorted(
                self.forecasting_results.items(), key=lambda x: x[1].mape
            )[0]
        else:
            best_of_all = sorted(
                self.forecasting_results.items(), key=lambda x: x[1].rmse
            )[0]
        self.logger.info(f"best : {best_of_all}")
        best_model = best_of_all[0]
        best_mape = best_of_all[1].mape
        best_rmse = best_of_all[1].rmse
        best_model_cfg = best_of_all[1].cfg
        best_model_instance = best_of_all[1].instance

        best_model_class_obj = getattr(sys.modules[__name__], best_model)()
        forecasted_values = best_model_class_obj.get_forecast(
            pred_steps, time_series, best_model_cfg
        )

        self.logger.info(f"best model : {best_model}")
        return forecasted_values, best_model, best_mape

    def best_model(self):
        """
        :return: Best model instance based on rmse
        """
        return self.best_model_instance
