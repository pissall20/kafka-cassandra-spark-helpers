import datetime
import os
import pickle

import settings
from forecasting_engine.best_forecast import BestForecast
from helper.cassandra import CassandraInterface
from logger import Logger


class ModelTrainer(object):
    """
    Trains a time series model on collected data
    """

    def __init__(self, initial_n_train, pred_steps_seconds):
        self.initial_train = initial_n_train
        self.pred_steps_seconds = pred_steps_seconds
        self.trained_model = None

        self.logger = Logger(self.__class__.__name__).get()

        self.cql_connect = CassandraInterface(
            settings.CASSANDRA_IP,
            settings.CASSANDRA_PORT,
            settings.CASSANDRA_KEY_SPACE,
            settings.CASSANDRA_TABLE_NAME,
            settings.TABLE_SCHEMA
        )

        self.last_timestamp = None

    def _load_all_data(self):
        initial_data, max_time_stamp = self.cql_connect.get_initial_data(time_column=settings.TIME_COLUMN)
        self.last_timestamp = max_time_stamp
        return initial_data

    # @TODO First focus on training initial model
    # def _load_current_data(self, last_timestamp=None):
    #     end_time_stamp = self.last_timestamp + datetime.timedelta(
    #         seconds=self.pred_steps_seconds
    #     )
    #     data = self.cql_connect.retrieve_with_timestamps(self.last_timestamp, end_time_stamp)
    #     self.last_timestamp = end_time_stamp
    #     return data

    def get_best_forecast(self, data, kpi_column, split_ratio=0.1):
        available_models = BestForecast.get_models()
        best_forecast_obj = BestForecast()
        predictions, best_model_name, best_mape, best_model_obj = best_forecast_obj.get_best_forecast(
            data[kpi_column], self.pred_steps_seconds, available_models, split_ratio
        )
        self.trained_model = best_model_obj
        return predictions, best_model_name, best_mape

    def initial_training(self, kpi_column):
        df = self._load_all_data()
        predictions, best_model_name, best_mape = self.get_best_forecast(df, kpi_column)
        self.save_model(settings.MODEL_LOCATION, best_model_name)

    def save_model(self, file_path, model_name, id_prefix=None):
        if not self.trained_model:
            self.logger.error('The model has not been trained to be saved.')
            raise ValueError("The model has not been trained to be saved.")
        date_time = datetime.datetime.today().strftime('%Y-%m-%d-%H:%M:%S')
        file_name = f"{(str(id_prefix) + '-') if id_prefix else ''}{model_name}-{date_time}.pkl"
        save_to = os.path.join(file_path, file_name)
        with open(save_to, "wb") as f:
            pickle.dump(self.trained_model, f)
            self.logger.info(f"Model has been saved to {save_to}")

