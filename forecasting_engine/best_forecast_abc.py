from abc import abstractmethod


class BestForecastInterface:
    @abstractmethod
    def data_preprocessing(self, time_series, split_ratio):
        """
        This method splits the given timeseries based on the split_ratio given
        and returns the train and test timeseries
        """

    @staticmethod
    @abstractmethod
    def get_models():
        """
        This static method returns a list of all the currently supported models
        by forecasting_engine
        """

    @abstractmethod
    def get_forecasts_from_all_models(
        self, time_series, train_ts, test_ts, forecasting_models
    ):
        """
        It gets the forecasted values along with the mape from all the models
        supported by the forecasting_engine
        """

    @abstractmethod
    def get_best_forecast(
        self, time_series, pred_steps, forecasting_models, split_ratio
    ):
        """
        Calls data_preprocessing and get_forecasts_from_all_models methods and
        returns the best forecasted values based on the mape values returned
        from all the models
        """

    @abstractmethod
    def best_model(self):
        """
        Returns the best model instance out of all the forecasting models
        """
