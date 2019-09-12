# coding=utf-8
import billiard as mp  # billiard used for compatibility with Celery
from forecasting_engine.holtwinters_forecast import HoltWinters
from forecasting_engine.arima_forecast import Arima
from forecasting_engine.auto_arima import Auto_Arima
from forecasting_engine.hw_new import HoltWinters_new
from forecasting_engine.kalman_filter import LocalLinearTrend
from forecasting_engine.lstm_forecast import Lstm
from forecasting_engine.weighted_MA import weighted_moving_average

from logger import Logger


class TimeSeriesIndependent:
    """
    This will contain all the models which treats the input time series as an independent variable.
    Inherit forecasting model's classes which do not require independent variables to train.
    Also do not forget to add "super().init()" in every class inherited, except in the last class.
    """

    """
    Update: Super is removed to implement multiple processing for model call. So "super.init()" is not longer needed.
    """
    models = {"Arima": Arima()}
    """
        "HoltWinters": HoltWinters(),
        "weighted_moving_average": weighted_moving_average(),
        "Auto_Arima": Auto_Arima(),
        "HoltWinters_new": HoltWinters_new(),
        "Lstm": Lstm(),
        "LocalLinearTrend": LocalLinearTrend(),
    }
    """

    def __init__(self, time_series, train_ts, test_ts):
        """
        Initialize a forecasting_results dictionary which will save all model's forecast.
        Call inherited classes
        """

        """
        Create a list of all the models.
        Map each model call with an object of Process class.
        """

        self.logger = Logger(self.__class__.__name__).get()

        self.logger.info("Inside TimeSeriesIndependent init")

        self.time_series = time_series
        self.train_ts = train_ts
        self.test_ts = test_ts

    def run_models(self, forecasting_models):

        """
        Create a manager list to pass the train/test data to all the processes.
        This data is stored in a shared memory which the manager class handles.
        """

        objects_list = list()

        if not forecasting_models:
            objects_list = list(TimeSeriesIndependent.models.values())
        else:
            for model in forecasting_models:
                objects_list.append(TimeSeriesIndependent.models[model])

        with mp.Manager() as data_manager:
            TrainTestData = data_manager.list()
            TrainTestData.append(self.train_ts)
            TrainTestData.append(self.test_ts)

            # timeseries data is required for quality checks and training lstm
            TrainTestData.append(self.time_series)

            """
            Create a manager dictonary to be passed as an argument that can
            reflect the changes back. This will store the forecasting_results
            from all the models. It is required to pass a manager dictionary,
            as a normal python dictonary won't be able to reflect any changes.

            For the same, a different manager is required as a dictionary of
            the same manager can't be appended to its own list.
            """
            with mp.Manager() as result_manager:
                result_dict = result_manager.dict()
                TrainTestData.append(result_dict)

                processes = [
                    mp.context.Process(target=x.apply_model, args=(TrainTestData,))
                    for x in objects_list
                ]

                # Run processes
                for p in processes:
                    p.start()

                # Exit the completed processes:
                for p in processes:
                    p.join()

                forecasting_results = TrainTestData[3].copy()

        return forecasting_results
