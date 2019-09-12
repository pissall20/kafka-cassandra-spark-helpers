# coding=utf-8
import numpy as np
import pandas as pd

from logger import Logger


class DataProcessing(object):
    """
    Initialize object, split into train test
    """

    def __init__(self, time_series):
        """
        Checks whether input time series is of list,pd.Series or np.array type
        Splits the time series into train test
        :param timeseries: Input time series
        :param split_ratio: Train and test split ratio
        """

        self.logger = Logger(self.__class__.__name__).get()

        self.logger.info("Inside DataProcessing init")

        if isinstance(time_series, (list, pd.Series)):
            # Initiate the time series
            # Mention the dtype so it throws a ValueError if the type is incorrect
            if not len(time_series):
                self.logger.error("TypeError: passed empty list")
                raise TypeError(" passed empty list")
            self.time_series = np.array(time_series, dtype=np.float64)
            # Save the length of the time series
            self.length = len(self.time_series)

        elif isinstance(time_series, np.ndarray):
            # Convert dtype to float so it throws a ValueError if the type is incorrect
            self.time_series = time_series.astype(np.float64)
            self.length = len(self.time_series)

        else:
            self.logger.error(
                "TypeError: time_series object must be a list, pandas.Series or numpy.ndarray \
                        type which consists of numbers"
            )
            raise TypeError(
                "Time series object must be a list, pandas.Series or numpy.ndarray type which \
                        consists of numbers"
            )

    def train_test_split(self, split_ratio=0.1):
        """
        Saves the train and test data in self object

        :param split_ratio: Train and test split ratio
        :return: Nothing
        """

        # if length of time series < 10, then test_ timeseries contain only 1 data point.
        if self.length < 10:
            self.do_split = self.length - 1
            self.train_ts = self.time_series[: self.do_split]
            self.test_ts = self.time_series[self.do_split :]
        else:
            self.do_split = self.length - int(self.length * split_ratio)
            self.train_ts = self.time_series[: self.do_split]
            self.test_ts = self.time_series[self.do_split :]

            self.logger.info(f"do_split : {self.do_split}")
            self.logger.info(f"train_len : {len(self.train_ts)}")
            self.logger.info(f"tesT_len : {len(self.test_ts)}")

        # uncomment if you want specific length of train and test time series.
        """
        split_len = 5
        self.do_split= self.length-split_len
        self.train_ts = self.time_series[:-split_len]
        self.test_ts = self.time_series[-split_len:]
        """

        return self.train_ts, self.test_ts
