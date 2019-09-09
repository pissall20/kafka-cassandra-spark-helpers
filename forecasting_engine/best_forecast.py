# coding=utf-8
import pandas as pd
from algorithms.forecasting_engine.best_forecast_abc import BestForecastInterface
from algorithms.forecasting_engine.data_process import DataProcessing
from algorithms.forecasting_engine.independent_models import TimeSeriesIndependent
import time
import operator
import numpy as np
import sys
from algorithms.forecasting_engine.holtwinters_forecast import HoltWinters
from algorithms.forecasting_engine.arima_forecast import Arima
from algorithms.forecasting_engine.auto_arima import Auto_Arima
from algorithms.forecasting_engine.hw_new import HoltWinters_new
from algorithms.forecasting_engine.kalman_filter import LocalLinearTrend
from algorithms.forecasting_engine.lstm_forecast import Lstm
from algorithms.forecasting_engine.weighted_MA import weighted_moving_average


start = time.time()


class BestForecast(BestForecastInterface):
    """
        Calls inherited classes Abstract, Independent and Dependent
        Saves the best results from all the models.
    """

    def __init__(self):

        print("Inside best forecast init")

        self.forecasting_results = dict()

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
        print("forecasting_values_results : ")
        print(self.forecasting_results)

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
        print("best  :")
        print(best_of_all)
        best_model = best_of_all[0]
        best_mape = best_of_all[1].mape
        best_rmse = best_of_all[1].rmse
        best_model_cfg = best_of_all[1].cfg
        print("best model for the time series is  : " + best_model)

        best_model_class_obj = getattr(sys.modules[__name__], best_model)()
        forecasted_values = best_model_class_obj.get_forecast(
            pred_steps, time_series, best_model_cfg
        )

        print("best model :" + best_model)
        return forecasted_values, best_model, best_mape

    def get_best_mape(self):
        """
        :return: Best mape among all the models
        """
        return self.best_mape


if __name__ == "__main__":

    # if you want to pass direct timeseries, uncomment below and comment rest of the things down below.
    """
    with open("../../test_cases.txt", "r") as fh:
        for line in fh:
            ts = line[:-1]
            timeseries = ts.strip("][").split(", ")
            print(type(timeseries))
            print(timeseries)
            best_f = BestForecast()
            forecasting_models = BestForecast.get_models()
            forecasted_values, best_model, best_mape = best_f.get_best_forecast(
                timeseries, 2, forecasting_models, sort_by_mape=False
            )
            print("forecasted_values, best_model, best_mape")
            print(forecasted_values, best_model, best_mape)
            end = time.time()
            print("TOTAL TIME TAKEN : " + str(end - start))
    """

    def main_func(data):
        best_f = BestForecast()
        forecasting_models = BestForecast.get_models()
        forecasted_values, best_model, best_mape = best_f.get_best_forecast(
            data[metric], 2
        )
        print("forecasted_values, best_model, best_mape")
        print(forecasted_values, best_model, best_mape)
        return forecasted_values, best_mape, best_model

    infile = "~/Documents/csv_files/Board_meeting_mkt_pmart.csv"
    metric = "Sales"
    date_col = "Date"
    split_ratio = 0.9
    pred_steps = 5
    hierarchy = [
        "CATEGORY_NAME",
        "MKT",
        "MANUFACTURER",
        "FRANCHISE",
        "BRAND",
        "SUBBRAND",
    ]
    date_format = "%Y-%m-%d"
    dateparse = lambda x: pd.datetime.strptime(x, date_format)
    data = pd.read_csv(infile, parse_dates=[date_col], date_parser=dateparse)

    data = data.sort_values(by=date_col)
    mape_list = []
    count = 1
    for path, j in data.groupby(hierarchy):

        actual = j[metric].iloc[-1]
        forecasted_values, mape, model = main_func(j)
        row = np.append(path, mape)
        row = np.append(row, model)
        row = np.append(row, actual)
        row = np.append(row, str(forecasted_values))
        mape_list.append(row)
        count += 1
        break

    print("mape :", mape_list)
    col_names = hierarchy + [
        "MAPE_value",
        "model_name",
        "Actual_value ",
        "Predicted_value",
    ]
    df = pd.DataFrame(mape_list, columns=col_names)
    print("dataframe :")
    print(df)
    df.to_csv("RB_complete_auto_arima.csv", index=False)

    end = time.time()
    print("TOTAL TIME TAKEN : " + str(end - start))
