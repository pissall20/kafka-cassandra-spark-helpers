import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from numpy import array
from algorithms.forecasting_engine.error_metrics import mape, rmse
from algorithms.forecasting_engine.data_classes import forecasting_result


class Lstm(object):
    def __init__(
        self,
        best_configuration_found=0,
        n_lag=3,
        n_seq=3,
        n_epochs=1500,
        n_batch=1,
        n_neurons=1,
        time_series=[],
        pred_steps=2,
    ):
        print("inside new lstm class :")
        # super(Lstm, self).__init__()   #super removed to implement mp

        # configure
        self.n_lag = n_lag
        self.n_epochs = n_epochs
        self.n_batch = n_batch
        self.n_neurons = n_neurons
        self.best_configuration_found = best_configuration_found
        self.time_series = time_series

        if self.best_configuration_found == 0:
            self.n_seq = n_seq

        self.train_ts = np.zeros(1)
        self.test_ts = np.zeros(1)
        self.forecasts = []

    def apply_model(self, TrainTestData=[]):
        if self.best_configuration_found == 0:
            self.train_ts = TrainTestData[0]
            self.test_ts = TrainTestData[1]
            self.time_series = TrainTestData[2]
            self.len_test_ts = len(self.test_ts)
            # prepare data
            self.scaler, self.xtrain_ts, self.ytrain_ts, self.xtest_ts, self.ytest_ts = (
                self.prepare_dataset()
            )

        else:
            self.scaler, self.xtrain_ts, self.ytrain_ts = self.prepare_dataset()
            self.xtest_ts = np.array(self.time_series[-self.n_lag :])
            self.xtest_ts = self.xtest_ts.reshape(1, self.xtest_ts.shape[0])

        # fit model
        model = self.fit_lstm()

        # make forecasts
        forecasts = self.make_forecasts(model)

        # inverse transform forecasts and test
        self.forecasts = self.inverse_transform(forecasts, self.len_test_ts + 2)
        # print ("final : ", self.forecasts)

        if self.best_configuration_found == 0:
            actual = [row for row in self.ytest_ts]
            actual = self.inverse_transform(actual, self.len_test_ts + 2)

            # evaluate forecasts
            mean_mape, mean_rmse = self.evaluate_forecasts(actual, self.forecasts)
            best_cfg = []
            # Save the forecasting results to be reflected back to the Manager
            TrainTestData[3]["Lstm"] = forecasting_result(
                mean_mape, mean_rmse, best_cfg
            )

    # create a differenced series
    def difference(self, dataset, interval=1):
        diff = list()
        for i in range(interval, len(dataset)):
            value = dataset[i] - dataset[i - interval]
            diff.append(value)
        return pd.Series(diff)

    # transform series into train and test sets for supervised learning
    def prepare_dataset(self):
        print("inside prepare data :")

        # extract raw values
        raw_values = self.time_series

        # transform data to be stationary
        diff_series = self.difference(raw_values, 1)
        diff_values = diff_series.values
        diff_values = diff_values.reshape(len(diff_values), 1)

        # rescale values to -1, 1
        scaler = MinMaxScaler(feature_range=(-1, 1))
        scaled_values = scaler.fit_transform(diff_values)
        scaled_values = scaled_values.reshape(len(scaled_values), 1)

        X, y = list(), list()
        for i in range(len(scaled_values)):
            # find the end of this pattern
            end_ix = i + self.n_lag
            out_end_ix = end_ix + self.n_seq
            # check if we are beyond the sequence
            if out_end_ix > len(scaled_values):
                break
            # gather input and output parts of the pattern
            seq_x, seq_y = scaled_values[i:end_ix], scaled_values[end_ix:out_end_ix]
            X.append(seq_x)
            y.append(seq_y)

        if self.best_configuration_found == 0:
            xtrain_ts = X[: -self.len_test_ts]
            xtest_ts = X[-self.len_test_ts :]
            ytrain_ts = y[: -self.len_test_ts]
            ytest_ts = y[-self.len_test_ts :]
            return (
                scaler,
                array(xtrain_ts),
                array(ytrain_ts),
                array(xtest_ts),
                array(ytest_ts),
            )
        else:
            xtrain_ts = X
            ytrain_ts = y
            return scaler, array(xtrain_ts), array(ytrain_ts)

    # fit an LSTM network to training data
    def fit_lstm(self):
        n_features = 1
        X = self.xtrain_ts.reshape(
            len(self.xtrain_ts), n_features, len(self.xtrain_ts[0])
        )
        y = self.ytrain_ts.reshape(len(self.ytrain_ts), len(self.ytrain_ts[0]))

        # design network
        model = Sequential()
        model.add(
            LSTM(
                self.n_neurons,
                batch_input_shape=(self.n_batch, X.shape[1], X.shape[2]),
                stateful=True,
            )
        )
        model.add(Dense(y.shape[1]))
        model.compile(loss="mean_squared_error", optimizer="adam")
        # fit network
        for i in range(self.n_epochs):
            model.fit(X, y, epochs=1, batch_size=self.n_batch, verbose=0, shuffle=False)
            model.reset_states()
        return model

    # make one forecast with an LSTM,
    def forecast_lstm(self, model, X):
        # reshape input pattern to [samples, timesteps, features]
        X = X.reshape(1, 1, len(X))
        # make forecast
        forecast = model.predict(X, batch_size=self.n_batch)
        # convert to array
        return_val = [x for x in forecast[0, :]]
        return return_val

    # evaluate the persistence model
    def make_forecasts(self, model):
        forecasts = list()
        for i in range(len(self.xtest_ts)):
            # X, y = test[i, 0:n_lag], test[i, n_lag:]
            # make forecast
            forecast = self.forecast_lstm(model, self.xtest_ts[i])
            # store the forecast
            forecasts.append(forecast)
        return forecasts

    # invert differenced forecast
    def inverse_difference(self, last_ob, forecast):
        # invert first forecast
        inverted = list()
        inverted.append(forecast[0] + last_ob)
        # propagate difference forecast using inverted first value
        for i in range(1, len(forecast)):
            inverted.append(forecast[i] + inverted[i - 1])
        return inverted

    # inverse data transform on forecasts
    def inverse_transform(self, forecasts, len_test_ts):
        inverted = list()
        for i in range(len(forecasts)):
            # create array from forecast
            forecast = array(forecasts[i])
            forecast = forecast.reshape(1, len(forecast))
            # invert scaling
            inv_scale = self.scaler.inverse_transform(forecast)
            inv_scale = inv_scale[0, :]
            # invert differencing
            index = len(self.time_series) - len_test_ts + i - 1
            last_ob = self.time_series[index]
            inv_diff = self.inverse_difference(last_ob, inv_scale)
            # store
            inverted.append(inv_diff)
        return inverted

    # evaluate the RMSE for each forecast time step
    def evaluate_forecasts(self, test, forecasts):
        rmse_vals = []
        mape_vals = []
        for i in range(self.n_seq):
            actual = [row[i] for row in test]
            predicted = [forecast[i] for forecast in forecasts]
            rmse_val = rmse(actual, predicted)
            mape_val = mape(actual, predicted)
            rmse_vals.append(rmse_val)
            mape_vals.append(mape_val)
            # print('t+%d RMSE: %f' % ((i + 1), rmse_val))

        return np.mean(mape_vals), np.mean(rmse_vals)

    def get_forecast(self, pred_steps, time_series, best_cfg):
        print("in get_forecast of lstm_new :")

        self.best_configuration_found = 1
        self.n_seq = pred_steps
        self.len_test_ts = pred_steps
        self.time_series = time_series

        self.apply_model()

        return self.forecasts
