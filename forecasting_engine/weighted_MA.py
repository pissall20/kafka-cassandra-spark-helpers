from forecasting_engine.error_metrics import mape, rmse
from forecasting_engine.data_classes import forecasting_result
import time
import numpy as np

start = time.time()


class weighted_moving_average(object):
    def __init__(
        self, best_configuration_found=0, window=5, time_series=[], pred_steps=5
    ):
        print("Inside Weighted_MA init")
        self.best_configuration_found = best_configuration_found
        self.time_series = time_series
        self.pred_steps = pred_steps
        self.window = window
        self.train_ts = np.zeros(1)
        self.test_ts = np.zeros(1)

    def apply_model(self, TrainTestData):
        if self.best_configuration_found == 0:
            # super(weighted_moving_average, self).__init__()    #super removed to implement mp
            self.train_ts = TrainTestData[0]
            self.test_ts = TrainTestData[1]
            self.time_series = TrainTestData[2]
            wma_mape, wma_rmse = self._best_wma_model()
            # Save the forecasting results to be reflected back to the Manager
            TrainTestData[3]["weighted_moving_average"] = forecasting_result(
                wma_mape, wma_rmse, [], self
            )

    def _best_wma_model(self):
        y = list(self.time_series)
        history = list(self.train_ts)
        actual = list(self.test_ts)
        predictions = []

        for i in range(len(actual)):
            len_history = len(history)
            wma_numerator = 0
            for j in range(self.window):
                wma_numerator = wma_numerator + history[len_history - j - 1] * (
                    self.window - j
                )

            wma_denominator = (self.window * (self.window + 1)) / 2
            predicted_value = wma_numerator / float(wma_denominator)
            history.append(predicted_value)
            predictions.append(predicted_value)
            print("wma forecast while training :")
            print(predicted_value)

        mape_error = mape(actual, predictions)
        rmse_error = rmse(actual, predictions)
        return mape_error, rmse_error

    def get_forecast(self, pred_steps, time_series, best_cfg):
        y = list(time_series)
        for i in range(pred_steps):
            len_y = len(y)
            wma_numerator = 0
            for j in range(self.window):
                wma_numerator = wma_numerator + y[len_y - j - 1] * (self.window - j)

            wma_denominator = (self.window * (self.window + 1)) / 2
            predicted_value = wma_numerator / wma_denominator
            y.append(predicted_value)
            print("wma :actual forecast : ")
            print(predicted_value)

        return y[-pred_steps:]


"""
infile = "LHR_Store_Supergroup_data.csv"
# infile = "RB_Data_with_less_than_mean.csv"

# dateparse = lambda x: pd.datetime.strptime(x, '%m/%d/%Y')
dateparse = lambda x: pd.datetime.strptime(x, '%m/%d/%Y')
date_col = "Date"
data = pd.read_csv(infile, parse_dates=[date_col], date_parser=dateparse)
metric = "VolS"
hierarchy = ["ID"]

data = data.sort_values(by=date_col)

result_list = []

count = 1

for path, j in data.groupby(hierarchy):
    print "count = "
    print count
    count = count + 1
    model = "Weighted_moving_average"
    mape_error, forecasted_values = weighted_moving_average(j[metric][:-1], window=5, fc=1)
    row = np.append(path, forecasted_values)
    row = np.append(row, model)
    row = np.append(row, mape_error)
    result_list.append(row)

df = pd.DataFrame(result_list, columns = ['ID', 'forecasted_value_using_WMA', model, "MAPE_ERROR"])
df.to_csv("wma_results_on_LHR_Store_Supergroup.csv", index=False)

end = time.time()
print ("TOTAL TIME TAKEN : " + str(end-start))

"""
