import collections

forecasting_result = collections.namedtuple(
    "forecasting_result", ["mape", "rmse", "cfg", "instance"]
)
