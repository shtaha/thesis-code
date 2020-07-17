import numpy as np
import pyomo.environ as pyo


class PyomoMixin:
    @staticmethod
    def _round_solution(x):
        x = np.round(x)
        x = x.astype(np.bool)
        return x

    @staticmethod
    def _dataframe_to_list_of_tuples(df):
        return [tuple(row) for row in df.to_numpy()]

    @staticmethod
    def _create_map_ids_to_values_sum(ids, sum_ids, values):
        return {idx: values[list(sum_ids[idx])].sum() for idx in ids}

    @staticmethod
    def _create_map_ids_to_values(ids, values):
        return {idx: value for idx, value in zip(ids, values)}

    @staticmethod
    def _access_pyomo_variable(var):
        return np.array([pyo.value(var[idx]) for idx in var])
