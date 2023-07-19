import array
from cython_test.right_constant import RightConstant


def get_demand_with_inflow_horizon(demand: float, inflow_horizon) -> RightConstant:
    if inflow_horizon < float("inf"):
        return RightConstant(array.array("d", [0.0, inflow_horizon]), array.array("d", [demand, 0.0]), (0, float("inf")))
    else:
        return RightConstant([0.0], [demand], (0, float("inf")))
