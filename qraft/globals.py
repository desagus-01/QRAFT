import operator as _op

from forecast.scenarios.types import ConstraintSigns
from numpy import array, asarray, inf
from pydantic import ConfigDict

sign_operations = {
    (ConstraintSigns.equal): _op.eq,
    "equal": _op.eq,
    (ConstraintSigns.equal_greater): _op.ge,
    "equal_greater": _op.ge,
    (ConstraintSigns.equal_less): _op.le,
    "equal_less": _op.le,
}

DEFAULT_ROUNDING = 4

model_cfg = ConfigDict(arbitrary_types_allowed=True)


# INFO: Below tau related stuff is taken straight from statsmodels
# These are the new estimates from MacKinnon 2010
# the first axis is N -1
# the second axis is 1 %, 5 %, 10 %
# the last axis is the coefficients

MACKIN_TAU_CUTOFFS = {
    "star_nc": [-1.04, -1.53, -2.68, -3.09, -3.07, -3.77],
    "min_nc": [-19.04, -19.62, -21.21, -23.25, -21.63, -25.74],
    "max_nc": [inf, 1.51, 0.86, 0.88, 1.05, 1.24],
    "star_c": [-1.61, -2.62, -3.13, -3.47, -3.78, -3.93],
    "min_c": [-18.83, -18.86, -23.48, -28.07, -25.96, -23.27],
    "max_c": [2.74, 0.92, 0.55, 0.61, 0.79, 1],
    "star_ct": [-2.89, -3.19, -3.50, -3.65, -3.80, -4.36],
    "min_ct": [-16.18, -21.15, -25.37, -26.63, -26.53, -26.18],
    "max_ct": [0.7, 0.63, 0.71, 0.93, 1.19, 1.42],
    "star_ctt": [-3.21, -3.51, -3.81, -3.83, -4.12, -4.63],
    "min_ctt": [-17.17, -21.1, -24.33, -24.03, -24.33, -28.22],
    "max_ctt": [0.54, 0.79, 1.08, 1.43, 3.49, 1.92],
}

small_scaling = array([1, 1, 1e-2])

large_scaling = array([1, 1e-1, 1e-1, 1e-2])

tau_nc_smallp = [
    [0.6344, 1.2378, 3.2496],
    [1.9129, 1.3857, 3.5322],
    [2.7648, 1.4502, 3.4186],
    [3.4336, 1.4835, 3.19],
    [4.0999, 1.5533, 3.59],
    [4.5388, 1.5344, 2.9807],
]


tau_c_smallp = [
    [2.1659, 1.4412, 3.8269],
    [2.92, 1.5012, 3.9796],
    [3.4699, 1.4856, 3.164],
    [3.9673, 1.4777, 2.6315],
    [4.5509, 1.5338, 2.9545],
    [5.1399, 1.6036, 3.4445],
]


tau_ct_smallp = [
    [3.2512, 1.6047, 4.9588],
    [3.6646, 1.5419, 3.6448],
    [4.0983, 1.5173, 2.9898],
    [4.5844, 1.5338, 2.8796],
    [5.0722, 1.5634, 2.9472],
    [5.53, 1.5914, 3.0392],
]


tau_ctt_smallp = [
    [4.0003, 1.658, 4.8288],
    [4.3534, 1.6016, 3.7947],
    [4.7343, 1.5768, 3.2396],
    [5.214, 1.6077, 3.3449],
    [5.6481, 1.6274, 3.3455],
    [5.9296, 1.5929, 2.8223],
]


tau_nc_largep = [
    [0.4797, 9.3557, -0.6999, 3.3066],
    [1.5578, 8.558, -2.083, -3.3549],
    [2.2268, 6.8093, -3.2362, -5.4448],
    [2.7654, 6.4502, -3.0811, -4.4946],
    [3.2684, 6.8051, -2.6778, -3.4972],
    [3.7268, 7.167, -2.3648, -2.8288],
]


tau_c_largep = [
    [1.7339, 9.3202, -1.2745, -1.0368],
    [2.1945, 6.4695, -2.9198, -4.2377],
    [2.5893, 4.5168, -3.6529, -5.0074],
    [3.0387, 4.5452, -3.3666, -4.1921],
    [3.5049, 5.2098, -2.9158, -3.3468],
    [3.9489, 5.8933, -2.5359, -2.721],
]


tau_ct_largep = [
    [2.5261, 6.1654, -3.7956, -6.0285],
    [2.85, 5.272, -3.6622, -5.1695],
    [3.221, 5.255, -3.2685, -4.1501],
    [3.652, 5.9758, -2.7483, -3.2081],
    [4.0712, 6.6428, -2.3464, -2.546],
    [4.4735, 7.1757, -2.0681, -2.1196],
]


tau_ctt_largep = [
    [3.0778, 4.9529, -4.1477, -5.9359],
    [3.4713, 5.967, -3.2507, -4.2286],
    [3.8637, 6.7852, -2.6286, -3.1381],
    [4.2736, 7.6199, -2.1534, -2.4026],
    [4.6679, 8.2618, -1.822, -1.9147],
    [5.0009, 8.3735, -1.6994, -1.6928],
]


MACKIN_TAU_PVALS = {
    "tau_nc_smallp": asarray(tau_nc_smallp) * small_scaling,
    "tau_c_smallp": asarray(tau_c_smallp) * small_scaling,
    "tau_ct_smallp": asarray(tau_ct_smallp) * small_scaling,
    "tau_ctt_smallp": asarray(tau_ctt_smallp) * small_scaling,
    "tau_nc_largep": asarray(tau_nc_largep) * large_scaling,
    "tau_c_largep": asarray(tau_c_largep) * large_scaling,
    "tau_ct_largep": asarray(tau_ct_largep) * large_scaling,
    "tau_ctt_largep": asarray(tau_ctt_largep) * large_scaling,
}


EQ_TYPE_ADDED_DETS: dict[str, int] = {
    "nc": 0,
    "c": 1,
    "ct": 2,
    "ctt": 3,
}

KPSS_CRIT_VALUES = {
    "c": [0.347, 0.463, 0.574, 0.739],
    "ct": [0.119, 0.146, 0.176, 0.216],
}

KPSS_P_VALS = [0.10, 0.05, 0.025, 0.01]
