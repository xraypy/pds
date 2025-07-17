from typing import Any, Callable

import numpy as np
from scipy.optimize import leastsq


def ave(x: np.ndarray) -> np.floating[Any]:
    """Average of an array."""
    return np.mean(x)


def std(x: np.ndarray) -> np.floating[Any]:
    """Standard deviation of an array."""
    return np.std(x)


def line(x: np.ndarray | float, offset: float, slope: float) -> np.ndarray | float:
    """Calculation of a line."""
    return slope * x + offset


def square(a: float) -> float:
    """Square of a number."""
    return a * a


def cosd(x: float) -> float:
    """np.cos(x), x in degrees."""
    return np.cos(np.radians(x))


def sind(x: float) -> float:
    """np.sin(x), x in degrees."""
    return np.sin(np.radians(x))


def tand(x: float) -> float:
    """np.tan(x), x in degrees."""
    return np.tan(np.radians(x))


def arccosd(x: float) -> float:
    """np.arccos(x), result returned in degrees."""
    return np.degrees(np.arccos(x))


def arcsind(x: float) -> float:
    """np.arcsin(x), result returned in degrees."""
    return np.degrees(np.arcsin(x))


def arctand(x: float) -> float:
    """np.arctan(x), result returned in degrees."""
    return np.degrees(np.arctan(x))


def cartesian_mag(v: np.ndarray) -> float:
    """Calculate the norm of a vector defined in a cartesian basis."""
    return float(np.sqrt(np.dot(v, v)))


def cartesian_angle(u: np.ndarray, v: np.ndarray) -> float:
    """Calculate angle between two vectors defined in a cartesian basis."""
    uv = np.dot(u, v)
    um = cartesian_mag(u)
    vm = cartesian_mag(v)
    denom = um * vm
    if denom == 0:
        return 0.0
    arg = uv / denom
    if np.fabs(arg) > 1.0:
        arg = arg / np.fabs(arg)
    alpha = arccosd(arg)
    return alpha


def minimize(f: Callable, x: np.ndarray, y: np.ndarray, params: tuple[float, ...], *args: Any, **kws: Any) -> tuple[float, ...]:
    """Simple wrapper around scipy.optimize.leastsq."""
    XX = x
    YY = y
    FUNC = f

    def _residual(parameters: tuple, *arguments: Any) -> np.ndarray:
        """If the last arg is a dictionary assume it's the kw args for the function."""
        kw = {}
        if len(arguments) > 0:
            if type(arguments[-1]) is dict:
                kw = arguments[-1]
                arguments = arguments[0:-1]
        # Now combine all parameters into a single tuple
        parameters = tuple(parameters) + tuple(arguments)
        # Calculate theory
        yc = FUNC(XX, *parameters, **kw)
        # Return residual
        return YY - yc

    # Ensure params is a tuple
    params = tuple(params)
    args = args + (kws,)

    try:
        test = _residual(params, *args)
        if len(test) != len(x):
            print("cannot minimize function")
            # Return original params if function is incompatible
            return params
    except Exception:
        print("cannot minimize function")
        # Return original params on any error
        return params

    result = leastsq(_residual, params, args=args)
    return result[0]


def random_seed(x: int | None = None) -> None:
    """Wrapper for numpy random seed. Seeds the random number generator."""
    if x is None:
        np.random.seed()
    else:
        try:
            np.random.seed([x])
        except Exception:
            np.random.seed()


def random(a: float = 1, b: float = 1, c: float = 1, npts: int = 1, distribution: str = "normal", **kw: Any) -> np.ndarray | float:
    """
    Wrapper for numpy random distributions

    Parameters:
    -----------
    * a,b,c: default arguments for the dist functions
      e.g. NR.normal a = mean, b = stdev of the distribution
    * npts: number of points

    Outputs:
    --------
    Returns npts random numbers. If npts=1, returns a scalar.
    """
    NR = np.random
    if distribution == "binomial":
        result = NR.binomial(a, b, size=npts)
    elif distribution == "geometric":
        result = NR.geometric(a, size=npts)
    elif distribution == "poisson":
        result = NR.poisson(a, size=npts)
    elif distribution == "zipf":
        result = NR.zipf(a, size=npts)
    elif distribution == "beta":
        result = NR.beta(a, b, size=npts)
    elif distribution == "chisquare":
        result = NR.chisquare(a, size=npts)
    elif distribution == "exponential":
        result = NR.exponential(a, size=npts)
    elif distribution == "gamma":
        result = NR.gamma(a, b, size=npts)
    elif distribution == "gumbel":
        result = NR.gumbel(a, b, size=npts)
    elif distribution == "laplace":
        result = NR.laplace(a, b, size=npts)
    elif distribution == "lognormal":
        result = NR.lognormal(a, b, size=npts)
    elif distribution == "logistic":
        result = NR.logistic(a, b, size=npts)
    elif distribution == "multivariate_normal":
        result = NR.multivariate_normal(a, b, size=npts)
    elif distribution == "noncentral_chisquare":
        result = NR.noncentral_chisquare(a, b, size=npts)
    elif distribution == "noncentral_f":
        result = NR.noncentral_f(a, b, c, size=npts)
    elif distribution == "normal":
        result = NR.normal(a, b, size=npts)
    elif distribution == "pareto":
        result = NR.pareto(a, size=npts)
    elif distribution == "power":
        result = NR.power(a, size=npts)
    elif distribution == "randint":
        result = NR.randint(a, b, size=npts)
    elif distribution == "random_integers":
        result = NR.random_integers(a, b, size=npts)
    elif distribution == "rayleigh":
        result = NR.rayleigh(a, size=npts)
    elif distribution == "standard_cauchy":
        result = NR.standard_cauchy(size=npts)
    elif distribution == "standard_exponential":
        result = NR.standard_exponential(size=npts)
    elif distribution == "standard_gamma":
        result = NR.standard_gamma(a, size=npts)
    elif distribution == "standard_normal":
        result = NR.standard_normal(size=npts)
    elif distribution == "standard_t":
        result = NR.standard_t(a, size=npts)
    elif distribution == "uniform":
        result = NR.uniform(a, b, size=npts)
    elif distribution == "wald":
        result = NR.wald(a, b, size=npts)
    elif distribution == "weibull":
        result = NR.weibull(a, b, size=npts)
    else:
        # Default to normal distribution
        result = NR.normal(a, b, size=npts)

    # Return scalar if single point requested
    if npts == 1:
        return float(result.item()) if hasattr(result, "item") else float(result)
    return result
