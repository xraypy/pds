import time

import numpy as np
from matplotlib import pyplot
from scipy.stats import linregress


def linear_background(data: np.ndarray, nbgr: int = 0) -> np.ndarray:
    """Calculate linear background based on endpoint regression."""
    ndat = len(data)
    if nbgr <= 0 or ndat < 2 * nbgr + 1:
        return np.zeros(ndat)

    # Create coordinate arrays for endpoints
    xlin = np.concatenate([np.arange(nbgr, dtype=float), np.arange(ndat - nbgr, ndat, dtype=float)])
    ylin = np.concatenate([np.array(data[:nbgr], dtype=float), np.array(data[-nbgr:], dtype=float)])

    # Linear regression
    slope, intercept, *_ = linregress(xlin, ylin)
    return slope * np.arange(ndat) + intercept


def background(data: np.ndarray, nbgr: int = 0, width: int = 0, pow: float = 0.5, tangent: bool = False, compress: int = 1) -> np.ndarray:
    """Calculate polynomial background under a curve using Kajfosz-Kwiatek algorithm."""
    # Validate and fix power parameter
    if pow < 0.0:
        print("Warning power is less than 0, changing it to positive")
        pow = abs(pow)

    # Calculate and subtract linear background
    linbgr = linear_background(data, nbgr=nbgr)
    if width <= 0.0 or pow == 0.0:
        return linbgr

    y = data - linbgr

    # Apply compression if requested
    if compress > 1:
        y, rem = compress_array(y, compress)
        width = max(1, int(width / compress))

    # Initialize background array
    ndat = len(y)
    bgr = np.zeros(ndat)

    # Calculate polynomial parameters
    if width <= 1:
        npoly = min(11, 2 * int(ndat / 2) + 1)
    else:
        npoly = min(10 * int(width / 2) + 1, 2 * int(ndat / 2) + 1)

    pdelx = np.arange(npoly, dtype=float) - (npoly - 1.0) / 2.0
    r = 2 * float(width)
    poly = -1.0 * (pdelx / r) ** (2.0 * pow)

    # Normalize polynomial
    pnorm = (data[:3].sum() + data[-3:].sum()) / 6.0
    poly = poly * pnorm

    # Main background fitting loop
    n = (npoly - 1) // 2
    for j in range(ndat):
        # Calculate data and polynomial indices
        dlidx = max(0, j - n)
        dridx = min(ndat, j + n + 1)
        plidx = max(0, n - j)
        pridx = min(npoly, ndat - j + n)

        # Calculate difference between data and polynomial
        delta = y[dlidx:dridx] - (y[j] + poly[plidx:pridx])

        # Apply tangent correction if requested
        if tangent:
            nl = len(y[dlidx:j])
            nr = len(y[j + 1 : dridx])

            lyave = np.sum(y[dlidx:j]) / nl if nl > 0 else 0.0
            lxave = np.sum(np.arange(dlidx, j)) if nl > 0 else 0.0
            ryave = np.sum(y[j + 1 : dridx]) / nr if nr > 0 else 0.0
            rxave = np.sum(np.arange(j + 1, dridx)) if nr > 0 else 0.0

            slope = (ryave - lyave) / np.abs(rxave - lxave) if rxave != lxave else 0.0
            delta = delta - slope * np.arange(-nl, nr + 1)

        bgr[j] = min(0, delta.min())

    # Apply secondary linear background correction
    linbgr2 = linear_background(-bgr, nbgr=nbgr)
    bgr = bgr + y + linbgr2

    # Expand if compressed
    if compress > 1:
        bgr = expand_array(bgr, compress)
        if rem > 0:
            bgr = np.append(bgr, np.full(rem, bgr[-1], dtype=bgr.dtype))

    # Add back original linear background
    bgr = bgr + linbgr

    return bgr


def show_bgr(data: np.ndarray, nbgr: int = 0, width: int = 0, pow: float = 0.5, tangent: bool = False, compress: int = 1) -> None:
    """Display a simple background plot with timing information."""
    t0 = time.time()
    bgr = background(data, nbgr=nbgr, width=width, pow=pow, tangent=tangent, compress=compress)
    print(f"Background calculation time: {time.time() - t0:.5f} seconds")

    pyplot.plot(data, label="Data")
    pyplot.plot(data - bgr, "r", label="Data - Background")
    pyplot.plot(bgr, "k-", label="Background")
    pyplot.legend()
    pyplot.show()


def plot_bgr(data: np.ndarray, nbgr: int = 0, width: int = 0, pow: float = 0.5, tangent: bool = False, compress: int = 1) -> None:
    """Create detailed background plots."""
    bgr = background(data, nbgr=nbgr, width=width, pow=pow, tangent=tangent, compress=compress)

    # Main plot
    pyplot.figure(1)
    pyplot.clf()
    pyplot.subplot(1, 1, 1)
    pyplot.plot(data, "k-o", label="Data")
    pyplot.plot(bgr, "r-*", label="Background")
    pyplot.plot(data - bgr, "g-", label="Data - Background")
    pyplot.axhline(0, color="k", linestyle="-", alpha=0.3)
    pyplot.legend(loc=2)

    pyplot.show()


def compress_array(array: np.ndarray, compress: int) -> tuple[np.ndarray, int]:
    """Compress 1D array by averaging over integer compression factor."""
    compress = int(compress)
    array_len = len(array)
    new_len = array_len // compress
    remainder = array_len % compress

    # Reshape and average
    reshaped = np.resize(array, (new_len, compress))
    compressed = np.mean(reshaped, axis=1)

    return compressed, remainder


def expand_array(array: np.ndarray, expand: int, sample: int = 0, rem: int = 0) -> np.ndarray:
    """Expand 1D array by integer factor using interpolation or sampling."""
    if expand == 1:
        return array

    if sample == 1:
        return np.repeat(array, expand)

    # Interpolation method
    kernel = np.ones(expand) / expand
    expanded = np.convolve(np.repeat(array, expand), kernel, mode="full")

    # Trim to correct size and fix endpoints
    expanded = expanded[expand - 1 : len(array) * expand + expand - 1]
    for i in range(1, expand):
        expanded[-i] = array[-1]

    # Preserve original dtype
    if expanded.dtype != array.dtype:
        expanded = expanded.astype(array.dtype)

    return expanded
