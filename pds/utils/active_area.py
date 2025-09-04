from typing import Optional

import numpy as np
from matplotlib import pyplot

from pds.utils.mathutil import cartesian_angle, cartesian_mag
from pds.utils.polygon import inner_polygon, plot_circle, plot_points, plot_polygon, poly_area, poly_area_num


def active_area(
    nm: np.ndarray,
    ki: np.ndarray = np.array([0.0, 1.0, 0.0]),
    kr: np.ndarray = np.array([0.0, 1.0, 0.0]),
    beam: Optional[list[list[float]]] = None,
    det: Optional[list[list[float]]] = None,
    sample: float | list[list[float]] | None = 1.0,
    plot: bool = False,
    fig: Optional[int] = None,
) -> tuple[float, float]:
    """
    Calculate geometric overlap area between beam, sample and detector polygons.
    Returns beam area and illuminated sample area within detector projection."""
    # Calculate surface system transformation matrix
    M = calc_surf_transform(nm)

    # Calculate k vectors in surface frame - normalize first for consistency
    ki = ki / cartesian_mag(ki)
    kr = kr / cartesian_mag(kr)
    ki_s = np.dot(M, ki)
    kr_s = np.dot(M, kr)

    # Process beam polygon
    if isinstance(beam, (list, tuple)) and len(beam) >= 3:
        beam_poly = []
        for v in beam:
            vs = np.dot(M, v)
            intercept = surface_intercept(ki_s, vs)
            if intercept is not None:
                beam_poly.append(intercept)
        if len(beam_poly) < 3:
            print("Error: insufficient valid beam intercepts")
            return (0.0, 0.0)
    elif beam is not None and len(beam) > 0:
        print("Error: beam is not list or tuple")
        return (0.0, 0.0)
    else:
        # beam is None or empty list - both treated the same
        beam_poly = None

    # Process detector polygon
    if isinstance(det, (list, tuple)) and len(det) >= 3:
        det_poly = []
        for v in det:
            vs = np.dot(M, v)
            intercept = surface_intercept(kr_s, vs)
            if intercept is not None:
                det_poly.append(intercept)
        if len(det_poly) < 3:
            print("Error: insufficient valid detector intercepts")
            return (0.0, 0.0)
    elif det is not None:
        print("Error: detector is not list or tuple")
        return (0.0, 0.0)
    else:
        det_poly = None

    # Process sample polygon
    if isinstance(sample, (list, tuple)) and len(sample) >= 3:
        sam_poly = []
        for v in sample:
            vs = np.dot(M, v)
            if np.abs(vs[2]) > 0.01:
                print("Warning: sample height problem")
            sam_poly.append(vs[:2])
        sample_shape = True
    elif isinstance(sample, (int, float)) or sample is None:
        # Valid for round sample case - sample should be numeric diameter
        sam_poly = None
        sample_shape = False
    else:
        print("Error: sample is not list or tuple")
        return (0.0, 0.0)

    # Compute areas based on sample geometry type
    # All polygons are lists of 2D surface frame vectors (in-plane vectors)
    if not sample_shape:
        return _area_round(beam_poly, det_poly, diameter=sample, plot=plot, fig=fig)
    else:
        return _area_polygon(beam_poly, det_poly, sam_poly, plot=plot, fig=fig)


def _area_round(
    beam_poly: Optional[list[list[float]]],
    det_poly: Optional[list[list[float]]],
    diameter: Optional[float] = None,
    plot: bool = False,
    fig: Optional[int] = None,
) -> tuple[float, float]:
    """
    Compute beam and intersection areas for circular sample geometry.
    Constrains beam polygon to circular sample boundary if diameter specified.
    """
    if plot and fig is not None:
        pyplot.figure(fig)
    if plot:
        pyplot.clf()

    if beam_poly is None:
        return (0.0, 0.0)

    A_beam = poly_area(beam_poly)

    if det_poly is not None:
        inner_poly = inner_polygon(beam_poly, det_poly)
    else:
        inner_poly = beam_poly

    if diameter is not None and diameter <= 0.0:
        diameter = None

    if diameter is None:
        A_int = poly_area(inner_poly)
    else:
        A_int = poly_area_num(beam_poly, diameter=diameter, plot=plot)

    # Generate visualization if requested
    if plot:
        plot_polygon(beam_poly, fmt="ro-", label="Beam")
        if det_poly is not None:
            plot_polygon(det_poly, fmt="ko-", label="Detector")
            plot_points(inner_poly, fmt="go-")
            plot_polygon(inner_poly, fmt="g--", linewidth=4, label="Intersection")
        if diameter is not None:
            plot_circle(diameter / 2.0, fmt="b-", label="Sample")
        pyplot.xlabel("Xs")
        pyplot.ylabel("Ys")
        ll = round(np.sqrt(100.0 * A_int))
        pyplot.xlim(-ll, ll)
        pyplot.ylim(-ll, ll)
        pyplot.grid()
        pyplot.legend()

    return (A_beam, A_int)


def _area_polygon(
    beam_poly: Optional[list[list[float]]],
    det_poly: Optional[list[list[float]]],
    sam_poly: Optional[list[list[float]]],
    plot: bool = False,
    fig: Optional[int] = None,
) -> tuple[float, float]:
    """Compute beam and intersection areas for polygonal sample geometry. Returns area of beam-sample-detector polygon intersection."""
    if plot and fig is not None:
        pyplot.figure(fig)
    if plot:
        pyplot.clf()

    if beam_poly is None:
        return (0.0, 0.0)

    A_beam = poly_area(beam_poly)

    # Start with beam polygon and intersect with sample and detector
    if sam_poly is None:
        inner_poly = beam_poly
    else:
        inner_poly = inner_polygon(beam_poly, sam_poly)

    if det_poly is not None:
        inner_poly = inner_polygon(inner_poly, det_poly)

    A_int = poly_area(inner_poly)

    # Generate visualization if requested
    if plot:
        plot_polygon(beam_poly, fmt="ro-", label="Beam")
        if sam_poly is not None:
            plot_polygon(sam_poly, fmt="bo-", label="Sample")
        if det_poly is not None:
            plot_polygon(det_poly, fmt="ko-", label="Detector")
        plot_points(inner_poly, fmt="go-")
        plot_polygon(inner_poly, fmt="g--", linewidth=4, label="Intersection")
        pyplot.xlabel("Xs")
        pyplot.ylabel("Ys")
        ll = round(np.sqrt(100.0 * A_int))
        pyplot.xlim(-ll, ll)
        pyplot.ylim(-ll, ll)
        pyplot.grid()
        pyplot.legend()

    return (A_beam, A_int)


def calc_surf_transform(nm: np.ndarray) -> np.ndarray:
    """Calculate transformation matrix from lab frame to surface frame. Surface frame has z along normal, y projected from lab -y axis."""
    v_z = nm / cartesian_mag(nm)

    v = np.array([0.0, -1.0, 0.0])
    v_y = v - (np.dot(v, v_z) * v_z) / (cartesian_mag(v_z) ** 2.0)
    v_y = v_y / cartesian_mag(v_y)

    v_x = np.cross(v_y, v_z)
    v_x = v_x / cartesian_mag(v_x)

    F = np.array([v_x, v_y, v_z])
    M = np.linalg.inv(F.transpose())

    return M


def surface_intercept(k: np.ndarray, v: np.ndarray) -> Optional[np.ndarray]:
    """Project 3D vector tip onto surface plane along direction k. Returns 2D surface coordinates or None if k parallel to surface."""
    # If k[2] = 0 then k is parallel to xy plane and no intercept possible
    if k[2] == 0:
        return None

    x_int = v[0] - (k[0] / k[2]) * v[2]
    y_int = v[1] - (k[1] / k[2]) * v[2]

    return np.array([x_int, y_int])


def surface_intercept_bounds(k: np.ndarray, v: np.ndarray, diameter: float) -> np.ndarray:
    """Project vector onto surface plane with circular boundary constraint. Uses quadratic scaling to constrain result within specified diameter."""
    vi = surface_intercept(k, v)
    if vi is None:
        return np.array([0.0, 0.0])

    r = cartesian_mag(vi)
    if r <= diameter / 2.0:
        return vi

    # If r > diameter/2, scale back using quadratic solution
    # Solve: |vi - ks*mag/|ks|| = diameter/2
    ks = np.array(k[0:2])
    ks_mag = cartesian_mag(ks)

    if ks_mag == 0:
        # k is vertical, just scale vi directly
        return vi * (diameter / 2.0) / r

    a = 1.0
    b = -2.0 * np.dot(vi, ks) / ks_mag
    c = r**2.0 - (diameter / 2.0) ** 2.0
    discriminant = b**2.0 - 4.0 * a * c

    # Choose solution that results in smaller angle change
    if discriminant > 0.0:
        mag1 = (-b + np.sqrt(discriminant)) / (2.0 * a)
        mag2 = (-b - np.sqrt(discriminant)) / (2.0 * a)
        vi1 = vi - ks * mag1 / ks_mag
        vi2 = vi - ks * mag2 / ks_mag

        angle1 = np.abs(cartesian_angle(vi, vi1))
        angle2 = np.abs(cartesian_angle(vi, vi2))
        return vi1 if angle1 < angle2 else vi2
    else:
        # No solution, just rescale original vector
        return vi * (diameter / 2.0) / r
