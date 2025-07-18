import numpy as np
from matplotlib import pyplot

from pds.utils.mathutil import cartesian_angle, cosd, sind


def inner_polygon(poly1: list[list[float]], poly2: list[list[float]]) -> list[list[float]] | None:
    """Find the new polygon made up from the inner intersections of two polygons."""

    npts1 = len(poly1)
    npts2 = len(poly2)
    if npts1 < 3 or npts2 < 3:
        return None
    (poly1, angles1) = sort_points(*poly1)
    (poly2, angles2) = sort_points(*poly2)

    # Loop through all possible line combinations looking for valid line intersections
    intercepts = []
    for j in range(npts1):
        p1 = poly1[j]
        if j == npts1 - 1:
            p2 = poly1[0]
        else:
            p2 = poly1[j + 1]
        for k in range(npts2):
            p3 = poly2[k]
            if k == npts2 - 1:
                p4 = poly2[0]
            else:
                p4 = poly2[k + 1]
            (intercept, flag) = line_intercept(p1, p2, p3, p4)
            if flag > 0:
                intercepts.append(intercept)

    # Determine which points we can get to from the origin without crossing any poly lines
    points = []
    for p in poly1:
        points.append(p)
    for p in poly2:
        points.append(p)
    for p in intercepts:
        points.append(p)
    (points, angles) = sort_points(*points)
    inner_points = []
    for p in points:
        # Check against poly1
        inner = is_inner(p, poly1)
        # Check against poly2
        if inner:
            inner = is_inner(p, poly2)
        if inner:
            inner_points.append(p)

    # Sort the inner points
    (inner_points, angles) = sort_points(*inner_points)
    return inner_points


def is_inner(point: list[float], poly: list[list[float]]) -> bool:
    """Check if a point is inside the polygon."""

    npts = len(poly)
    p1 = [0.0, 0.0]
    p2 = point
    inner = True
    k = 0

    while inner and k < npts:
        p3 = poly[k]
        if k == npts - 1:
            p4 = poly[0]
        else:
            p4 = poly[k + 1]
        (intercept, flag) = line_intercept(p1, p2, p3, p4)
        if flag == 1:
            inner = False
        k = k + 1

    return inner


def line_intercept(p1: list[float], p2: list[float], p3: list[float], p4: list[float]) -> tuple[np.ndarray | None, int]:
    """Compute the point of intersection of two lines defined by p1-p2 and p3-p4."""

    # Note if vertical line m = None and b holds x-val
    (m1, b1) = line_param(p1, p2)
    (m2, b2) = line_param(p3, p4)
    if (m1 is not None) and (m2 is not None):
        if (m1 - m2) != 0.0:
            x = (b2 - b1) / (m1 - m2)
            y = m1 * x + b1
        else:
            return (None, 0)
    elif (m1 is None) and (m2 is not None):
        x = b1
        y = m2 * x + b2
    elif (m1 is not None) and (m2 is None):
        x = b2
        y = m1 * x + b1
    else:
        return (None, 0)

    # Min and max of points.
    max_x1 = max(p1[0], p2[0])
    min_x1 = min(p1[0], p2[0])
    max_y1 = max(p1[1], p2[1])
    min_y1 = min(p1[1], p2[1])
    max_x2 = max(p3[0], p4[0])
    min_x2 = min(p3[0], p4[0])
    max_y2 = max(p3[1], p4[1])
    min_y2 = min(p3[1], p4[1])

    # Check if the intersection is in bounds
    flag = 1
    if x > max_x1 or x < min_x1:
        flag = 0
    elif x > max_x2 or x < min_x2:
        flag = 0
    elif y > max_y1 or y < min_y1:
        flag = 0
    elif y > max_y2 or y < min_y2:
        flag = 0

    # Check if the intersection point corresponds to an end point
    intercept = np.array([x, y])

    def _same(p1, p2, prec=0.0001):
        """Check if two points are the same within precision."""

        # Return np.all(np.equal(p1,p2))
        t1 = np.fabs(p1[0] - p2[0]) < prec
        t2 = np.fabs(p1[1] - p2[1]) < prec

        return t1 and t2

    if flag == 1:
        if _same(intercept, p1):
            flag = 2
        elif _same(intercept, p2):
            flag = 2
        elif _same(intercept, p3):
            flag = 2
        elif _same(intercept, p4):
            flag = 2
    return (intercept, flag)


def line_param(v1: list[float], v2: list[float]) -> tuple[float | None, float]:
    """Calculate line parameters for straight line defined from two vectors."""

    if v1[0] - v2[0] != 0.0:
        m = (v1[1] - v2[1]) / (v1[0] - v2[0])
        b = -m * v1[0] + v1[1]
        if np.fabs(m) > 1.0e6:
            m = None
            b = v1[0]
    else:
        m = None
        b = v1[0]
    return (m, b)


def poly_area(polygon: list[list[float]], sort: bool = True) -> float:
    """Compute the area of a polygon."""

    npts = len(polygon)
    if npts < 3:
        return 0.0
    if sort:
        (points, angles) = sort_points(*polygon)
    else:
        points = polygon

    # Loop through points cyclically computing area of each polygon segment
    A = []
    for j in range(npts):
        p1 = points[j]
        if j == npts - 1:
            p2 = points[0]
        else:
            p2 = points[j + 1]
        a = segment_area(p1, p2)
        A.append(a)

    return np.sum(A)


def sort_points(*pts: list[float]) -> tuple[list[np.ndarray], list[float]]:
    """Sort points according to angle (ccw w/r/t x-axis)."""

    npts = len(pts)
    points = []
    angles = []

    def _angle(v):
        """Compute angle of vector w/r/t x-axis in degrees."""

        angle = cartesian_angle(v, [1.0, 0.0])
        if v[1] < 0.0:
            return 360.0 - angle
        else:
            return angle

    for v in pts:
        v = np.array(v[0:2])
        an = _angle(v)
        j = 0
        while j < npts - 1:
            if j > len(points) - 1:
                break
            if an < angles[j]:
                break
            else:
                j = j + 1
        points.insert(j, v)
        angles.insert(j, an)

    return (points, angles)


def segment_area(p1: list[float], p2: list[float]) -> float:
    """Compute the in-plane area of polygon defined by origin and two points."""

    a = (p1[0] * p2[1]) ** 2.0 + (p2[0] * p1[1]) ** 2.0 - (2.0 * p1[0] * p2[0] * p1[1] * p2[1])
    if a < 0:
        a = 0
    else:
        a = 0.5 * np.sqrt(a)
    return a


def poly_area_num(polygon: list[list[float]], diameter: float | None = None, num_int: int = 100, plot: bool = False) -> float:
    """Numerically compute the area of a polygon."""

    npts = len(polygon)
    if npts < 3:
        return 0.0
    polygon = np.array(polygon)
    min_y = min(polygon[:, 1])
    max_y = max(polygon[:, 1])
    min_x = min(polygon[:, 0])
    max_x = max(polygon[:, 0])

    # Compute x-values for integration
    dx = np.fabs((max_x - min_x) / float(num_int + 1))
    x = np.arange(min_x - 0.5 * dx, max_x + 1.5 * dx, dx)
    # Loop through all x-vals and compute segment area
    A = 0.0

    if plot:
        pline = [[], []]

    for xx in x:
        # Find polygon lines that contain x
        lines = []

        for j in range(npts):
            p1 = polygon[j]

            if j == npts - 1:
                p2 = polygon[0]
            else:
                p2 = polygon[j + 1]

            if xx >= min(p1[0], p2[0]) and xx <= max(p1[0], p2[0]):
                lines.append([p1, p2])

        if len(lines) > 0:
            # Get y intercepts with vert line at xx
            p3 = [xx, min_y]
            p4 = [xx, max_y]
            y = []

            for p1, p2 in lines:
                (inter, flag) = line_intercept(p1, p2, p3, p4)

                if flag == 0:
                    print("Error, should always get intercepts here")
                    return 0.0
                else:
                    y.append(inter[1])

            numy = len(y)

            if np.mod(numy, 2.0) != 0 or numy == 1:
                print("Error, wrong number of intercepts!")
                return 0.0

            y = np.array(y)
            y = y[np.argsort(y)]

            # Figure the length of y inside the polynomial.
            j = 0
            while j < numy:
                ytop = y[j + 1]
                ybot = y[j]
                if diameter is not None:
                    cy = (diameter / 2.0) ** 2.0 - xx**2.0
                    if cy > 0.0:
                        cy_max = np.sqrt(cy)
                        cy_min = -1.0 * cy_max
                        if ytop <= cy_min or ybot >= cy_max:
                            ytop = 0.0
                            ybot = 0.0
                        else:
                            if ytop > cy_max:
                                ytop = cy_max
                            if ybot < cy_min:
                                ybot = cy_min
                    else:
                        ytop = 0.0
                        ybot = 0.0
                dy = ytop - ybot
                A = A + dy * dx
                if plot:
                    pline[0].append([xx, xx])
                    pline[1].append([ybot, ytop])
                j = j + 2

    # Plot the integration lines for debugging
    if plot:
        for j in range(len(pline[0])):
            pyplot.plot(pline[0][j], pline[1][j], "k-")
    return A


def poly_y_intercepts(polygon: list[list[float]]) -> list[np.ndarray]:
    """Find all the y-axis intercepts of the polygon."""

    npts = len(polygon)
    polygon = np.array(polygon)
    min_x = min(polygon[:, 0])
    max_x = max(polygon[:, 0])
    p1 = np.array([min_x, 0.0])
    p2 = np.array([max_x, 0.0])
    intercepts = []

    for j in range(npts):
        p3 = polygon[j]
        if j == npts - 1:
            p4 = polygon[0]
        else:
            p4 = polygon[j + 1]
        (intercept, flag) = line_intercept(p1, p2, p3, p4)
        if flag > 0:
            intercepts.append(intercept)

    return intercepts


def trans_point(p: list[float], theta: float = 0.0, scale: float = 1.0) -> np.ndarray:
    """Simple in-plane rotation of points."""

    M = np.array([[cosd(theta), -sind(theta)], [sind(theta), cosd(theta)]])
    pp = scale * np.dot(M, p)

    return pp


def plot_polygon(polygon: list[list[float]], **kw) -> None:
    """Plot the lines around a polygon."""

    try:
        fmt = kw.pop("fmt")
    except Exception:
        fmt = "k"
    try:
        label = kw.pop("label")
    except Exception:
        label = None
    (points, angles) = sort_points(*polygon)
    npts = len(points)

    if npts < 3:
        return
    for j in range(npts):
        p1 = points[j]
        if j == npts - 1:
            p2 = points[0]
        else:
            p2 = points[j + 1]
        if j < npts - 1:
            pyplot.plot([p1[0], p2[0]], [p1[1], p2[1]], fmt, **kw)
        else:
            pyplot.plot([p1[0], p2[0]], [p1[1], p2[1]], fmt, label=label, **kw)


def plot_points(points: list[list[float]], **kw) -> None:
    """Plot a bunch of in-plane ([x,y]) points."""

    try:
        fmt = kw.pop("fmt")
    except Exception:
        fmt = "k"
    try:
        label = kw.pop("label")
    except Exception:
        label = None

    npts = len(points)

    if npts == 0:
        return
    xy = np.zeros((npts, 2))

    for j in range(npts):
        v = points[j]
        xy[j, 0] = v[0]
        xy[j, 1] = v[1]

    idx = np.argsort(xy[:, 0])
    xy = xy[idx]

    for j in range(len(xy)):
        if j < npts - 1:
            pyplot.plot([0.0, xy[j, 0]], [0, xy[j, 1]], fmt, **kw)
        else:
            pyplot.plot([0.0, xy[j, 0]], [0, xy[j, 1]], fmt, label=label, **kw)


def plot_circle(r: float, **kw) -> None:
    """Plot a circle of given radius."""

    try:
        fmt = kw.pop("fmt")
    except Exception:
        fmt = "k"
    try:
        label = kw.pop("label")
    except Exception:
        label = None

    x = np.arange(-r, r + 0.01, 0.01)
    y = np.sqrt(np.fabs(r**2.0 - x**2.0))
    pyplot.plot(x, y, fmt, **kw)
    pyplot.plot(x, -y, fmt, label=label, **kw)
