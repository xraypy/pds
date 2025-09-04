import copy
import os
from typing import Any

import numpy as np
import numpy.typing as npt
import tables
from matplotlib import pyplot
from scipy import ndimage

from pds.utils.background import background

# Type aliases for better code documentation
ImageArray = npt.NDArray[np.int32]
FloatArray = npt.NDArray[np.float64]
# [x1, y1, x2, y2]
ROI = list[int]
# [[x, y], ...]
PixelList = list[list[int]]

IMG_BGR_PARAMS: dict[str, int | float | bool] = {
    "bgrflag": 1,
    "cnbgr": 5,
    "cwidth": 0,
    "cpow": 2.0,
    "ctan": False,
    "rnbgr": 5,
    "rwidth": 0,
    "rpow": 2.0,
    "rtan": False,
    "nline": 1,
    "filter": False,
    "compress": 1,
}


def read(file: str | list[str], pixel_map: str | None = None) -> ImageArray | list[ImageArray] | None:
    """Read image file(s) with optional bad pixel correction."""
    try:
        from PIL import Image

        imopen = Image.open
    except ImportError as e:
        print(f"Error importing PIL.Image: {e}")
        return None

    # Load pixel map if provided
    bad_pixels, good_pixels = None, None
    if pixel_map is not None:
        try:
            bad_pixels, good_pixels = read_pixel_map(pixel_map)
        except Exception as e:
            print(f"Warning: Could not read pixel map {pixel_map}: {e}")

    def _read_single_file(filepath: str) -> ImageArray | None:
        """Read a single image file."""
        try:
            with imopen(filepath) as im:
                arr = np.frombuffer(im.tobytes(), dtype=np.int32)
                arr = arr.reshape((im.size[1], im.size[0]))

                if bad_pixels is not None:
                    arr = pixel_mask(arr, bad_pixels, good_pixels or [])
                return arr
        except Exception as e:
            print(f"Error reading file {filepath}: {e}")
            return None

    # Handle single file vs list of files
    if isinstance(file, str):
        return _read_single_file(file)
    elif isinstance(file, list):
        images = []
        for filepath in file:
            img = _read_single_file(filepath)
            if img is not None:
                images.append(img)
        return images if images else None
    else:
        raise TypeError(f"Expected str or list[str], got {type(file)}")


def correct_image(image: ImageArray, pixel_map: str | None = None) -> ImageArray:
    """Apply bad pixel correction to an image."""
    if not pixel_map or pixel_map.startswith(("[]", "None")):
        return image

    try:
        pixel_map_data = eval(pixel_map)
        bad_pixels, good_pixels = pixel_map_data
    except Exception:
        bad_pixels, good_pixels = [], []

    return pixel_mask(image, bad_pixels, good_pixels)


def read_files(file_prefix: str, start: int = 0, end: int = 100, nfmt: int = 3, pixel_map: str | None = None) -> list[ImageArray]:
    """Read a sequence of numbered image files."""
    images = []
    format_str = f"%{nfmt}.{nfmt}d"

    for j in range(start, end + 1):
        ext = format_str % j
        filepath = f"{file_prefix}_{ext}.tif"

        arr = read(filepath, pixel_map=pixel_map)
        if arr is not None:
            images.append(arr)

    return images


def pixel_mask(image: ImageArray, bad_pixels: PixelList | None = None, good_pixels: PixelList | None = None) -> ImageArray | None:
    """Correct bad pixels using neighbor averaging or good pixel values."""
    if bad_pixels is None:
        return None

    if good_pixels is None:
        good_pixels = []

    # Create a copy to avoid modifying the original
    corrected_image = image.copy()
    use_averaging = len(good_pixels) == 0

    ny, nx = image.shape

    for i, (x, y) in enumerate(bad_pixels):
        # Bounds checking
        if not (0 <= x < nx and 0 <= y < ny):
            continue

        if use_averaging:
            # Calculate average of neighboring pixels
            neighbors = []
            for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                nx_pos, ny_pos = x + dx, y + dy
                if 0 <= nx_pos < nx and 0 <= ny_pos < ny:
                    neighbors.append(image[ny_pos, nx_pos])

            if neighbors:
                corrected_image[y, x] = int(np.mean(neighbors))
        else:
            # Use corresponding good pixel if available
            if i < len(good_pixels):
                x2, y2 = good_pixels[i]
                if 0 <= x2 < nx and 0 <= y2 < ny:
                    corrected_image[y, x] = image[y2, x2]

    return corrected_image


def read_pixel_map(fname: str) -> tuple[PixelList, PixelList]:
    """Read bad and good pixel coordinates from a text file."""
    bad_pixels: PixelList = []
    good_pixels: PixelList = []

    try:
        with open(fname) as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                # Skip empty lines and comments
                if not line or line.startswith("#"):
                    continue

                parts = line.split()
                if not parts:
                    continue

                # Parse bad pixel coordinates
                try:
                    bad_coords = [int(x) for x in parts[0].split(",")]
                    if len(bad_coords) != 2:
                        raise ValueError("Bad pixel coordinates must have exactly 2 values")
                    bad_pixels.append(bad_coords)
                except ValueError as e:
                    print(f"Warning: Invalid bad pixel format on line {line_num}: {e}")
                    continue

                # Parse good pixel coordinates if present
                if len(parts) > 1:
                    try:
                        good_coords = [int(x) for x in parts[1].split(",")]
                        if len(good_coords) != 2:
                            raise ValueError("Good pixel coordinates must have exactly 2 values")
                        good_pixels.append(good_coords)
                    except ValueError as e:
                        print(f"Warning: Invalid good pixel format on line {line_num}: {e}")

        return bad_pixels, good_pixels

    except FileNotFoundError:
        print(f"Error: Pixel map file not found: {fname}")
        return [], []
    except Exception as e:
        print(f"Error reading pixel map file {fname}: {e}")
        return [], []


def clip_image(image: ImageArray, roi: ROI | None = None, rotangle: float = 0.0, cp: bool = False) -> ImageArray:
    """Clip an image to a specified region of interest (ROI)."""
    if roi is None or len(roi) != 4:
        roi = [0, 0, image.shape[1], image.shape[0]]

    c1, r1, c2, r2 = _sort_roi(roi)

    # Apply rotation if specified
    if rotangle != 0:
        image = ndimage.rotate(image, rotangle)

    clipped = image[r1:r2, c1:c2]
    return copy.copy(clipped) if cp else clipped


def _sort_roi(roi: ROI) -> ROI:
    """Ensure ROI coordinates are in correct order [x1, y1, x2, y2]."""
    c1, r1, c2, r2 = roi

    # Ensure proper ordering
    x1, x2 = (c1, c2) if c1 < c2 else (c2, c1)
    y1, y2 = (r1, r2) if r1 < r2 else (r2, r1)

    return [x1, y1, x2, y2]


def calc_roi(dx: int = 100, dy: int = 100, shape: tuple[int, int] = (195, 487), cen: tuple[int, int] | None = None) -> ROI:
    """Calculate ROI coordinates given width, height, and center position."""
    if cen is None:
        cen = (shape[0] // 2, shape[1] // 2)

    center_row, center_col = cen

    # Calculate ROI bounds
    x1 = center_col - dx // 2
    x2 = center_col + dx // 2
    y1 = center_row - dy // 2
    y2 = center_row + dy // 2

    return _sort_roi([x1, y1, x2, y2])


def image_plot(
    img: ImageArray,
    fig: int | None = None,
    figtitle: str = "",
    cmap: str | None = None,
    verbose: bool = False,
    im_max: float | None = None,
    rotangle: float = 0.0,
    roi: ROI | None = None,
) -> None:
    """Display an image with optional ROI overlay."""
    if verbose:
        print("Image Statistics:")
        print(f"  Total intensity: {img.sum()}")
        print(f"  Max value: {img.max()}")
        print(f"  Min value: {img.min()}")
        print(f"  Shape: {img.shape}")

    if fig is not None:
        pyplot.figure(fig)
        pyplot.clf()

    # Apply rotation if specified
    display_img = ndimage.rotate(img, rotangle) if rotangle != 0 else img.copy()

    # Set up colormap
    if cmap is not None and isinstance(cmap, str):
        try:
            cmap = getattr(pyplot.cm, cmap)
        except AttributeError:
            print(f"Warning: Colormap '{cmap}' not found, using default")
            cmap = None

    # Validate intensity maximum
    if im_max is not None and im_max < 1:
        im_max = None

    # Add ROI overlay if specified
    if roi is not None:
        c1, r1, c2, r2 = _sort_roi(roi)
        overlay = np.zeros_like(display_img)

        # Draw ROI border (2 pixels wide)
        max_val = display_img.max()
        overlay[r1 - 2 : r1, c1:c2] = max_val
        overlay[r2 : r2 + 2, c1:c2] = max_val
        overlay[r1:r2, c1 - 2 : c1] = max_val
        overlay[r1:r2, c2 : c2 + 2] = max_val

        display_img = display_img + overlay

        if im_max is None:
            im_max = np.max(display_img[r1:r2, c1:c2])

    # Display image
    pyplot.imshow(display_img, cmap=cmap, vmax=im_max)
    pyplot.colorbar(orientation="horizontal")

    if figtitle:
        pyplot.title(figtitle, fontsize=12)


def sum_plot(
    image: ImageArray,
    bgrflag: int = 0,
    cnbgr: int = 5,
    cwidth: int = 0,
    cpow: float = 2.0,
    ctan: bool = False,
    rnbgr: int = 5,
    rwidth: int = 0,
    rpow: float = 2.0,
    rtan: bool = False,
    fig: int | None = None,
) -> None:
    """Plot column and row sums with background subtraction."""
    if fig is not None:
        pyplot.figure(fig)
        pyplot.clf()
    else:
        pyplot.figure()

    # Column sum plot
    pyplot.subplot(2, 1, 1)
    pyplot.title("Column Sum")
    data, data_idx, bgr = line_sum(image, sumflag="c", nbgr=cnbgr, width=cwidth, pow=cpow, tangent=ctan)
    pyplot.plot(data_idx, data, "r", label="Data")
    pyplot.plot(data_idx, bgr, "b", label="Background")
    pyplot.legend()

    # Row sum plot
    pyplot.subplot(2, 1, 2)
    pyplot.title("Row Sum")
    data, data_idx, bgr = line_sum(image, sumflag="r", nbgr=rnbgr, width=rwidth, pow=rpow, tangent=rtan)
    pyplot.plot(data_idx, data, "r", label="Data")
    pyplot.plot(data_idx, bgr, "b", label="Background")
    pyplot.legend()

    pyplot.tight_layout()


def line_sum(
    image: ImageArray,
    sumflag: str = "c",
    nbgr: int = 0,
    width: int = 0,
    pow: float = 2.0,
    tangent: bool = False,
    compress: int = 1,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Sum image data along columns ('c') or rows ('r') with background calculation."""
    if sumflag == "c":
        data = image.sum(axis=0).astype(np.float64)
    elif sumflag == "r":
        data = image.sum(axis=1).astype(np.float64)
    else:
        raise ValueError(f"sumflag must be 'c' or 'r', got '{sumflag}'")

    npts = len(data)
    data_idx = np.arange(npts, dtype=np.float64)

    # Compute background
    bgr = background(data, nbgr=nbgr, width=width, pow=pow, tangent=tangent, compress=compress)

    return data, data_idx, bgr


def line_sum_integral(
    image: ImageArray,
    sumflag: str = "c",
    nbgr: int = 0,
    width: int = 0,
    pow: float = 2.0,
    tangent: bool = False,
    compress: int = 1,
) -> tuple[float, float, float]:
    """Calculate integrated intensity after background subtraction."""
    # Get line sum and background
    data, _, bgr = line_sum(image, sumflag=sumflag, nbgr=nbgr, width=width, pow=pow, tangent=tangent, compress=compress)

    # Calculate integrals
    total_intensity = float(data.sum())
    background_intensity = float(bgr.sum())
    net_intensity = total_intensity - background_intensity

    # Calculate error using Poisson statistics
    error_intensity = total_intensity + background_intensity
    if error_intensity > 0.0:
        error = np.sqrt(error_intensity)
    else:
        error = 0.0

    return net_intensity, error, background_intensity


def image_bgr(
    image: ImageArray,
    lineflag: str = "c",
    nbgr: int = 3,
    width: int = 100,
    pow: float = 2.0,
    tangent: bool = False,
    nline: int = 1,
    filter: bool = False,
    compress: int = 1,
    plot: bool = False,
) -> FloatArray:
    """Calculate 2D background array for an image."""
    bgr_arr = np.zeros(image.shape, dtype=np.float64)

    # Apply spline filtering if requested (can remove intensity - use carefully)
    working_image = image.astype(np.float64)
    if filter:
        working_image = ndimage.interpolation.spline_filter(working_image, order=3)

    if lineflag == "r":
        # Fit background to each row
        if nline > 1:
            # Average multiple lines for smoother background
            half_lines = nline // 2
            n_rows = image.shape[0]

            for j in range(n_rows):
                # Get indices of lines to average
                line_indices = [k for k in range(j - half_lines, j + half_lines + 1) if 0 <= k < n_rows]

                # Average the selected lines
                averaged_line = np.mean(working_image[line_indices, :], axis=0)

                # Calculate background for this averaged line
                bgr_arr[j, :] = background(averaged_line, nbgr=nbgr, width=width, pow=pow, tangent=tangent, compress=compress)
        else:
            # Process each row individually
            for j in range(image.shape[0]):
                bgr_arr[j, :] = background(working_image[j, :], nbgr=nbgr, width=width, pow=pow, tangent=tangent, compress=compress)

    elif lineflag == "c":
        # Fit background to each column
        if nline > 1:
            # Average multiple columns for smoother background
            half_lines = nline // 2
            n_cols = image.shape[1]

            for j in range(n_cols):
                # Get indices of columns to average
                line_indices = [k for k in range(j - half_lines, j + half_lines + 1) if 0 <= k < n_cols]

                # Average the selected columns
                averaged_line = np.mean(working_image[:, line_indices], axis=1)

                # Calculate background for this averaged line
                bgr_arr[:, j] = background(averaged_line, nbgr=nbgr, width=width, pow=pow, tangent=tangent, compress=compress)
        else:
            # Process each column individually
            for j in range(image.shape[1]):
                bgr_arr[:, j] = background(working_image[:, j], nbgr=nbgr, width=width, pow=pow, tangent=tangent, compress=compress)
    else:
        raise ValueError(f"lineflag must be 'c' or 'r', got '{lineflag}'")

    # Display diagnostic plots if requested
    if plot:
        pyplot.figure(3)
        pyplot.clf()

        pyplot.subplot(3, 1, 1)
        pyplot.imshow(working_image, aspect="auto")
        pyplot.title("Original Image")
        pyplot.colorbar()

        pyplot.subplot(3, 1, 2)
        pyplot.imshow(bgr_arr, aspect="auto")
        pyplot.title("2D Background")
        pyplot.colorbar()

        pyplot.subplot(3, 1, 3)
        pyplot.imshow(working_image - bgr_arr, aspect="auto")
        pyplot.title("Background Subtracted")
        pyplot.colorbar()

        pyplot.tight_layout()

    return bgr_arr


class ImageAna:
    """Comprehensive image analysis for Pilatus detector data with background subtraction."""

    def __init__(
        self,
        image: ImageArray,
        roi: ROI | None = None,
        rotangle: float = 0.0,
        bgrflag: int = 1,
        cnbgr: int = 5,
        cwidth: int = 0,
        cpow: float = 2.0,
        ctan: bool = False,
        rnbgr: int = 5,
        rwidth: int = 0,
        rpow: float = 2.0,
        rtan: bool = False,
        nline: int = 1,
        filter: bool = False,
        compress: int = 1,
        plot: bool = True,
        fig: int | None = None,
        figtitle: str = "",
        clpimg: ImageArray | None = None,
        bgrimg: FloatArray | None = None,
        integrated: bool = False,
        Iitg: float = 0.0,
        Ibgr: float = 0.0,
        Ierr: float = 0.0,
        I_c: float = 0.0,
        I_r: float = 0.0,
        Ibgr_c: float = 0.0,
        Ibgr_r: float = 0.0,
        Ierr_c: float = 0.0,
        Ierr_r: float = 0.0,
        im_max: float = -1,
    ) -> None:
        """Initialize ImageAna for comprehensive peak analysis."""
        # Initialize ROI with proper bounds checking
        if roi is None or len(roi) < 4:
            roi = [0, 0, image.shape[1], image.shape[0]]

        # Ensure ROI coordinates are properly ordered
        c1, r1, c2, r2 = roi[:4]
        if c1 > c2:
            c1, c2 = c2, c1
        if r1 > r2:
            r1, r2 = r2, r1

        # Validate ROI bounds
        height, width = image.shape
        c1 = max(0, min(c1, width - 1))
        c2 = max(c1 + 1, min(c2, width))
        r1 = max(0, min(r1, height - 1))
        r2 = max(r1 + 1, min(r2, height))

        self.roi = (int(c1), int(r1), int(c2), int(r2))
        self.rotangle = float(rotangle)
        self.image = image
        self.clpimg = clpimg
        self.bgrimg = bgrimg
        self.integrated = integrated

        # Analysis metadata
        self.title = figtitle

        # Intensity results
        self.I = float(Iitg)  # Total integrated intensity
        self.Ibgr = float(Ibgr)  # Total background intensity
        self.Ierr = float(Ierr)  # Total integration error

        # Column integration results
        self.I_c = float(I_c)  # Column integrated intensity
        self.Ibgr_c = float(Ibgr_c)  # Column background intensity
        self.Ierr_c = float(Ierr_c)  # Column integration error

        # Row integration results
        self.I_r = float(I_r)  # Row integrated intensity
        self.Ibgr_r = float(Ibgr_r)  # Row background intensity
        self.Ierr_r = float(Ierr_r)  # Row integration error

        # Background parameters
        self.bgrflag = int(bgrflag)
        self.cbgr = {"nbgr": int(cnbgr), "width": int(cwidth), "pow": float(cpow), "tan": bool(ctan)}
        self.rbgr = {"nbgr": int(rnbgr), "width": int(rwidth), "pow": float(rpow), "tan": bool(rtan)}

        # Processing parameters
        self.nline = int(nline)
        self.filter = bool(filter)
        self.compress = int(compress)
        self.plotflag = bool(plot)
        self.im_max = float(im_max)

        # Perform integration if not already done
        if not self.integrated:
            self.integrate()

        # Generate plots if requested
        if self.plotflag:
            self.plot(fig=fig)

    ############################################################################
    def get_vars(self) -> tuple[ImageArray | None, FloatArray | None, bool, float, float, float, float, float, float, float, float, float]:
        """Return analysis results as a tuple."""
        return (self.clpimg, self.bgrimg, self.integrated, self.I, self.Ibgr, self.Ierr, self.I_c, self.I_r, self.Ibgr_c, self.Ibgr_r, self.Ierr_c, self.Ierr_r)

    def integrate(self) -> None:
        """Perform comprehensive peak integration with background subtraction."""
        try:
            # Clip image to ROI with optional rotation
            self.clpimg = clip_image(self.image, list(self.roi), rotangle=self.rotangle)

            # Calculate total intensity in ROI
            self.I = float(np.sum(self.clpimg))

            # Initialize background
            self.bgrimg = None
            self.Ibgr = 0.0

            # Calculate 2D background based on method
            if self.bgrflag > 0:
                if self.bgrflag == 1:
                    # Column direction background
                    self.bgrimg = image_bgr(
                        self.clpimg,
                        lineflag="c",
                        nbgr=self.cbgr["nbgr"],
                        width=self.cbgr["width"],
                        pow=self.cbgr["pow"],
                        tangent=self.cbgr["tan"],
                        nline=self.nline,
                        filter=self.filter,
                        compress=self.compress,
                        plot=False,
                    )
                elif self.bgrflag == 2:
                    # Row direction background
                    self.bgrimg = image_bgr(
                        self.clpimg,
                        lineflag="r",
                        nbgr=self.rbgr["nbgr"],
                        width=self.rbgr["width"],
                        pow=self.rbgr["pow"],
                        tangent=self.rbgr["tan"],
                        nline=self.nline,
                        filter=self.filter,
                        compress=self.compress,
                        plot=False,
                    )
                else:
                    # Combined row and column background (averaged)
                    bgr_c = image_bgr(
                        self.clpimg,
                        lineflag="c",
                        nbgr=self.cbgr["nbgr"],
                        width=self.cbgr["width"],
                        pow=self.cbgr["pow"],
                        tangent=self.cbgr["tan"],
                        nline=self.nline,
                        filter=self.filter,
                        compress=self.compress,
                        plot=False,
                    )
                    bgr_r = image_bgr(
                        self.clpimg,
                        lineflag="r",
                        nbgr=self.rbgr["nbgr"],
                        width=self.rbgr["width"],
                        pow=self.rbgr["pow"],
                        tangent=self.rbgr["tan"],
                        nline=self.nline,
                        filter=self.filter,
                        compress=self.compress,
                        plot=False,
                    )
                    # Average the two background methods
                    self.bgrimg = (bgr_c + bgr_r) / 2.0

                # Subtract 2D background
                if self.bgrimg is not None:
                    self.Ibgr = float(np.sum(self.bgrimg))
                    self.I = self.I - self.Ibgr

            # Calculate error using Poisson statistics
            error_variance = abs(self.I + self.Ibgr)
            self.Ierr = np.sqrt(error_variance) if error_variance > 0 else 0.0

            # Perform column integration
            if self.bgrimg is not None:
                # Use background-subtracted image
                self.I_c, self.Ierr_c, self.Ibgr_c = line_sum_integral(self.clpimg - self.bgrimg, sumflag="c", nbgr=0)
            else:
                # Calculate background during integration
                self.I_c, self.Ierr_c, self.Ibgr_c = line_sum_integral(
                    self.clpimg,
                    sumflag="c",
                    nbgr=self.rbgr["nbgr"],
                    width=self.rbgr["width"],
                    pow=self.rbgr["pow"],
                    tangent=self.rbgr["tan"],
                    compress=self.compress,
                )

            # Perform row integration
            if self.bgrimg is not None:
                # Use background-subtracted image
                self.I_r, self.Ierr_r, self.Ibgr_r = line_sum_integral(self.clpimg - self.bgrimg, sumflag="r", nbgr=0)
            else:
                # Calculate background during integration
                self.I_r, self.Ierr_r, self.Ibgr_r = line_sum_integral(
                    self.clpimg,
                    sumflag="r",
                    nbgr=self.cbgr["nbgr"],
                    width=self.cbgr["width"],
                    pow=self.cbgr["pow"],
                    tangent=self.cbgr["tan"],
                    compress=self.compress,
                )

            # Mark integration as complete
            self.integrated = True

        except Exception as e:
            print(f"Error during integration: {e}")
            # Set safe default values
            self.I = self.Ierr = self.Ibgr = 0.0
            self.I_c = self.Ierr_c = self.Ibgr_c = 0.0
            self.I_r = self.Ierr_r = self.Ibgr_r = 0.0
            self.integrated = False

    def plot(self, fig: int | None = None) -> None:
        """Generate comprehensive 4-panel analysis plot."""
        if not self.integrated:
            self.integrate()

        # Set up figure
        if fig is not None:
            pyplot.figure(fig)
            pyplot.clf()
            pyplot.figure(fig, figsize=[12, 8])
        else:
            pyplot.figure(figsize=[12, 8])

        # Prepare titles with results
        title_c = f"Col sum\nI_c = {self.I_c:.3g}, Ierr_c = {self.Ierr_c:.3g}, Ibgr_c = {self.Ibgr_c:.3g}"
        title_r = f"Row sum\nI_r = {self.I_r:.3g}, Ierr_r = {self.Ierr_r:.3g}, Ibgr_r = {self.Ibgr_r:.3g}"
        title_roi = f"I = {self.I:.3g}, Ierr = {self.Ierr:.3g}, Ibgr = {self.Ibgr:.3g}"
        if self.bgrimg is not None:
            title_roi += "\n(background subtracted)"

        # Prepare full image with ROI overlay
        c1, r1, c2, r2 = self.roi
        display_image = ndimage.rotate(self.image, self.rotangle) if self.rotangle != 0.0 else copy.copy(self.image)

        # Calculate intensity maximum for display
        im_max = np.max(display_image[r1:r2, c1:c2])

        # Draw ROI boundary
        max_intensity = display_image.max()
        display_image[r1 - 1 : r1, c1:c2] = max_intensity
        display_image[r2 : r2 + 1, c1:c2] = max_intensity
        display_image[r1:r2, c1 - 1 : c1] = max_intensity
        display_image[r1:r2, c2 : c2 + 1] = max_intensity

        # Panel 1: Column sum analysis
        pyplot.subplot(221)
        pyplot.title(title_c, fontsize=12)

        # Plot raw column sum
        data, data_idx, _ = line_sum(self.clpimg, sumflag="c", nbgr=0)
        rawmax = data.max()
        pyplot.plot(data_idx, data, "k", label="raw sum")

        # Plot background and corrected data
        if self.bgrimg is not None:
            # Background-subtracted data
            data_corr, _, _ = line_sum(self.clpimg - self.bgrimg, sumflag="c", nbgr=0)
            bgr = self.bgrimg.sum(axis=0)
        else:
            # Calculate background on-the-fly
            _, _, bgr = line_sum(self.clpimg, sumflag="c", nbgr=self.rbgr["nbgr"], width=self.rbgr["width"], pow=self.rbgr["pow"], tangent=self.rbgr["tan"])
            data_corr = data - bgr

        pyplot.plot(data_idx, bgr, "r", label="bgr")
        pyplot.plot(data_idx, data_corr, "b", label="data-bgr")
        pyplot.axis([0, data_idx.max(), 0, rawmax * 1.25])
        pyplot.legend(loc=0)

        # Panel 2: Full image with ROI
        pyplot.subplot(222)
        pyplot.title(self.title, fontsize=12)
        pyplot.imshow(display_image, cmap=pyplot.cm.hot, vmax=im_max)
        pyplot.colorbar(orientation="horizontal")

        # Panel 3: ROI zoom
        pyplot.subplot(223)
        pyplot.title(title_roi, fontsize=12)
        roi_image = (self.clpimg - self.bgrimg) if self.bgrimg is not None else self.clpimg
        pyplot.imshow(roi_image, cmap=pyplot.cm.hot, aspect="auto")

        # Panel 4: Row sum analysis
        pyplot.subplot(224)
        pyplot.title(title_r, fontsize=12)

        # Plot raw row sum
        data, data_idx, _ = line_sum(self.clpimg, sumflag="r", nbgr=0)
        rawmax = data.max()
        pyplot.plot(data, data_idx, "k", label="raw sum")

        # Plot background and corrected data
        if self.bgrimg is not None:
            # Background-subtracted data
            data_corr, _, _ = line_sum(self.clpimg - self.bgrimg, sumflag="r", nbgr=0)
            bgr = self.bgrimg.sum(axis=1)
        else:
            # Calculate background on-the-fly
            _, _, bgr = line_sum(self.clpimg, sumflag="r", nbgr=self.cbgr["nbgr"], width=self.cbgr["width"], pow=self.cbgr["pow"], tangent=self.cbgr["tan"])
            data_corr = data - bgr

        pyplot.plot(bgr, data_idx, "r", label="bgr")
        pyplot.plot(data_corr, data_idx, "b", label="data-bgr")
        pyplot.axis([0, rawmax * 1.25, data_idx.max(), 0])
        pyplot.xticks(rotation=-45)
        pyplot.legend(loc=0)

    def embed_plot(self, fig) -> None:
        """Create fancy 4-panel plot to embed in wxPython."""
        if not self.integrated:
            self.integrate()

        fig.clear()
        colormap = None
        title_c = "Col sum\nI_c = %g, Ierr_c = %g, Ibgr_c = %g" % (self.I_c, self.Ierr_c, self.Ibgr_c)
        title_r = "Row sum\nI_r = %g, Ierr_r = %g, Ibgr_r = %g" % (self.I_r, self.Ierr_r, self.Ibgr_r)
        title_roi = "I = %g, Ierr = %g, Ibgr = %g" % (self.I, self.Ierr, self.Ibgr)
        if self.bgrimg is not None:
            title_roi = title_roi + "\n(background subtracted)"

        # calc full image with an roi box
        (c1, r1, c2, r2) = self.roi
        if self.rotangle != 0.0:
            bild = ndimage.rotate(self.image, self.rotangle)
        else:
            bild = copy.copy(self.image)
        # whats the max inside the roi
        if self.im_max == -1:
            im_max = np.max(bild[r1:r2, c1:c2])
        else:
            im_max = self.im_max
        #
        bildMax = bild.max()
        bild[r1 - 2 : r1, c1:c2] = bildMax
        bild[r2 : r2 + 2, c1:c2] = bildMax
        bild[r1:r2, c1 - 2 : c1] = bildMax
        bild[r1:r2, c2 : c2 + 2] = bildMax

        # plot column sum
        self.subplot1 = fig.add_subplot(221)
        self.subplot1.set_title(title_c, fontsize=12)
        # plot raw sum
        (data, data_idx, bgr) = line_sum(self.clpimg, sumflag="c", nbgr=0)
        rawmax = data.max()
        self.subplot1.plot(data_idx, data, "k", label="raw sum")
        # get bgr and data-bgr
        if self.bgrimg is not None:
            # here data is automatically bgr subracted
            (data, data_idx, xx) = line_sum(self.clpimg - self.bgrimg, sumflag="c", nbgr=0)
            bgr = self.bgrimg.sum(axis=0)
        else:
            # here data is data and bgr is correct, therefore data = data-bgr
            (data, data_idx, bgr) = line_sum(
                self.clpimg, sumflag="c", nbgr=self.rbgr["nbgr"], width=self.rbgr["width"], pow=self.rbgr["pow"], tangent=self.rbgr["tan"]
            )
            data = data - bgr
        # plot bgr and bgr subtracted data
        self.subplot1.plot(data_idx, bgr, "r", label="bgr")
        self.subplot1.plot(data_idx, data, "b", label="data-bgr")
        self.subplot1.axis([0, data_idx.max(), 0, rawmax * 1.25])
        self.subplot1.legend(loc=0)

        # plot full image with ROI
        self.subplot2 = fig.add_subplot(222, title=self.title)
        self.subplot2.set_title(self.title, fontsize=12)
        forColorbar = self.subplot2.imshow(bild, cmap=colormap, vmax=im_max)
        fig.colorbar(forColorbar, ax=self.subplot2, orientation="horizontal")

        # plot zoom on image
        self.subplot3 = fig.add_subplot(223, title=title_roi)
        self.subplot3.set_title(title_roi, fontsize=12)
        if self.bgrimg is not None:
            self.subplot3.imshow(self.clpimg - self.bgrimg, cmap=colormap, aspect="auto")
        else:
            self.subplot3.imshow(self.clpimg, cmap=colormap, aspect="auto")

        # plot row sum
        self.subplot4 = fig.add_subplot(224, title=title_r)
        self.subplot4.set_title(title_r, fontsize=12)
        # plot raw sum
        (data, data_idx, bgr) = line_sum(self.clpimg, sumflag="r", nbgr=0)
        rawmax = data.max()
        self.subplot4.plot(data, data_idx, "k", label="raw sum")
        # get bgr and data-bgr
        if self.bgrimg is not None:
            # here data is automatically bgr subracted
            (data, data_idx, xx) = line_sum(self.clpimg - self.bgrimg, sumflag="r", nbgr=0)
            bgr = self.bgrimg.sum(axis=1)
        else:
            # here data is data and bgr is correct, therefore data = data-bgr
            (data, data_idx, bgr) = line_sum(
                self.clpimg, sumflag="r", nbgr=self.cbgr["nbgr"], width=self.cbgr["width"], pow=self.cbgr["pow"], tangent=self.cbgr["tan"]
            )
            data = data - bgr
        # plot bgr and bgr subtracted data
        self.subplot4.plot(bgr, data_idx, "r", label="bgr")
        self.subplot4.plot(data, data_idx, "b", label="data-bgr")
        self.subplot4.axis([0, rawmax * 1.25, data_idx.max(), 0])
        # self.subplot4.xticks(rotation=-45)
        self.subplot4.legend(loc=0)

        im_max = np.max(bild[r1:r2, c1:c2])

        fig.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1, wspace=0.4, hspace=0.4)

        return (im_max, colormap, self.subplot2)


class ImageScan:
    """Manage a collection of images associated with a scan."""

    def __init__(
        self,
        image: list[ImageArray] | ImageArray | None = None,
        rois: list[ROI] | ROI | None = None,
        rotangle: list[float] | float | None = None,
        bgr_params: list[dict[str, int | float | bool]] | dict[str, int | float | bool] | None = None,
        archive: dict[str, str] | None = None,
    ) -> None:
        """Initialize ImageScan for multi-image analysis."""
        # Handle single image or empty input
        if image is None:
            image = []
        elif not isinstance(image, list):
            image = [image]

        # Set up image storage (archive vs memory)
        if archive is not None:
            file = archive.get("file", "images.h5")
            path = archive.get("path")
            setname = archive.get("setname", "S1")
            descr = archive.get("descr", "Scan Data Archive")
            self.image = _ImageList(image, file=file, path=path, setname=setname, descr=descr)
        else:
            self.image = image

        # Initialize arrays
        self.rois: list[ROI] | None = None
        self.rotangle: list[float] | None = None
        self.bgrpar: list[dict[str, int | float | bool]] | None = None
        self.im_max: list[float] = []
        self.peaks: dict[str, npt.NDArray[np.float64]] = {}
        self._is_integrated = False

        # Initialize image-dependent arrays
        self._init_image()

        # Update parameters if provided
        npts = len(self.image)
        self._update_rois(rois, npts)
        self._update_rotangles(rotangle, npts)
        self._update_bgr_params(bgr_params, npts)

    def _update_rois(self, rois: list[ROI] | ROI | None, npts: int) -> None:
        """Update ROI parameters for all images."""
        if rois is not None:
            if isinstance(rois, list) and len(rois) == 4 and not isinstance(rois[0], list):
                # Single ROI applied to all images
                for j in range(npts):
                    if self.rois is not None:
                        self.rois[j] = copy.copy(rois)
            elif isinstance(rois, list) and len(rois) == npts:
                # Individual ROI per image
                for j in range(npts):
                    if self.rois is not None:
                        self.rois[j] = rois[j]

    def _update_rotangles(self, rotangle: list[float] | float | None, npts: int) -> None:
        """Update rotation angles for all images."""
        if rotangle is not None:
            if isinstance(rotangle, (int, float)):
                # Single angle for all images
                for j in range(npts):
                    if self.rotangle is not None:
                        self.rotangle[j] = float(rotangle)
            elif isinstance(rotangle, list) and len(rotangle) == npts:
                # Individual angle per image
                for j in range(npts):
                    if self.rotangle is not None:
                        self.rotangle[j] = rotangle[j]

    def _update_bgr_params(self, bgr_params: list[dict[str, int | float | bool]] | dict[str, int | float | bool] | None, npts: int) -> None:
        """Update background parameters for all images."""
        if bgr_params is not None:
            if isinstance(bgr_params, dict):
                # Single parameter set for all images
                for j in range(npts):
                    if self.bgrpar is not None:
                        self.bgrpar[j] = copy.copy(bgr_params)
            elif isinstance(bgr_params, list) and len(bgr_params) == npts:
                # Individual parameters per image
                for j in range(npts):
                    if self.bgrpar is not None:
                        self.bgrpar[j] = bgr_params[j]

    def __repr__(self) -> str:
        """Return string representation of ImageScan."""
        return f"ImageScan(images={len(self.image)}, integrated={self._is_integrated})"

    def _init_image(self) -> None:
        """Initialize arrays based on number of images."""
        npts = len(self.image)

        # Handle empty image list
        if npts == 0:
            self.rois = []
            self.rotangle = []
            self.bgrpar = []
            self.peaks = {}
            return

        # Initialize arrays if needed
        if self.rois is None:
            self.rois = []
        if self.rotangle is None:
            self.rotangle = []
        if self.bgrpar is None:
            self.bgrpar = []

        # Initialize ROIs with full image dimensions
        if len(self.rois) != npts:
            self.rois = []
            for j in range(npts):
                img_shape = self.image[j].shape
                self.rois.append([0, 0, img_shape[1], img_shape[0]])

        # Initialize rotation angles
        if len(self.rotangle) != npts:
            self.rotangle = [0.0] * npts

        # Initialize display maximum values
        if len(self.im_max) != npts:
            self.im_max = [-1.0] * npts

        # Initialize background parameters
        if len(self.bgrpar) != npts:
            self.bgrpar = [copy.copy(IMG_BGR_PARAMS) for _ in range(npts)]

        # Initialize analysis results arrays
        self.peaks = {
            "I": np.zeros(npts, dtype=np.float64),
            "Ierr": np.zeros(npts, dtype=np.float64),
            "Ibgr": np.zeros(npts, dtype=np.float64),
            "I_c": np.zeros(npts, dtype=np.float64),
            "Ierr_c": np.zeros(npts, dtype=np.float64),
            "Ibgr_c": np.zeros(npts, dtype=np.float64),
            "I_r": np.zeros(npts, dtype=np.float64),
            "Ierr_r": np.zeros(npts, dtype=np.float64),
            "Ibgr_r": np.zeros(npts, dtype=np.float64),
        }

        self._is_integrated = False

    def _is_init(self) -> bool:
        """Check if analysis arrays are properly initialized."""
        if not self.peaks:
            return False
        return len(self.peaks["I"]) == len(self.image)

    def plot(self, idx: int = 0, fig: int | None = None, figtitle: str = "", cmap: str | None = None, verbose: bool = False) -> None:
        """Plot a single image from the scan."""
        if not (0 <= idx < len(self.image)):
            raise IndexError(f"Image index {idx} out of range [0, {len(self.image)})")

        image_plot(
            self.image[idx],
            fig=fig,
            figtitle=figtitle,
            cmap=cmap,
            verbose=verbose,
            im_max=self.im_max[idx] if self.im_max[idx] > 0 else None,
            rotangle=self.rotangle[idx] if self.rotangle else 0.0,
            roi=self.rois[idx] if self.rois else None,
        )

    def integrate(
        self,
        idx: list[int] | None = None,
        roi: list[float] | list[list[float]] | None = None,
        rotangle: float | list[float] | None = None,
        bgr_params: dict[str, Any] | None = None,
        bad_points: list[int] | None = None,
        plot: bool = False,
        fig=None,
    ) -> None:
        """Integrate images at specified indices with optional parameters."""
        # make sure arrays exist:
        if not self._is_init():
            self._init_image()
        # idx of images to integrate
        if idx is None:
            idx = []
        if len(idx) == 0:
            idx = np.arange(len(self.image))
        # Bad points handling
        if bad_points is None:
            bad_points = []
        # update roi
        if roi is not None:
            if len(roi) == 4:
                if type(roi[0]) is list:
                    for j in idx:
                        self.rois[j] = roi[j]
                else:
                    for j in idx:
                        self.rois[j] = roi
            elif len(roi) == len(idx):
                for j in idx:
                    self.rois[j] = roi[j]
        # update rot angles
        if rotangle is not None:
            if type(rotangle) is float:
                for j in idx:
                    self.rotangle[j] = rotangle
            elif len(rotangle) == len(idx):
                for j in idx:
                    self.rotangle[j] = rotangle[j]
        # update bgr
        if bgr_params is not None:
            if type(bgr_params) is dict:
                for j in idx:
                    self.bgrpar[j] = copy.copy(bgr_params)
            elif len(bgr_params) == len(idx):
                for j in idx:
                    self.bgrpar[j] = copy.copy(bgr_params[j])
        # do integrations
        for j in idx:
            if j not in bad_points:
                self._integrate(idx=j, plot=plot, fig=fig)
            else:
                self.peaks["I"][j] = 0.0
                self.peaks["Ierr"][j] = 0.0
                self.peaks["Ibgr"][j] = 0.0
                #
                self.peaks["I_c"][j] = 0.0
                self.peaks["Ierr_c"][j] = 0.0
                self.peaks["Ibgr_c"][j] = 0.0
                #
                self.peaks["I_r"][j] = 0.0
                self.peaks["Ierr_r"][j] = 0.0
                self.peaks["Ibgr_r"][j] = 0.0

        self._is_integrated = True

    def _integrate(self, idx: int = 0, plot: bool = True, fig=None) -> None:
        """Integrate an image at specified index."""
        if idx < 0 or idx > len(self.image):
            return None
        #
        figtitle = "Scan Point = %i" % (idx)
        roi = self.rois[idx]
        rotangle = self.rotangle[idx]
        bgr_params = self.bgrpar[idx]

        img_ana = ImageAna(self.image[idx], roi=roi, rotangle=rotangle, plot=plot, fig=fig, figtitle=figtitle, **bgr_params)

        # results into image_peaks dictionary
        self.peaks["I"][idx] = img_ana.I
        self.peaks["Ierr"][idx] = img_ana.Ierr
        self.peaks["Ibgr"][idx] = img_ana.Ibgr
        #
        self.peaks["I_c"][idx] = img_ana.I_c
        self.peaks["Ierr_c"][idx] = img_ana.Ierr_c
        self.peaks["Ibgr_c"][idx] = img_ana.Ibgr_c
        #
        self.peaks["I_r"][idx] = img_ana.I_r
        self.peaks["Ierr_r"][idx] = img_ana.Ierr_r
        self.peaks["Ibgr_r"][idx] = img_ana.Ibgr_r


class _ImageList:
    """Manage images in HDF5 file using PyTables for efficient storage."""

    def __init__(
        self, images: list[ImageArray] | None, file: str = "images.h5", path: str | None = None, setname: str = "S000", descr: str = "Scan data images"
    ) -> None:
        self.path = path
        self.file = file
        self.setname = setname

        if images is not None:
            self.nimages = len(images)
            try:
                self._write_image_tables(images, setname, descr)
            except Exception as e:
                self._cleanup()
                print("Unable to write images to HDF5:")
                print(f"  File: {file}, Setname: {setname}")
                print(f"  Error: {e}")
        else:
            self.nimages = 0

    def _cleanup(self) -> None:
        """Clean up any open HDF5 file handles."""
        try:
            tables.file.close_open_files()
        except Exception:
            pass

    def __len__(self) -> int:
        """Return number of images stored."""
        return self.nimages

    def __getitem__(self, arg: int | slice) -> ImageArray | list[ImageArray]:
        """Get image(s) from HDF5."""
        images = self._read_image_tables()
        if images is None:
            raise RuntimeError("Failed to read images from HDF5 file")
        return images[arg]

    def __setitem__(self, arg: int, value: ImageArray) -> None:
        """Set item. Not implemented for read-only HDF5."""
        raise NotImplementedError("HDF5 image is read-only after creation")

    def _make_fname(self) -> str:
        """Generate absolute path to HDF5 file."""
        if self.path is not None:
            fname = os.path.join(self.path, self.file)
            return os.path.abspath(fname)
        else:
            return os.path.abspath(self.file)

    def _write_image_tables(self, images: list[ImageArray], setname: str, descr: str) -> None:
        """Write images to HDF5 file using PyTables."""

        # Convert to numpy array for storage
        image_array = np.array(images)
        fname = self._make_fname()

        # Open or create HDF5 file
        if os.path.exists(fname):
            h = tables.open_file(fname, mode="a")
            if not hasattr(h.root, "image_data"):
                h.create_group(h.root, "image_data", "Image Data")
        else:
            h = tables.open_file(fname, mode="w", title="Scan Data Archive")
            h.create_group(h.root, "image_data", "Image Data")

        try:
            # Check if dataset already exists
            if hasattr(h.root.image_data, setname):
                print(f"Warning: Dataset '{setname}' already exists in {fname}")
                print("Existing data will not be overwritten")
            else:
                # Create new dataset
                h.create_group("/image_data", setname, "Image Data")
                grp = f"/image_data/{setname}"
                h.create_array(grp, "images", image_array, descr)
                print(f"Successfully wrote {len(images)} images to {fname}:{grp}")
        finally:
            h.close()

    def _read_image_tables(self) -> npt.NDArray[np.int32] | None:
        """Read images from HDF5 file."""

        fname = self._make_fname()
        if not os.path.exists(fname):
            print(f"Archive file not found: {fname}")
            return None

        grp = f"/image_data/{self.setname}"
        try:
            with tables.open_file(fname, mode="r") as h:
                node = h.get_node(grp, "images")
                return node.read()
        except Exception as e:
            self._cleanup()
            print(f"Error reading image tables from {grp}: {e}")
            return None
