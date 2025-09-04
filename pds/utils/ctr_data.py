from typing import Any, Optional

import numpy as np

from pds.utils.active_area import active_area
from pds.utils.converters import bytes_to_str, extract_value
from pds.utils.gonio_psic import beam_vectors, det_vectors, psic_from_spec, sample_vectors
from pds.utils.mathutil import cosd, sind


def image_point_F(
    scan: dict[str, Any],
    point: int,
    Iitg: str = "I",
    Inorm: str = "io",
    Ierr: str = "Ierr",
    Ibgr: str = "Ibgr",
    transm: str = "transm",
    corr_params: Optional[dict[str, Any]] = None,
    preparsed: bool = False,
) -> dict[str, float]:
    """Computes structure factor F and error for single scan point with intensity corrections. Returns dictionary with intensity values and calculated F, Ferr."""
    # Default to empty dict if None passed
    if corr_params is None:
        corr_params = {}

    d = {
        "I": extract_value(scan[Iitg], point),
        "Inorm": extract_value(scan[Inorm], point),
        "Ierr": extract_value(scan[Ierr], point),
        "Ibgr": extract_value(scan[Ibgr], point),
        "transm": extract_value(scan[transm], point),
        "F": 0.0,
        "Ferr": 0.0,
        "ctot": 1.0,
        "alpha": 0.0,
        "beta": 0.0,
    }

    scale = corr_params.get("scale", 1.0)
    corr = _get_corr(scan, point, corr_params, preparsed)
    if corr is not None:
        d["ctot"] = corr.ctot_stationary()
        d["alpha"] = corr.gonio.pangles["alpha"]
        d["beta"] = corr.gonio.pangles["beta"]

    # Calculate structure factor F
    if d["I"] > 0.0 and d["Inorm"] > 0.0:
        scale_factor = scale / d["transm"] * d["ctot"] / d["Inorm"]
        d["F"] = float(np.sqrt(scale_factor * d["I"]))
        d["Ferr"] = float(0.5 * np.sqrt(scale_factor) * d["Ierr"] / np.sqrt(d["I"]))
    return d


def _get_corr(scan: dict[str, Any], point: int, corr_params: dict[str, Any], preparsed: bool = False) -> Optional["CtrCorrectionPsic"]:
    """Returns CtrCorrection instance for specified geometry and scan point. Uses correction parameters to configure goniometer and correction factors."""
    # Return None if no goniometer data available
    if "G" not in scan:
        return None

    geom = corr_params.get("geom", "psic")
    beam = corr_params.get("beam_slits", {})
    det = corr_params.get("det_slits")
    sample = corr_params.get("sample")

    if geom == "psic" or bytes_to_str(geom) == "psic":
        gonio = psic_from_spec(scan["G"], preparsed=preparsed)
        _update_psic_angles(gonio, scan, point)
    else:
        raise NotImplementedError(f"Geometry {bytes_to_str(geom)} not implemented")
    return CtrCorrectionPsic(gonio=gonio, beam_slits=beam, det_slits=det, sample=sample)


def _update_psic_angles(gonio: Any, scan: dict[str, Any], point: int, verbose: bool = True) -> None:
    """Updates goniometer angles for specific scan point using scan data. Extracts angle values from scan dictionary handling both scalar and array data."""
    try:
        if hasattr(scan, "dims"):
            npts = int(scan.dims[0])
        else:
            npts = int(scan.get("dims", (1, 0))[0])
    except Exception as e:
        print("Error getting scan dims:", e)
        npts = scan.get("dims", (1, 0))[0]
    try:
        if hasattr(scan, "name"):
            scan_name = bytes_to_str(scan.name) or ""
        else:
            scan_name = bytes_to_str(scan.get("name", "")) or ""
    except Exception as e:
        print("Error getting scan name:", e)
        scan_name = ""

    try:
        if type(scan["phi"]) is float:
            phi = scan["phi"]
        elif len(scan["phi"]) == npts:
            phi = scan["phi"][point]
    except Exception as e:
        print("Error getting phi angle:", e)
        phi = 0.0
    if phi is None and verbose:
        print("Warning no phi angle:", scan_name)

    try:
        if type(scan["chi"]) is float:
            chi = scan["chi"]
        elif len(scan["chi"]) == npts:
            chi = scan["chi"][point]
    except Exception as e:
        print("Error getting chi angle:", e)
        chi = 0.0
    if chi is None and verbose:
        print("Warning no chi angle", scan_name)

    try:
        if type(scan["eta"]) is float:
            eta = scan["eta"]
        elif len(scan["eta"]) == npts:
            eta = scan["eta"][point]
    except Exception as e:
        print("Error getting eta angle:", e)
        eta = 0.0
    if eta is None and verbose:
        print("Warning no eta angle", scan_name)

    try:
        if type(scan["mu"]) is float:
            mu = scan["mu"]
        elif len(scan["mu"]) == npts:
            mu = scan["mu"][point]
    except Exception as e:
        print("Error getting mu angle:", e)
        mu = 0.0
    if mu is None and verbose:
        print("Warning no mu angle", scan_name)

    try:
        if type(scan["nu"]) is float:
            nu = scan["nu"]
        elif len(scan["nu"]) == npts:
            nu = scan["nu"][point]
    except Exception as e:
        print("Error getting nu angle:", e)
        nu = 0.0
    if nu is None and verbose:
        print("Warning no nu angle", scan_name)

    try:
        if type(scan["del"]) is float:
            delta = scan["del"]
        elif len(scan["del"]) == npts:
            delta = scan["del"][point]
    except Exception as e:
        print("Error getting delta angle:", e)
        delta = 0.0
    if delta is None and verbose:
        print("Warning no del angle", scan_name)

    gonio.set_angles(phi=phi, chi=chi, eta=eta, mu=mu, nu=nu, delta=delta)

    # Ensure beta is valid
    if gonio.pangles["beta"] < 0.0:
        gonio.pangles["beta"] = 0.0
        if verbose:
            print("Warning: beta is less than 0.0, setting to 0.0")


class CtrCorrectionPsic:
    """Provides correction factors for measured intensities in Psic geometry. Combines polarization, Lorentz, and geometric area corrections."""

    def __init__(
        self,
        gonio: Optional[Any] = None,
        beam_slits: dict[str, Any] = {},
        det_slits: Optional[dict[str, Any]] = None,
        sample: dict[str, Any] | float = {},
    ) -> None:
        """Initializes correction settings with goniometer and slit geometries. Configures pseudo-angle calculations if needed."""
        self.gonio = gonio
        if self.gonio.calc_psuedo is False:
            self.gonio.calc_psuedo = True
            self.gonio._update_psuedo()
        self.beam_slits = beam_slits
        self.det_slits = det_slits
        self.sample = sample

        # Fraction horizontal polarization
        self.fh = 1.0

    def ctot_stationary(self, plot: bool = False, fig: Optional[Any] = None) -> float:
        """Calculates overall correction factor for stationary measurements. Combines polarization, Lorentz, and area corrections."""
        cp = self.polarization()
        cl = self.lorentz_stationary()
        ca = self.active_area(plot=plot, fig=fig)
        ct = (cp) * (cl) * (ca)
        if plot:
            print("Correction factors (mult by I)")
            print("   Polarization=%f" % cp)
            print("   Lorentz=%f" % cl)
            print("   Area=%f" % ca)
            print("   Total=%f" % ct)
        return ct

    def lorentz_stationary(self) -> float:
        """Calculates Lorentz correction factor for stationary measurements. Returns sine of beta angle."""
        beta = self.gonio.pangles["beta"]
        cl = sind(beta)
        return cl

    def polarization(self) -> float:
        """Calculates polarization correction factor based on beam and detector angles. Uses delta and nu angles from goniometer."""
        delta = self.gonio.angles["delta"]
        nu = self.gonio.angles["nu"]
        p = 1.0 - (cosd(delta) * sind(nu)) ** 2.0
        if p == 0.0:
            cp = 0.0
        else:
            cp = 1.0 / p
        return cp

    def active_area(self, plot: bool = False, fig: Optional[Any] = None) -> float:
        """Calculates correction factor based on beam, detector, and sample area overlap. Requires beam slit specifications."""
        if self.beam_slits == {} or self.beam_slits is None:
            print("Warning beam slits not specified")
            return 1.0
        alpha = self.gonio.pangles["alpha"]
        beta = self.gonio.pangles["beta"]
        if plot:
            print("Alpha = ", alpha, ", Beta = ", beta)
        if alpha < 0.0:
            print("alpha is less than 0.0")
            return 0.0
        elif beta < 0.0:
            print("beta is less than 0.0")
            return 0.0

        bh = self.beam_slits["horz"]
        bv = self.beam_slits["vert"]
        beam = beam_vectors(h=bh, v=bv)

        if self.det_slits is None:
            det = None
        else:
            dh = self.det_slits["horz"]
            dv = self.det_slits["vert"]
            det = det_vectors(h=dh, v=dv, nu=self.gonio.angles["nu"], delta=self.gonio.angles["delta"])

        if isinstance(self.sample, dict):
            sample_dia = self.sample.get("dia", 0.0)
            sample_vecs = self.sample.get("polygon", None)
            sample_angles = self.sample.get("angles", {})

            if sample_vecs is not None and sample_dia <= 0.0:
                sample = sample_vectors(sample_vecs, angles=sample_angles, gonio=self.gonio)
            elif sample_dia > 0.0:
                sample = sample_dia
            else:
                sample = None
        else:
            sample = self.sample

        (A_beam, A_int) = active_area(self.gonio.nm, ki=self.gonio.ki, kr=self.gonio.kr, beam=beam, det=det, sample=sample, plot=plot, fig=fig)
        if A_int == 0.0:
            ca = 0.0
        else:
            ca = A_beam / (A_int**2)
        return ca
