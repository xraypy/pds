import numpy as np

from pds.utils.active_area import active_area
from pds.utils.gonio_psic import beam_vectors, det_vectors, psic_from_spec, sample_vectors
from pds.utils.mathutil import cosd, sind


def image_point_F(scan, point, Iitg="I", Inorm="io", Ierr="Ierr", Ibgr="Ibgr", transm="transm", corr_params={}, preparsed=False):
    """Computes the structure factor F and its error for a single scan point in an image scan, applying intensity corrections."""

    d = {"I": 0.0, "Inorm": 0.0, "Ierr": 0.0, "Ibgr": 0.0, "transm": 0.0, "F": 0.0, "Ferr": 0.0, "ctot": 1.0, "alpha": 0.0, "beta": 0.0}
    d["I"] = scan[Iitg][point]
    d["Inorm"] = scan[Inorm][point]
    d["Ierr"] = scan[Ierr][point]
    d["Ibgr"] = scan[Ibgr][point]
    d["transm"] = scan[transm][point]

    if corr_params is None:
        d["ctot"] = 1.0
        scale = 1.0
    else:
        # compute correction factors
        scale = corr_params.get("scale")
        if scale is None:
            scale = 1.0
        scale = float(scale)
        corr = _get_corr(scan, point, corr_params, preparsed)
        if corr is None:
            d["ctot"] = 1.0
        else:
            d["ctot"] = corr.ctot_stationary()
            d["alpha"] = corr.gonio.pangles["alpha"]
            d["beta"] = corr.gonio.pangles["beta"]

    # compute F
    if d["I"] <= 0.0 or d["Inorm"] <= 0.0:
        d["F"] = 0.0
        d["Ferr"] = 0.0
    else:
        scale = scale / d["transm"] * d["ctot"] / d["Inorm"]
        # scale = scale * d['ctot']/d['Inorm']
        d["F"] = np.sqrt(scale * d["I"])
        d["Ferr"] = 0.5 * scale**0.5 * d["Ierr"] / d["I"] ** 0.5
    return d


def _get_corr(scan, point, corr_params, preparsed=False):
    """Returns a CtrCorrection instance for the specified geometry and scan point, using provided correction parameters."""

    geom = corr_params.get("geom", "psic")
    beam = corr_params.get("beam_slits", {})
    det = corr_params.get("det_slits")
    sample = corr_params.get("sample")

    # get gonio instance for corrections
    if geom == "psic" or geom == b"psic":
        # Implement the handling for 'psic' geometry
        gonio = psic_from_spec(scan["G"], preparsed=preparsed)
        _update_psic_angles(gonio, scan, point)
    else:
        raise NotImplementedError(f"Geometry {geom} not implemented")
    return CtrCorrectionPsic(gonio=gonio, beam_slits=beam, det_slits=det, sample=sample)


def _update_psic_angles(gonio, scan, point, verbose=True):
    """Updates the goniometer angles for a specific scan point using values from the scan data."""

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
            scan_name = scan.name
        else:
            scan_name = scan.get("name", "")
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
    """Provides correction factors for measured intensities in Psic geometry, including polarization, Lorentz, and geometric area corrections."""

    def __init__(self, gonio=None, beam_slits={}, det_slits=None, sample={}):
        """Initializes correction settings with goniometer, beam slits, detector slits, and sample geometry for the Psic setup."""

        self.gonio = gonio
        if self.gonio.calc_psuedo is False:
            self.gonio.calc_psuedo = True
            self.gonio._update_psuedo()
        self.beam_slits = beam_slits
        self.det_slits = det_slits
        self.sample = sample

        # fraction horz polarization
        self.fh = 1.0

    def ctot_stationary(self, plot=False, fig=None):
        """Calculates the overall correction factor for stationary (image) measurements, combining polarization, Lorentz, and area corrections."""

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

    def lorentz_stationary(self):
        """Calculates the Lorentz correction factor for stationary (image) measurements."""

        beta = self.gonio.pangles["beta"]
        cl = sind(beta)
        return cl

    def polarization(self):
        """Calculates the polarization correction factor based on the beam and detector angles."""

        delta = self.gonio.angles["delta"]
        nu = self.gonio.angles["nu"]
        p = 1.0 - (cosd(delta) * sind(nu)) ** 2.0
        if p == 0.0:
            cp = 0.0
        else:
            cp = 1.0 / p

        return cp

    def active_area(self, plot=False, fig=None):
        """Calculates a correction factor based on the overlap between the beam, detector, and sample areas."""

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

        # get beam vectors
        bh = self.beam_slits["horz"]
        bv = self.beam_slits["vert"]
        beam = beam_vectors(h=bh, v=bv)

        # get detector vectors
        if self.det_slits is None:
            det = None
        else:
            dh = self.det_slits["horz"]
            dv = self.det_slits["vert"]
            det = det_vectors(h=dh, v=dv, nu=self.gonio.angles["nu"], delta=self.gonio.angles["delta"])

        # get sample polygon
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

        # compute active area correction
        (A_beam, A_int) = active_area(self.gonio.nm, ki=self.gonio.ki, kr=self.gonio.kr, beam=beam, det=det, sample=sample, plot=plot, fig=fig)
        if A_int == 0.0:
            ca = 0.0
        else:
            ca = A_beam / (A_int**2)
        return ca
