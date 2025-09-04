import copy

import numpy as np

from pds.utils.lattice import Lattice
from pds.utils.mathutil import arccosd, arcsind, cartesian_angle, cartesian_mag, cosd, sind, tand


class Psic:
    """Manage orientation and calculations for PSIC geometry."""

    def __init__(
        self,
        a: float = 10.0,
        b: float = 10.0,
        c: float = 10.0,
        alpha: float = 90.0,
        beta: float = 90.0,
        gamma: float = 90.0,
        lam: float = 1.0,
    ) -> None:
        """Initialize Psic geometry with lattice parameters."""
        # Set lattice and lambda
        self.lattice = Lattice(a, b, c, alpha, beta, gamma, lam)

        # Hold gonio angles - all in degrees
        self.angles: dict[str, float] = {
            "phi": 0.0,
            "chi": 0.0,
            "eta": 0.0,
            "mu": 0.0,
            "nu": 0.0,
            "delta": 0.0,
        }

        # Hold pseudo angles
        self.pangles: dict[str, float] = {}
        self.calc_psuedo: bool = True

        # Hold n (reference) vector in HKL - surface normal vector for pseudo angles
        self.n: np.ndarray = np.array([0.0, 0.0, 1.0], dtype=float)

        # Z matrix and calculated h vector
        self.Z: np.ndarray = np.array([])
        self.Q: np.ndarray = np.array([])
        self.ki: np.ndarray = np.array([])
        self.kr: np.ndarray = np.array([])
        self.h: list[float] = [0.0, 0.0, 0.0]

        # Dummy primary reflection
        tth = self.lattice.tth([0.0, 0.0, 1.0], lam=lam)
        self.or0: dict[str, np.ndarray | float] = {
            "h": np.array([0.0, 0.0, 1.0]),
            "phi": 0.0,
            "chi": 0.0,
            "eta": 0.0,
            "mu": tth / 2.0,
            "nu": tth,
            "delta": 0.0,
            "lam": lam,
        }

        # Dummy secondary reflection
        tth = self.lattice.tth([0.0, 1.0, 0.0], lam=lam)
        self.or1: dict[str, np.ndarray | float] = {
            "h": np.array([0.0, 1.0, 0.0]),
            "phi": 0.0,
            "chi": 0.0,
            "eta": tth / 2.0,
            "mu": 0.0,
            "nu": 0.0,
            "delta": tth,
            "lam": lam,
        }

        # Orientation matrices
        self.U: np.ndarray = np.array([])
        self.B: np.ndarray = np.array([])
        self.UB: np.ndarray = np.array([])
        self.nm: np.ndarray = np.array([])
        self._calc_UB()

    def __repr__(self) -> str:
        """Return PSIC geometry information as a string."""
        lout = self.lattice.__repr__()
        lout += f"Primary:\n   h={self.or0['h'][0]:3.2f},k={self.or0['h'][1]:3.2f},"
        lout += f"l={self.or0['h'][2]:3.2f}, lam={self.or0['lam']:6.6f}\n"
        lout += f"   phi={self.or0['phi']:6.3f},chi={self.or0['chi']:6.3f},"
        lout += f"eta={self.or0['eta']:6.3f},mu={self.or0['mu']:6.3f},"
        lout += f"nu={self.or0['nu']:6.3f},delta={self.or0['delta']:6.3f}\n"

        lout += f"Secondary:\n   h={self.or1['h'][0]:3.2f},k={self.or1['h'][1]:3.2f},"
        lout += f"l={self.or1['h'][2]:3.2f}, lam={self.or1['lam']:6.6f}\n"
        lout += f"   phi={self.or1['phi']:6.3f},chi={self.or1['chi']:6.3f},"
        lout += f"eta={self.or1['eta']:6.3f},mu={self.or1['mu']:6.3f},"
        lout += f"nu={self.or1['nu']:6.3f},delta={self.or1['delta']:6.3f}\n"

        lout += f"Setting:   h={self.h[0]:3.2f},k={self.h[1]:3.2f},l={self.h[2]:3.2f}\n"
        lout += f"   phi={self.angles['phi']:6.3f},chi={self.angles['chi']:6.3f},"
        lout += f"eta={self.angles['eta']:6.3f},mu={self.angles['mu']:6.3f},"
        lout += f"nu={self.angles['nu']:6.3f},delta={self.angles['delta']:6.3f}\n"

        if self.calc_psuedo and self.pangles:
            lout += f"   TTH={self.pangles['tth']:6.3f},"
            lout += f"SIGMA_AZ={self.pangles['sigma_az']:6.3f},"
            lout += f"TAU_AZ={self.pangles['tau_az']:6.3f},"
            lout += f"N_AZ={self.pangles['naz']:6.3f},"
            lout += f"ALPHA={self.pangles['alpha']:6.3f},"
            lout += f"BETA={self.pangles['beta']:6.3f}\n"
            lout += f"   TAU={self.pangles['tau']:6.3f},"
            lout += f"PSI={self.pangles['psi']:6.3f},"
            lout += f"Q_AZ={self.pangles['qaz']:6.3f},"
            lout += f"OMEGA={self.pangles['omega']:6.3f}"

        return lout

    def set_lat(
        self,
        a: float | None = None,
        b: float | None = None,
        c: float | None = None,
        alpha: float | None = None,
        beta: float | None = None,
        gamma: float | None = None,
        lam: float | None = None,
    ) -> None:
        """Update lattice parameters and wavelength."""
        self.lattice.update(a=a, b=b, c=c, alpha=alpha, beta=beta, gamma=gamma, lam=lam)
        self._calc_UB()

    def set_spec_G(self, G: list[float], preparsed: bool = False) -> None:
        """Set lattice and orientation from G array."""
        if not preparsed:
            cell, or0, or1, n = spec_psic_G(G)
        else:
            cell, or0, or1, n = G

        self.n = n
        self.or0 = or0
        self.or1 = or1
        self.lattice = Lattice(*cell)
        self._calc_UB()

    def set_or0(
        self,
        h: list[float] | np.ndarray | None = None,
        phi: float | None = None,
        chi: float | None = None,
        eta: float | None = None,
        mu: float | None = None,
        nu: float | None = None,
        delta: float | None = None,
        lam: float | None = None,
    ) -> None:
        """Set primary orientation reflection parameters."""
        if h is not None:
            self.or0["h"] = np.array(h, dtype=float)
        if phi is not None:
            self.or0["phi"] = float(phi)
        if chi is not None:
            self.or0["chi"] = float(chi)
        if eta is not None:
            self.or0["eta"] = float(eta)
        if mu is not None:
            self.or0["mu"] = float(mu)
        if nu is not None:
            self.or0["nu"] = float(nu)
        if delta is not None:
            self.or0["delta"] = float(delta)
        if lam is not None:
            self.or0["lam"] = float(lam)
        self._calc_UB()

    def set_or1(
        self,
        h: list[float] | np.ndarray | None = None,
        phi: float | None = None,
        chi: float | None = None,
        eta: float | None = None,
        mu: float | None = None,
        nu: float | None = None,
        delta: float | None = None,
        lam: float | None = None,
    ) -> None:
        """Set secondary orientation reflection parameters."""
        if h is not None:
            self.or1["h"] = np.array(h, dtype=float)
        if phi is not None:
            self.or1["phi"] = float(phi)
        if chi is not None:
            self.or1["chi"] = float(chi)
        if eta is not None:
            self.or1["eta"] = float(eta)
        if mu is not None:
            self.or1["mu"] = float(mu)
        if nu is not None:
            self.or1["nu"] = float(nu)
        if delta is not None:
            self.or1["delta"] = float(delta)
        if lam is not None:
            self.or1["lam"] = float(lam)
        self._calc_UB()

    def swap_or(self) -> None:
        """Swap primary and secondary reflections."""
        tmp = copy.copy(self.or0)
        self.or0 = copy.copy(self.or1)
        self.or1 = tmp
        self._calc_UB()

    def _calc_UB(self) -> None:
        """Calculate orientation matrix U."""
        cross = np.cross
        norm = np.linalg.norm

        # Calculate B matrix
        (a, b, c, alp, bet, gam) = self.lattice.cell()
        (ar, br, cr, alpr, betr, gamr) = self.lattice.rcell()
        B = np.array([[ar, br * cosd(gamr), cr * cosd(betr)], [0.0, br * sind(gamr), -cr * sind(betr) * cosd(alp)], [0.0, 0.0, 1.0 / c]])
        self.B = B

        # Calculate Z and Q for reflections
        Z1 = calc_Z(self.or0["phi"], self.or0["chi"], self.or0["eta"], self.or0["mu"])
        Q1 = calc_Q(self.or0["nu"], self.or0["delta"], self.or0["lam"])
        #
        Z2 = calc_Z(self.or1["phi"], self.or1["chi"], self.or1["eta"], self.or1["mu"])
        Q2 = calc_Q(self.or1["nu"], self.or1["delta"], self.or1["lam"])

        # Calculate phi frame coordinates for diffraction vectors
        vphi_1 = np.dot(np.linalg.inv(Z1), (Q1 / (2.0 * np.pi)))
        vphi_2 = np.dot(np.linalg.inv(Z2), (Q2 / (2.0 * np.pi)))

        # Calculate cartesian coordinates for h vectors
        hc_1 = np.dot(self.B, self.or0["h"])
        hc_2 = np.dot(self.B, self.or1["h"])

        # Define normalized vectors from hc vectors
        tc_1 = hc_1 / norm(hc_1)
        tc_3 = cross(tc_1, hc_2) / norm(cross(tc_1, hc_2))
        tc_2 = cross(tc_3, tc_1) / norm(cross(tc_3, tc_1))

        # Define tphi vectors from vphi vectors
        tphi_1 = vphi_1 / norm(vphi_1)
        tphi_3 = cross(tphi_1, vphi_2) / norm(cross(tphi_1, vphi_2))
        tphi_2 = cross(tphi_3, tphi_1) / norm(cross(tphi_3, tphi_1))

        # Define matrices
        Tc = np.transpose(np.array([tc_1, tc_2, tc_3]))
        Tphi = np.transpose(np.array([tphi_1, tphi_2, tphi_3]))

        # Calculate orientation matrix U
        self.U = np.dot(Tphi, np.linalg.inv(Tc))

        # Calculate UB
        self.UB = np.dot(self.U, self.B)

        # Update h and pseudo angles
        self.set_angles()

    def set_angles(
        self,
        phi: float | None = None,
        chi: float | None = None,
        eta: float | None = None,
        mu: float | None = None,
        nu: float | None = None,
        delta: float | None = None,
    ) -> None:
        """Set goniometer angles in degrees."""
        if phi is not None:
            self.angles["phi"] = float(phi)
        if chi is not None:
            self.angles["chi"] = float(chi)
        if eta is not None:
            self.angles["eta"] = float(eta)
        if mu is not None:
            self.angles["mu"] = float(mu)
        if nu is not None:
            self.angles["nu"] = float(nu)
        if delta is not None:
            self.angles["delta"] = float(delta)
        # Update h and calculate Z
        self._calc_h()
        # Update pseudo angles
        self._update_psuedo()

    def _calc_h(self) -> None:
        """Calculate hkl values for the diffraction condition."""

        self.Z = calc_Z(phi=self.angles["phi"], chi=self.angles["chi"], eta=self.angles["eta"], mu=self.angles["mu"])
        (Q, ki, kr) = calc_Q(self.angles["nu"], self.angles["delta"], self.lattice.lam, ret_k=True)
        self.Q = Q
        self.ki = ki
        self.kr = kr

        hphi = np.dot(np.linalg.inv(self.Z), self.Q) / (2.0 * np.pi)
        h = np.dot(np.linalg.inv(self.UB), hphi)
        self.h = h

    def set_n(self, n: list[float] | np.ndarray | None = None) -> None:
        """Set reference vector for pseudo angles."""
        if n is None:
            n = [0.0, 0.0, 1.0]
        self.n = np.array(n, dtype=float)
        self._update_psuedo()

    def calc_n(self, fchi: float = 0.0, fphi: float = 0.0) -> None:
        """Calculate hkl values of reference vector."""
        # Define polar angles
        sig_az = -fchi
        tau_az = -fphi

        # Convert chi and phi to polar coordinates
        if sig_az < 0.0:
            sig_az = -1.0 * sig_az
            if tau_az < 0.0:
                tau_az = 180.0 + tau_az
            elif tau_az > 0.0:
                tau_az = tau_az - 180.0

        # Define n in unrotated lab frame
        n_phi = np.array([sind(sig_az) * cosd(tau_az), -sind(sig_az) * sind(tau_az), cosd(sig_az)])
        # Define n in HKL
        n_hkl = np.dot(np.linalg.inv(self.UB), n_phi)
        n_hkl = n_hkl / np.max(np.abs(n_hkl))

        # If l-component is negative, reverse direction
        if n_hkl[2] < 0.0:
            n_hkl = -1.0 * n_hkl

        # Set n to trigger recalculation
        self.set_n(n_hkl)

    def _update_psuedo(self) -> None:
        """Compute pseudo angles."""
        self.pangles = {}
        if self.calc_psuedo:
            self._calc_tth()
            self._calc_nm()
            self._calc_sigma_az()
            self._calc_tau_az()
            self._calc_naz()
            self._calc_alpha()
            self._calc_beta()
            self._calc_tau()
            self._calc_psi()
            self._calc_qaz()
            self._calc_omega()

    def _calc_tth(self) -> None:
        """Calculate 2Theta scattering angle."""
        nu = self.angles["nu"]
        delta = self.angles["delta"]
        tth = arccosd(cosd(delta) * cosd(nu))
        self.pangles["tth"] = tth

    def _calc_nm(self) -> None:
        """Calculate rotated cartesian lab indices of reference vector."""
        # Calculate n in rotated lab frame as a unit vector
        n = self.n
        Z = self.Z
        UB = self.UB
        nm = np.dot(np.dot(Z, UB), n)
        nm = nm / cartesian_mag(nm)
        self.nm = nm

    def _calc_sigma_az(self) -> None:
        """Calculate sigma_az angle between z-axis and n in phi frame."""
        # Calculate n in unrotated lab frame as a unit vector
        n_phi = np.dot(self.UB, self.n)
        n_phi = n_phi / cartesian_mag(n_phi)

        # Result of acosd is between 0 and pi; get sign from x-component
        sigma_az = arccosd(n_phi[2])
        self.pangles["sigma_az"] = sigma_az

    def _calc_tau_az(self) -> None:
        """Calculate tau_az angle between n projection and x-axis."""
        # Calculate n in unrotated lab frame as a unit vector
        n_phi = np.dot(self.UB, self.n)
        n_phi = n_phi / cartesian_mag(n_phi)

        tau_az = np.arctan2(-n_phi[1], n_phi[0])
        tau_az = tau_az * 180.0 / np.pi
        self.pangles["tau_az"] = tau_az

    def _calc_naz(self) -> None:
        """Calculate naz angle between reference vector n and yz plane."""
        # Get normalized reference vector in cartesian lab frame
        nm = self.nm
        naz = np.arctan2(nm[0], nm[2])
        naz = np.degrees(naz)
        self.pangles["naz"] = naz

    def _calc_alpha(self) -> None:
        """Calculate alpha incidence angle."""
        nm = self.nm
        ki = np.array([0.0, -1.0, 0.0])
        alpha = arcsind(np.dot(nm, ki))
        self.pangles["alpha"] = alpha

    def _calc_beta(self) -> None:
        """Calculate beta exit angle."""
        # Calculate normalized kr
        nm = self.nm
        kr = self.kr / cartesian_mag(self.kr)
        beta = arcsind(np.dot(nm, kr))
        self.pangles["beta"] = beta

    def _calc_tau(self) -> None:
        """Calculate tau angle between n and scattering plane."""
        tau = cartesian_angle(self.Q, self.nm)
        self.pangles["tau"] = tau

    def _calc_psi(self) -> None:
        """Calculate psi azimuthal angle of n with respect to Q."""
        tau = self.pangles["tau"]
        tth = self.pangles["tth"]
        alpha = self.pangles["alpha"]
        # Calculate intermediate values for psi
        xx = cosd(tau) * sind(tth / 2.0) - sind(alpha)
        denom = sind(tau) * cosd(tth / 2.0)
        if denom == 0:
            self.pangles["psi"] = 0.0
            return
        xx = xx / denom
        psi = arccosd(xx)
        self.pangles["psi"] = psi

    def _calc_qaz(self) -> None:
        """Calculate qaz angle between Q and yz plane."""
        nu = self.angles["nu"]
        delta = self.angles["delta"]
        qaz = np.arctan2(sind(delta), cosd(delta) * sind(nu))
        qaz = np.degrees(qaz)
        self.pangles["qaz"] = qaz

    def _calc_omega(self) -> None:
        """Calculate omega angle between Q and plane perpendicular to chi axis."""
        eta = self.angles["eta"]
        mu = self.angles["mu"]
        H = np.array([[cosd(eta), sind(eta), 0.0], [-sind(eta), cosd(eta), 0.0], [0.0, 0.0, 1.0]], float)
        M = np.array([[1.0, 0.0, 0.0], [0.0, cosd(mu), -sind(mu)], [0.0, sind(mu), cosd(mu)]], float)
        # Check multiplication order for T
        T = np.dot(M.transpose(), H.transpose())
        Qpp = np.dot(T, self.Q)
        # Calculate omega using cartesian_angle
        omega = cartesian_angle([Qpp[0], 0, Qpp[2]], Qpp)
        self.pangles["omega"] = omega


def psic_from_spec(G: list[float] | None, angles: dict[str, float] | None = None, preparsed: bool = False) -> Psic:
    """Create Psic instance from spec G array and angles."""
    if angles is None:
        angles = {}
    gonio = Psic()
    if G is not None:
        gonio.set_spec_G(G, preparsed)
    gonio.set_angles(**angles)
    return gonio


def spec_psic_G(G: list[float]) -> tuple[np.ndarray, dict, dict, np.ndarray]:
    """Parse lattice and orientation data from spec G array."""
    # Azimuthal reference vector n (hkl)
    n = np.array(G[3:6], dtype=float)

    # Lattice parameters a,b,c,alp,bet,gam
    cell = G[22:28]
    # Add lambda to end of cell
    cell.append(G[66])
    cell = np.array(cell, dtype=float)

    # Primary reflection or0
    or0 = {}
    or0["h"] = np.array(G[34:37], dtype=float)
    or0.update(_spec_or_angles(np.array(G[40:46], dtype=float)))
    or0["lam"] = float(G[52])

    # Secondary reflection or1
    or1 = {}
    or1["h"] = np.array(G[37:40], dtype=float)
    or1.update(_spec_or_angles(np.array(G[46:52], dtype=float)))
    or1["lam"] = float(G[53])

    return (cell, or0, or1, n)


def _spec_or_angles(angles: np.ndarray, calc_kappa: bool = False) -> dict[str, float]:
    """Convert spec angle array to orientation reflection dictionary."""
    # Extract angles from spec array
    delta = angles[0]
    eta = angles[1]
    chi = angles[2]
    phi = angles[3]
    nu = angles[4]
    mu = angles[5]

    # Calculate kappa angles if requested
    if calc_kappa:
        kap_alp = 50.031
        keta = eta - arcsind(-tand(chi / 2.0) / tand(kap_alp))
        kphi = phi - arcsind(-tand(chi / 2.0) / tand(kap_alp))
        kappa = arcsind(sind(chi / 2.0) / sind(kap_alp))
        return {"phi": phi, "chi": chi, "eta": eta, "mu": mu, "delta": delta, "nu": nu, "keta": keta, "kphi": kphi, "kappa": kappa}
    else:
        return {"phi": phi, "chi": chi, "eta": eta, "mu": mu, "delta": delta, "nu": nu}


def calc_Z(phi: float = 0.0, chi: float = 0.0, eta: float = 0.0, mu: float = 0.0) -> np.ndarray:
    """Calculate PSIC goniometer rotation matrix Z from sample angles."""
    P = np.array([[cosd(phi), sind(phi), 0.0], [-sind(phi), cosd(phi), 0.0], [0.0, 0.0, 1.0]], float)
    X = np.array([[cosd(chi), 0.0, sind(chi)], [0.0, 1.0, 0.0], [-sind(chi), 0.0, cosd(chi)]], float)
    H = np.array([[cosd(eta), sind(eta), 0.0], [-sind(eta), cosd(eta), 0.0], [0.0, 0.0, 1.0]], float)
    M = np.array([[1.0, 0.0, 0.0], [0.0, cosd(mu), -sind(mu)], [0.0, sind(mu), cosd(mu)]], float)
    Z = np.dot(np.dot(np.dot(M, H), X), P)
    return Z


def calc_Q(nu: float = 0.0, delta: float = 0.0, lam: float = 1.0, ret_k: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray] | np.ndarray:
    """Calculate Q vector in cartesian lab frame."""
    (ki, kr) = calc_kvecs(nu=nu, delta=delta, lam=lam)
    Q = kr - ki
    if ret_k:
        return (Q, ki, kr)
    else:
        return Q


def calc_kvecs(nu: float = 0.0, delta: float = 0.0, lam: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """Calculate incident and reflected beam vectors in lab frame."""
    k = 2.0 * np.pi / lam
    ki = k * np.array([0.0, 1.0, 0.0], dtype=float)
    kr = k * np.array([sind(delta), cosd(nu) * cosd(delta), sind(nu) * cosd(delta)], dtype=float)
    return (ki, kr)


def calc_D(nu: float = 0.0, delta: float = 0.0) -> np.ndarray:
    """Calculate detector rotation matrix."""
    D1 = np.array([[cosd(delta), sind(delta), 0.0], [-sind(delta), cosd(delta), 0.0], [0.0, 0.0, 1.0]])

    D2 = np.array([[1.0, 0.0, 0.0], [0.0, cosd(nu), -sind(nu)], [0.0, sind(nu), cosd(nu)]])

    D = np.dot(D2, D1)
    return D


def beam_vectors(h: float = 1.0, v: float = 1.0) -> list[np.ndarray]:
    """Compute beam aperture vectors in lab frame."""
    # Define beam vectors in lab frame
    bh = np.array([0.0, 0.0, 0.5 * h])
    bv = np.array([0.5 * v, 0.0, 0.0])

    # Calculate corners of beam aperture
    a = bv + bh
    b = bv - bh
    c = -bv - bh
    d = -bv + bh
    beam = [a, b, c, d]

    return beam


def det_vectors(h: float = 1.0, v: float = 1.0, nu: float = 0.0, delta: float = 0.0) -> list[np.ndarray]:
    """Compute detector aperture vectors in lab frame."""
    # Define detector vectors in lab frame with rotation
    dh = np.array([0.0, 0.0, 0.5 * h])
    dv = np.array([0.5 * v, 0.0, 0.0])
    D = calc_D(nu=nu, delta=delta)
    dh = np.dot(D, dh)
    dv = np.dot(D, dv)

    # Calculate corners of detector aperture
    e = dv + dh
    f = dv - dh
    g = -dv - dh
    h = -dv + dh
    det = [e, f, g, h]

    return det


def sample_vectors(sample: list[list[float]] | None, angles: dict[str, float] | None = None, gonio: Psic | None = None) -> list[np.ndarray] | None:
    """Transform sample vectors between coordinate frames."""
    if sample is None:
        return None
    if len(sample) < 3:
        print("Sample polygon must be 3 or more points")
        return None
    # Transform to phi frame if angles provided
    if angles is None:
        angles = {}
    if len(angles) > 0:
        # Calculate sample rotation matrix
        Z = calc_Z(**angles)
        Zinv = np.linalg.inv(Z)
        polygon_phi = []
        # Convert 2D vectors to 3D by adding zero for z
        for p in sample:
            if len(p) == 2:
                p = [p[0], p[1], 0.0]
            p_phi = np.dot(Zinv, p)
            polygon_phi.append(p_phi)
    else:
        polygon_phi = sample

    # Rotate to m-frame if gonio provided, otherwise return phi frame
    polygon = []
    if gonio is not None:
        for p in polygon_phi:
            # Convert 2D vectors to 3D by adding zero for z
            if len(p) == 2:
                p = [p[0], p[1], 0.0]
            p_m = np.dot(gonio.Z, p)
            polygon.append(p_m)
    else:
        polygon = polygon_phi

    return polygon
