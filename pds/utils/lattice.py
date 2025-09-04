from typing import Optional

import numpy as np

from pds.utils.mathutil import arccosd, arcsind, cosd, sind, tand


class Lattice:
    """Crystal lattice for metric tensor calculations and coordinate transformations."""

    def __init__(
        self,
        a: float = 10.0,
        b: float = 10.0,
        c: float = 10.0,
        alpha: float = 90.0,
        beta: float = 90.0,
        gamma: float = 90.0,
        lam: float = 1.5406,
    ) -> None:
        """Initialize lattice with unit cell parameters."""
        self.update(a=a, b=b, c=c, alpha=alpha, beta=beta, gamma=gamma, lam=lam)

    def __repr__(self) -> str:
        """Display lattice parameters in readable format."""
        lout = f"a={self.a:6.5f}, b={self.b:6.5f}, c={self.c:6.5f}"
        lout += f", alpha={self.alpha:6.5f}, beta={self.beta:6.5f}, gamma={self.gamma:6.5f}\n"
        lout += f"ar={self.ar:6.5f}, br={self.br:6.5f}, cr={self.cr:6.5f}"
        lout += f", alphar={self.alphar:6.5f}, betar={self.betar:6.5f}, gammar={self.gammar:6.5f}\n"
        lout += f"Default wavelength for angle calculations={self.lam:6.5f}\n"

        return lout

    def update(
        self,
        a: Optional[float] = None,
        b: Optional[float] = None,
        c: Optional[float] = None,
        alpha: Optional[float] = None,
        beta: Optional[float] = None,
        gamma: Optional[float] = None,
        lam: Optional[float] = None,
    ) -> None:
        """Update lattice parameters and recalculate derived quantities."""
        if a is not None:
            self.a = float(a)
        if b is not None:
            self.b = float(b)
        if c is not None:
            self.c = float(c)
        if alpha is not None:
            self.alpha = float(alpha)
        if beta is not None:
            self.beta = float(beta)
        if gamma is not None:
            self.gamma = float(gamma)
        if lam is not None:
            self.lam = float(lam)
        # update calc quantities
        self._calc_g()

    def cell(self) -> np.ndarray:
        """Return array of real lattice cell parameters."""
        return np.array([self.a, self.b, self.c, self.alpha, self.beta, self.gamma], dtype=float)

    def rcell(self) -> np.ndarray:
        """Return array of reciprocal lattice cell parameters."""
        return np.array([self.ar, self.br, self.cr, self.alphar, self.betar, self.gammar], dtype=float)

    def _calc_g(self) -> None:
        """Calculate metric tensors and reciprocal lattice parameters."""
        (a, b, c, alp, bet, gam) = self.cell()
        # real metric tensor
        self.g = np.array(
            [[a * a, a * b * cosd(gam), a * c * cosd(bet)], [b * a * cosd(gam), b * b, b * c * cosd(alp)], [c * a * cosd(bet), c * b * cosd(alp), c * c]]
        )
        # recip lattice metric tensor and recip lattice params
        self.gr = np.linalg.inv(self.g)
        self.ar = np.sqrt(self.gr[0, 0])
        self.br = np.sqrt(self.gr[1, 1])
        self.cr = np.sqrt(self.gr[2, 2])
        self.alphar = arccosd(self.gr[1, 2] / (self.br * self.cr))
        self.betar = arccosd(self.gr[0, 2] / (self.ar * self.cr))
        self.gammar = arccosd(self.gr[0, 1] / (self.ar * self.br))

    def vol(self, recip: bool = False) -> float:
        """Calculate cell volume in real (ang³) or reciprocal (ang⁻³) space."""
        g = self.gr if recip else self.g
        det = np.linalg.det(g)
        return np.sqrt(det) if det > 0 else 0.0

    def dot(self, u: np.ndarray | list[float], v: np.ndarray | list[float], recip: bool = False) -> float:
        """Calculate generalized dot product using metric tensor."""
        g = self.gr if recip else self.g
        u_arr = np.array(u, dtype=float)
        v_arr = np.array(v, dtype=float)
        return np.dot(u_arr, np.dot(g, v_arr))

    def mag(self, v: np.ndarray | list[float], recip: bool = False) -> float:
        """Calculate vector magnitude using metric tensor."""
        return np.sqrt(self.dot(v, v, recip=recip))

    def angle(self, u: np.ndarray | list[float], v: np.ndarray | list[float], recip: bool = False) -> float:
        """Calculate angle between two vectors in degrees."""
        uv = self.dot(u, v, recip=recip)
        um = self.mag(u, recip=recip)
        vm = self.mag(v, recip=recip)
        arg = uv / (um * vm)
        if np.abs(arg) > 1.0:
            arg = arg / np.abs(arg)
        return arccosd(arg)

    def angle_rr(self, x: np.ndarray | list[float], h: np.ndarray | list[float]) -> float:
        """Calculate angle between real space vector and reciprocal space vector."""
        x_arr = np.array(x, dtype=float)
        h_arr = np.array(h, dtype=float)
        hx = np.sum(x_arr * h_arr)
        xm = self.mag(x_arr, recip=False)
        hm = self.mag(h_arr, recip=True)
        arg = hx / (hm * xm)
        if np.abs(arg) > 1.0:
            arg = arg / np.abs(arg)
        return arccosd(arg)

    def d(self, hkl: np.ndarray | list[float]) -> float:
        """Calculate d-spacing for given Miller indices [h,k,l]."""
        if len(hkl) != 3:
            print("Error need an array of [h,k,l]")
            return 0.0
        h = self.mag(hkl, recip=True)
        return 1.0 / h if h != 0.0 else 0.0

    def tth(self, hkl: np.ndarray | list[float], lam: Optional[float] = None) -> float:
        """Calculate 2θ diffraction angle for given Miller indices and wavelength."""
        if lam is not None:
            self.lam = float(lam)
        d = self.d(hkl)
        if d == 0.0:
            return 0.0
        r = self.lam / (2.0 * d)
        if np.abs(r) > 1.0:
            r = r / np.abs(r)
        return 2.0 * arcsind(r)

    def dvec(self, hkl: np.ndarray | list[float]) -> np.ndarray:
        """Calculate real space vector normal to plane with magnitude equal to d-spacing."""
        dvec = self.recip_to_real(hkl)
        dspc = self.d(hkl)
        if dspc == 0:
            return np.array([0.0, 0.0, 0.0])
        return (dspc**2.0) * dvec

    def recip_to_real(self, hkl: np.ndarray | list[float]) -> np.ndarray:
        """Transform reciprocal lattice vector to real lattice indices."""
        hkl_arr = np.array(hkl, dtype=float)
        return np.dot(self.gr, hkl_arr)

    def real_to_recip(self, v: np.ndarray | list[float]) -> np.ndarray:
        """Transform real lattice vector to reciprocal lattice indices."""
        v_arr = np.array(v, dtype=float)
        return np.dot(v_arr, self.g)


class LatticeTransform:
    """Generalized lattice transformations for basis rotations and shifts."""

    def __init__(
        self,
        lattice: Lattice,
        Va: Optional[np.ndarray | list[float]] = None,
        Vb: Optional[np.ndarray | list[float]] = None,
        Vc: Optional[np.ndarray | list[float]] = None,
        shift: Optional[np.ndarray | list[float]] = None,
    ) -> None:
        """Initialize lattice transformation with basis vectors and shift."""
        self.lattice = lattice
        self.Va = np.array([1.0, 0.0, 0.0])
        self.Vb = np.array([0.0, 1.0, 0.0])
        self.Vc = np.array([0.0, 0.0, 1.0])
        self.shift = np.array([0.0, 0.0, 0.0])
        self._update(Va=Va, Vb=Vb, Vc=Vc, shift=shift)

    def basis(
        self,
        Va: Optional[np.ndarray | list[float]] = None,
        Vb: Optional[np.ndarray | list[float]] = None,
        Vc: Optional[np.ndarray | list[float]] = None,
        shift: Optional[np.ndarray | list[float]] = None,
    ) -> None:
        """Define new basis vectors in fractional coordinates of original lattice."""
        self._update(Va=Va, Vb=Vb, Vc=Vc, shift=shift)

    def _update(
        self,
        Va: Optional[np.ndarray | list[float]] = None,
        Vb: Optional[np.ndarray | list[float]] = None,
        Vc: Optional[np.ndarray | list[float]] = None,
        shift: Optional[np.ndarray | list[float]] = None,
    ) -> None:
        """Update transformation matrices from basis vectors."""
        if Va is not None:
            self.Va = np.array(Va, dtype=float)
        if Vb is not None:
            self.Vb = np.array(Vb, dtype=float)
        if Vc is not None:
            self.Vc = np.array(Vc, dtype=float)
        if shift is not None:
            self.shift = np.array(shift, dtype=float)
        F = [self.Va, self.Vb, self.Vc]
        self.F = np.array(F, dtype=float)
        self.G = np.linalg.inv(self.F)
        self.M = self.G.transpose()
        self.N = self.F.transpose()

    def cartesian(self, shift: list[float] = [0.0, 0.0, 0.0]) -> None:
        """Calculate cartesian basis with a parallel to a, b in ab plane."""
        (a, b, c, alp, bet, gam) = self.lattice.cell()
        (ar, br, cr, alpr, betr, gamr) = self.lattice.rcell()
        Va = [1.0 / a, 0.0, 0.0]
        Vb = [-1.0 / (a * tand(gam)), 1.0 / (b * sind(gam)), 0.0]
        Vc = [ar * cosd(betr), br * cosd(alpr), cr]
        self.basis(Va=Va, Vb=Vb, Vc=Vc, shift=shift)

    def xp(self, x: np.ndarray | list[float]) -> np.ndarray:
        """Transform vector from original to new basis."""
        x_arr = np.array(x, dtype=float)
        if self.shift.sum() != 0.0:
            x_arr = x_arr - self.shift
        return np.dot(self.M, x_arr)

    def x(self, xp: np.ndarray | list[float]) -> np.ndarray:
        """Transform vector from new to original basis."""
        xp_arr = np.array(xp, dtype=float)
        x = np.dot(self.N, xp_arr)
        if self.shift.sum() != 0.0:
            x = x + self.shift
        return x

    def hp(self, h: np.ndarray | list[float]) -> np.ndarray:
        """Transform reciprocal lattice vector from original to new basis."""
        h_arr = np.array(h, dtype=float)
        return np.dot(self.F, h_arr)

    def h(self, hp: np.ndarray | list[float]) -> np.ndarray:
        """Transform reciprocal lattice vector from new to original basis."""
        hp_arr = np.array(hp, dtype=float)
        return np.dot(self.G, hp_arr)

    def plat(self) -> Lattice:
        """Return Lattice instance for the transformed basis."""
        a = self.lattice.mag(self.Va)
        b = self.lattice.mag(self.Vb)
        c = self.lattice.mag(self.Vc)
        alp = self.lattice.angle(self.Vb, self.Vc)
        bet = self.lattice.angle(self.Va, self.Vc)
        gam = self.lattice.angle(self.Va, self.Vb)
        return Lattice(a, b, c, alp, bet, gam)
