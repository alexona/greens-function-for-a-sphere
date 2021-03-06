# -*- coding: utf-8 -*-
"""
Moduł służący do obliczania funkcji Greena dla jednorodnej sfery zanurzonej
w jednorodnym ośrodku.

Resources
---------
Bessel functions in SciPy
    https://www.johndcook.com/blog/bessel_python/
Spherical Harmonic
    http://functions.wolfram.com/HypergeometricFunctions/SphericalHarmonicYGeneral/


Z modułu scipy.special importujemy
----------------------------------
sph_harm
    - harmoniki sferyczne
spherical_jn
    - sferyczna funkcja Bessela pierwszego rodzaju
spherical_yn
    - sferyczna funkcja Bessela drugiego rodzaju


Zostały stworzone funkcje
-------------------------
_sph_harm
    - harmoniki sferyczne z definicji fizycznej (funkcja prywatna dla modulu)
spherical_hn
    - sferyczna funkcja Hankel'a pierwszego rodzaju
spherical_gradient
    - gradient danej funkcji sferycznej
sph_harm_diff_theta
    - pochodna cząstkowa harmonik sferycznych po theta
sph_harm_diff_phi
    - pochodna cząstkowa harmonik sferycznych po phi
sph_harm_gradient
    - gradient z harmonik sferycznych
vsh1, vsh2, vsh3
    - wektorowe harmoniki sferyczne


Convention
----------
cartesian coordinates
    :(x, y, z):
spherical coordinates
    :(r, theta, phi): where
        :r:     radial coordinate; must be in (0, oo);
        :theta: polar coordinate; must be in [0, pi];
        :phi:   azimuthal coordinate; must be in [0, 2*pi];
"""

__author__ = "Monika Kubek"
__credits__ = ["Monika Kubek", "Karolina Słowik"]
__email__ = "kubek.monika@wp.pl"


import numpy as np
from scipy.special import spherical_jn, spherical_yn, sph_harm
from scipy.constants import speed_of_light


def _sph_harm(m, n, theta, phi):
    """
    Redefine spherical harmonics from scipy.special
    to match physics convention.

    Parameters
    ----------
    m : int, array_like
        Order of the harmonic (int); ``|m| <= n``.
    n : int, array_like
       Degree of the harmonic (int); ``n >= 0``. This is
       often denoted by ``l`` (lower case L) in descriptions of
       spherical harmonics.
    theta : array_like
       Polar (colatitudinal) coordinate; must be in ``[0, pi]``.
    phi : array_like
       Azimuthal (longitudinal) coordinate; must be in ``[0, 2*pi]``.

    Returns
    -------
    y_mn : complex float
       The harmonic :math:`Y^m_n` sampled at ``theta`` and ``phi``.
    """
    return sph_harm(m, n, phi, theta)


def spherical_hn(l, r, derivative=False):
    """
    Spherical Hankel function of the First kind or its derivative.

    Parameters
    ----------
    l : int
        Order of the Hankel function; (l >= 0).
    r : float, array_like
        Argument of the Hankel function.
    derivative : boolean, default False
        Set to True to return a derivative of the function.

    References
    ----------
    .. [1]  Weisstein, Eric W. "Spherical Hankel Function of the First Kind."
            From MathWorld--A Wolfram Web Resource.
            http://mathworld.wolfram.com/SphericalHankelFunctionoftheFirstKind.html

    .. [2]  NIST
            http://dlmf.nist.gov/10.47
    """
    sph_func = lambda l, r: spherical_jn(l, r) + 1j * spherical_yn(l, r)

    if derivative == False:
        return sph_func(l, r)
    elif derivative == True:
        try:
            return (l / r) * sph_func(l, r) - sph_func(l+1, r)
        except ZeroDivisionError:
            return (l * sph_func(l-1, r) - (l+1) * sph_func(l+1, r)) / (2*l + 1)
    else:
        raise TypeError("derivative must be a boolean value")


def spherical_gradient(l, r, sph_func):
    """
    Calculate a gradient of a chosen spherical function (_jn, _yn, _hn).

    Parameters
    ----------
    l: int
        Order of the spherical function.
    r: float
        Is a real, positive variable.
    sph_func: func
        Spherical function which gradient is calculated.

    Returns
    -------
    grad: ndarray
        Gradient of the function.
    """
    assert l%1 == 0 and l >= 0, "l must be a natural number"
    return np.array((sph_func(l, r, derivative=True), 0, 0))


def sph_harm_diff_theta(m, l, theta, phi):
    """
    First derivative of spherical harmonics with respect to theta.

    Parameters
    ----------
    m,l: int
        Degree and order.
    theta, phi: float
        Spherical coordinates.

    Returns
    -------
    dY/dtheta: complex
        Derivative of spherical harmonics over theta.

    References
    --------
    .. [1]  Wolfram functions
            http://functions.wolfram.com/Polynomials/SphericalHarmonicY/20/01/01/
    """
    check_degree_and_order(m, l)
    return (m * _sph_harm(m, l, theta, phi) / np.tan(theta) +
            np.sqrt( (l - m) * (l + m + 1) ) * np.exp( -1j*phi ) *
            _sph_harm(m+1, l, theta, phi))


def sph_harm_diff_phi(m, l, theta, phi):
    """
    First derivative of spherical harmonics with respect to phi.

    Parameters
    ----------
    m,l: int
        Degree and order.
    theta, phi: float
        Spherical coordinates.

    Returns
    -------
    dY/dtheta: complex
        Derivative of spherical harmonics over phi.

    References
    --------
    .. [1]  Wolfram functions
            http://functions.wolfram.com/Polynomials/SphericalHarmonicY/20/01/02/
    """
    check_degree_and_order(m, l)
    return 1j * m * _sph_harm(m, l, theta, phi)


def sph_harm_gradient(m, l, r, theta, phi):
    """
    Gradient of spherical harmonics.

    Parameters
    ----------
    m,l: int
        degree and order
    r, theta, phi: real
        spherical coordinates

    Returns
    -------
    grad: ndarray
        (c_r, c_theta, c_phi)
    """
    c_theta = sph_harm_diff_theta(m, l, theta, phi) / r
    c_phi = sph_harm_diff_phi(m, l, theta, phi) / (r * np.sin(theta))
    return np.array((0, c_theta, c_phi))


def vsh3_pub(m, l, theta, phi):
    """
    Calculate a vector spherical harmonics 3 (VSH3) using the definiotion (39)
    from Ref. [1].

    Parameters
    ----------
    m,l: int
        Degree and order.
    theta, phi: float
        Spherical coordinates.

    Returns
    -------
    vsh3: ndarray
        Vector spherical harmonics 3.

    References
    ---------
    .. [1]  R. Sainidou, N. Stefanou, A. Modinos, "Green's function formalism
            for phononic crystals", Physical Review B (2004).
    """
    check_degree_and_order(m, l)
    if l == 0: return np.array((0,0,0))

    alpha = lambda m, l: np.sqrt((l - m) * (l + m + 1)) / 2

    c_theta = (alpha(-m, l) * np.cos(theta) * np.exp( 1j*phi ) *
               _sph_harm(m-1, l, theta, phi) - m * np.sin(theta) *
               _sph_harm(m, l, theta, phi) + alpha(m, l) * np.cos(theta) *
               np.exp( -1j*phi ) * _sph_harm(m+1, l, theta, phi)
               )
    c_phi = (alpha(-m, l) * np.exp( 1j*phi ) * _sph_harm(m-1, l, theta, phi) -
             alpha(m, l) * np.exp( -1j*phi ) * _sph_harm(m+1, l, theta, phi)
             ) * 1j

    return np.array((0, c_theta, c_phi)) / np.sqrt( l*(l+1) )


def vsh3(m, l, theta, phi):
    """
    Calculate third of vector spherical harmonics (VSH3).

    Parameters
    ----------
    m,l: int
        Degree and order.
    theta, phi: float
        Spherical coordinates.

    Returns
    -------
    vsh1: ndarray
        Vector spherical harmonics 3.
    """
    if l==0: return np.array([0, 0, 0])

    r = 1
    R = np.array([r, 0, 0])
    gradY = sph_harm_gradient(m, l, r, theta, phi)
    return -1j * np.cross(R, gradY) / np.sqrt( l * (l + 1))


def vsh3_test(dec=10, mneg=False):
    """
    Sprawdza rownosc vsh3 i vsh3_definiotion
    """
    l = np.random.randint(1, 10, 1)
    m = np.random.randint(-l*mneg, l, 1)
    theta, phi = np.random.random(2)
    a = np.round(vsh3_pub(m, l, theta, phi), dec)
    b = np.round(vsh3(m, l, theta, phi), dec)
    print('params:', m, l, theta, phi)
    print(a)
    print(b)
    return a == b

# =============================================================================
# for i in range(1000):
#     arr = vsh3_rand()
#     if not arr.all():
#         print('break', i)
#         break
# =============================================================================

def vsh1(m, l, theta, phi):
    """
    Calculate first of vector spherical harmonics (VSH1).

    Parameters
    ----------
    m,l: int
        Degree and order.
    theta, phi: float
        Spherical coordinates.

    Returns
    -------
    vsh1: ndarray
        Vector spherical harmonics 1.
    """
    check_degree_and_order(m, l)
    return np.array((_sph_harm(m, l, theta, phi), 0, 0))


def vsh2(m, l, theta, phi):
    """
    Calculate second of vector spherical harmonics (VSH2).

    Parameters
    ----------
    m,l: int
        Degree and order.
    theta, phi: float
        Spherical coordinates.

    Returns
    -------
    vsh2: ndarray
        Vector spherical harmonics 2.
    """
    c_theta = 1/np.sin(phi) * sph_harm_diff_theta(m, l, theta, phi)
    c_phi = sph_harm_diff_phi(m, l, theta, phi)
    return np.array((0, c_theta, c_phi))


def check_degree_and_order(m, l):
    """
    Check degree and order that function is receiving

    Parameters
    ----------
    l: int
        degree
    m: int
        order
    """
    assert l >= 0 and l%1 == 0, "l is not a natural number"
    assert m%1 == 0, "m is not an intiger"
    assert abs(m) <= l, "|m| is greater than l"


"""
Below are definiotions of functions that are elements from T and C matrices.

zn = omega * S / cn
xn = omega * S / csn
"""
def d11(l, zt):
    return zt * spherical_hn(l, zt, derivative=True) + spherical_hn(l, zt)

def d21(l, zt):
    return l * (l + 1) * spherical_hn(l, zt)

def d31(l, zt):
    return ((l * (l + 1) - zt**2 / 2 - 1) * spherical_hn(l, zt)
            - zt * spherical_hn(l, zt, derivative=True))

def d41(l, zt):
    return (l * (l + 1) * (zt * spherical_hn(l, zt, derivative=True)
            - spherical_hn(l, zt)))

d12 = lambda l, zl: spherical_hn(l, zl)

d22 = lambda l, zl: zl * spherical_hn(l, zl, derivative=True)

d32 = lambda l, zl: zl * spherical_hn(l, zl, derivative=True) \
                    - spherical_hn(l, zl)

d42 = lambda l, zl: (l * (l + 1) - zl**2 / 2) * spherical_hn(l, zl) \
                    - 2 * zl * spherical_hn(l, zl, derivative=True)

d13 = lambda l, xt: xt * spherical_jn(l, xt, derivative=True) \
                    + spherical_jn(l, xt)

d23 = lambda l, xt: l * (l + 1) * spherical_jn(l, xt)

d33 = lambda l, xt, zt, rhos, rho: \
                    rhos / rho * (zt / xt)**2 \
                    * ((l * (l + 1) - xt**2 / 2 - 1) * spherical_jn(l, xt) \
                       - xt * spherical_jn(l, xt, derivative=True))

d43 = lambda l, xt, zt, rhos, rho: \
                    rhos / rho * (zt / xt)**2 * l * (l + 1) \
                    * (xt * spherical_jn(l, xt, derivative=True) \
                       - spherical_jn(l, xt))

d14 = lambda l, xl: spherical_jn(l, xl)

d24 = lambda l, xl: xl * spherical_jn(l, xl, derivative=True)

d34 = lambda l, xl, xt, zt, rhos, rho: \
                    rhos / rho * (zt / xt)**2 \
                    * (xl * spherical_jn(l, xl, derivative=True) \
                       - spherical_jn(l, xl))

d44 = lambda l, xl, xt, zt, rhos, rho: \
                    rhos / rho * (zt / xt)**2 \
                    * ((l * (l + 1) - xt**2 / 2) * spherical_jn(l, xl) \
                       - 2 * xl * spherical_jn(l, xl, derivative=True))

dN1 = lambda l, zt: zt * spherical_jn(l, zt, derivative=True) \
                    + spherical_jn(l, zt)

dN2 = lambda l, zt: l * (l + 1) * spherical_jn(l, zt)

dN3 = lambda l, zt: (l * (l + 1) - zt**2 / 2 - 1) * spherical_jn(l, zt) \
                    - zt * spherical_jn(l, zt, derivative=True)

dN4 = lambda l, zt: l * (l + 1) * (zt * spherical_jn(l, zt, derivative=True) \
                                   - spherical_jn(l, zt))

dL1 = lambda l, zt: spherical_jn(l, zt)

dL2 = lambda l, zl: zl * spherical_jn(l, zl, derivative=True)

dL3 = lambda l, zl: zl * spherical_jn(l, zl, derivative=True) \
                    - spherical_jn(l, zl)

dL4 = lambda l, zt, zl: (l * (l + 1) - zt**2 / 2) * spherical_jn(l, zl) \
                    - 2 * zl * spherical_jn(l, zl, derivative=True)

w11 = lambda l, xt: - xt * spherical_hn(l, xt, derivative=True) \
                    - spherical_hn(l, xt)

w21 = lambda l, xt: -l * (l + 1) * spherical_hn(l, xt)

w31 = lambda l, xt, zt, rhos, rho: \
                    -rhos/rho * (zt / xt)**2 \
                    * ((l * (l + 1) - xt**2/2 - 1) * spherical_hn(l, xt) \
                       -xt * spherical_hn(l, xt, derivative=True))

w41 = lambda l, xt, zt, rhos, rho: \
                    -rhos/rho * (zt / xt)**2 * l * (l + 1) \
                    * (xt * spherical_hn(l, xt, derivative=True) \
                       - spherical_hn(l, xt))

w12 = lambda l, xl: - spherical_hn(l, xl)

w22 = lambda l, xl: - xl * spherical_hn(l, xl, derivative=True)

w32 = lambda l, xl, xt, zt, rhos, rho: \
                    -rhos/rho * (zt / xt)**2 \
                    * (xl * spherical_hn(l, xl, derivative=True) \
                       - spherical_hn(l, xl))

w42 = lambda l, xl, xt, zt, rhos, rho: \
                    -rhos/rho * (zt / xt)**2 \
                    * ((l * (l + 1) - xt**2 / 2) * spherical_hn(l, xl) \
                       - 2 * xl * spherical_hn(l, xl, derivative=True))

wN1 = lambda l, zt: zt * spherical_hn(l, zt, derivative=True) \
                    + spherical_hn(l, zt)

wN2 = lambda l, zt: l * (l + 1) * spherical_hn(l, zt)

wN3 = lambda l, zt: (l * (l + 1) - zt**2 / 2 - 1) * spherical_hn(l, zt) \
                    - zt * spherical_hn(l, zt, derivative=True)

wN4 = lambda l, zt: l * (l + 1) * (zt * spherical_hn(l, zt, derivative=True) \
                                  - spherical_hn(l, zt))

wL1 = lambda l, zl: spherical_hn(l, zl)

wL2 = lambda l, zl: zl * spherical_hn(l, zl, derivative=True)

wL3 = lambda l, zl: zl * spherical_hn(l, zl, derivative=True) \
                    - spherical_hn(l, zl)

wL4 = lambda l, zl, zt: (l * (l + 1) - zt**2 / 2) * spherical_hn(l, zl) \
                    - 2 * zl * spherical_hn(l, zl, derivative=True)

def matrix_M1(l, omega, S, cn, csn, rhos, rho):
    """
    Creates the M matrix.

    Parameters
    ----------
    l: int
        order
    omega: float
        wave anguelar frequency
    S: float
        radius of the sphere
    cn: dict
        a dictionary with velocity values outside the sphere
        ``{'l': value, 't': value}``
    csn: dict
        a dictionary with velocity values inside the sphere
        ``{'l': value, 't': value}``
    rhos: float
        sphere density
    rho: float
        host medium density

    Returns
    -------
    M: ndarray
        the M matrix
    """
    sqrt = np.sqrt(l * (l + 1))
    zl = omega * S / cn['l']
    zt = omega * S / cn['t']
    xl = omega * S / csn['l']
    xt = omega * S / csn['t']
    col1 = np.array((d11(l, zt), d21(l, zt), d31(l, zt), d41(l, zt))) / zt
    col2 = (-sqrt * np.array((d12(l, zl), d22(l, zl),
                              d32(l, zl), d42(l, zl))) / zl
            )
    col3 = (- np.array((d13(l, xt), d23(l, xt),
                       d33(l, xt, zt, rhos, rho),
                       d43(l, xt, zt, rhos, rho))) / xt
            )
    col4 = (sqrt * np.array((d14(l, xl), d24(l, xl),
                            d34(l, xl, xt, zt, rhos, rho),
                            d44(l, xl, xt, zt, rhos, rho))) / xl
            )
    M = np.array((col1, col2, col3, col4))
    return M.T


def matrix_N1(l, omega, S, cn):
    """
    Creates the N matrix.

    Parameters
    ----------
    l: int
        order
    omega: float
        wave anguelar frequency
    S: float
        radius of the sphere
    cn: dict
        a dictionary with velocity values outside the sphere
        ``{'l': value, 't': value}``

    Returns
    -------
    N: ndarray
        the N matrix
    """
    sqrt = np.sqrt(l * (l + 1))
    zl = omega * S / cn['l']
    zt = omega * S / cn['t']
    col1 = - np.array((dN1(l, zt), dN2(l, zt), dN3(l, zt), dN4(l, zt))) / zt
    col2 = (sqrt * np.array((dL1(l, zt), dL2(l, zl),
                            dL3(l, zl), dL4(l, zt, zl))) / zl
            )
    N = np.array((col1, col2))
    return N.T


def matrix_K1(l, omega, S, cn, csn, rhos, rho):
    """
    Creates the K matrix.

    Parameters
    ----------
    l: int
        order
    omega: float
        wave anguelar frequency
    S: float
        radius of the sphere
    cn: dict
        a dictionary with velocity values outside the sphere
        ``{'l': value, 't': value}``
    csn: dict
        a dictionary with velocity values inside the sphere
        ``{'l': value, 't': value}``
    rhos: float
        sphere density
    rho: float
        host medium density

    Returns
    -------
    K: ndarray
        the K matrix
    """
    zt = omega * S / cn['t']
    xt = omega * S / csn['t']
    row1 = np.array((- d21(l, zt), d23(l, xt)))
    row2 = np.array((- d41(l, zt), d43(l, xt, zt, rhos, rho)))
    return np.array((row1, row2))


def matrix_L1(l, omega, S, cn):
    """
    Creates the L matrix.

    Parameters
    ----------
    l: int
        order
    omega: float
        wave anguelar frequency
    S: float
        radius of the sphere
    cn: dict
        a dictionary with velocity values outside the sphere
        ``{'l': value, 't': value}``

    Returns
    -------
    L: ndarray
        the L matrix
    """
    zt = omega * S / cn['t']
    L = np.array((dN2(l, zt), dN4(l, zt)))
    return L.T


def matrix_M2(l, omega, S, cn, csn, rhos, rho):
    """
    Creates the M' matrix.

    Parameters
    ----------
    l: int
        order
    omega: float
        wave anguelar frequency
    S: float
        radius of the sphere
    cn: dict
        a dictionary with velocity values outside the sphere
        ``{'l': value, 't': value}``
    csn: dict
        a dictionary with velocity values inside the sphere
        ``{'l': value, 't': value}``
    rhos: float
        sphere density
    rho: float
        host medium density

    Returns
    -------
    M': ndarray
        the M' matrix
    """
    sqrt = np.sqrt(l * (l + 1))
    zt = omega * S / cn['t']
    xl = omega * S / csn['l']
    xt = omega * S / csn['t']
    col1 = (np.array((w11(l, xt), w21(l, xt), w31(l, xt, zt, rhos, rho),
                     w41(l, xt, zt, rhos, rho))) / xt
            )
    col2 = (-sqrt * np.array((w12(l, xl), w22(l, xl),
                             w32(l, xl, xt, zt, rhos, rho),
                             w42(l, xl, zt, zt, rhos, rho))) / xl
            )
    col3 = (- np.array((d13(l, xt), d23(l, xt),
                       d33(l, xt, zt, rhos, rho),
                       d43(l, xt, zt, rhos, rho))) / xt
            )
    col4 = (sqrt * np.array((d14(l, xl), d24(l, xl),
                            d34(l, xl, xt, zt, rhos, rho),
                            d44(l, xl, xt, zt, rhos, rho))) / xl
            )
    M = np.array((col1, col2, col3, col4))
    return M.T


def matrix_N2(l, omega, S, cn):
    """
    Creates the N' matrix.

    Parameters
    ----------
    l: int
        order
    omega: float
        wave anguelar frequency
    S: float
        radius of the sphere
    cn: dict
        a dictionary with velocity values outside the sphere
        ``{'l': value, 't': value}``

    Returns
    -------
    N': ndarray
        the N' matrix
    """
    sqrt = np.sqrt(l * (l + 1))
    zl = omega * S / cn['l']
    zt = omega * S / cn['t']
    col1 = - np.array((wN1(l, zt), wN2(l, zt), wN3(l, zt), wN4(l, zt))) / zt
    col2 = (sqrt * np.array((wL1(l, zt), wL2(l, zl),
                            wL3(l, zl), wL4(l, zl, zt))) / zl
            )
    N = np.array((col1, col2))
    return N.T


def matrix_K2(l, omega, S, cn, csn, rhos, rho):
    """
    Creates the K' matrix.

    Parameters
    ----------
    l: int
        order
    omega: float
        wave anguelar frequency
    S: float
        radius of the sphere
    cn: dict
        a dictionary with velocity values outside the sphere
        ``{'l': value, 't': value}``
    csn: dict
        a dictionary with velocity values inside the sphere
        ``{'l': value, 't': value}``
    rhos: float
        sphere density
    rho: float
        host medium density

    Returns
    -------
    K': ndarray
        the K' matrix
    """
    zt = omega * S / cn['t']
    xt = omega * S / csn['t']
    row1 = np.array((- w21(l, xt), d23(l, xt)))
    row2 = np.array((- w41(l, xt, zt, rhos, rho), d43(l, xt, zt, rhos, rho)))
    return np.array((row1, row2))


def matrix_L2(l, omega, S, cn):
    """
    Creates the L' matrix.

    Parameters
    ----------
    l: int
        order
    omega: float
        wave anguelar frequency
    S: float
        radius of the sphere
    cn: dict
        a dictionary with velocity values outside the sphere
        ``{'l': value, 't': value}``

    Returns
    -------
    L': ndarray
        the L' matrix
    """
    zt = omega * S / cn['t']
    L = np.array(wN2(l, zt), wN4(l, zt))
    return L.T


def matrices_TC(l, omega, S, cn, csn, rhos, rho):
    """
    Creates the T and C matrices.

    Parameters
    ----------
    l: int
        order
    omega: float
        wave anguelar frequency
    S: float
        radius of the sphere
    cn: dict
        a dictionary with velocity values outside the sphere
        ``{'l': value, 't': value}``
    csn: dict
        a dictionary with velocity values inside the sphere
        ``{'l': value, 't': value}``
    rhos: float
        sphere density
    rho: float
        host medium density

    Returns
    -------
    T, C: (ndarray, ndarray)
        tuple of ndarrays
    """
    MN = (np.linalg.inv(matrix_M1(l, omega, S, cn, csn, rhos, rho))
          * matrix_N1(l, omega, S, cn)
          )
    KL = (np.linalg.inv(matrix_K1(l, omega, S, cn, csn, rhos, rho))
          * matrix_L1(l, omega, S, cn)
          )
    T = np.zeros((3,3))
    C = np.zeros((3,3))
    T[:2,:2] = MN[:2]
    T[ 3, 3] = KL[0]
    C[:2,:2] = MN[2:]
    C[ 3, 3] = KL[1]
    return T, C


def matrices_QP(l, omega, S, cn, csn, rhos, rho):
    """
    Creates the Q and P matrices.

    Parameters
    ----------
    l: int
        order
    omega: float
        wave anguelar frequency
    S: float
        radius of the sphere
    cn: dict
        a dictionary with velocity values outside the sphere
        ``{'l': value, 't': value}``
    csn: dict
        a dictionary with velocity values inside the sphere
        ``{'l': value, 't': value}``
    rhos: float
        sphere density
    rho: float
        host medium density

    Returns
    -------
    Q, P: (ndarray, ndarray)
        tuple of ndarrays
    """
    MN = (np.linalg.inv(matrix_M2(l, omega, S, cn, csn, rhos, rho))
          * matrix_N2(l, omega, S, cn)
          )
    KL = (np.linalg.inv(matrix_K2(l, omega, S, cn, csn, rhos, rho))
          * matrix_L2(l, omega, S, cn)
          )
    Q = np.zeros((3,3))
    P = np.zeros((3,3))
    Q[:2,:2] = MN[:2]
    Q[ 3, 3] = KL[0]
    P[:2,:2] = MN[2:]
    P[ 3, 3] = KL[1]
    return Q, P


class Vector3D():
    """
    Klasa tworząca wektor o współrzędnych rzeczywistych

    Argumenty
        :c1, c2, c3:    trzy współrzędne
        :coortype:      określa typ współrzędnych
                        "cartesian" - kartezjańskie
                        "spherical" - sferyczne

    Współrzędne kartezjańskie
        (c1, c2, c3) = (x, y, z)

    Współrzędne sferyczne
        (c1, c2, c3) = (r, theta, phi)
        gdzie
            :r:       -   radialna
            :theta:   -   polarna [0, pi]
            :phi:     -   azymutalna [0, 2*pi]
    """

    def __init__(self, c1=1, c2=0, c3=0, coortype='cartesian'):
        if coortype == 'cartesian':
            self.cart = np.array([c1, c2, c3])
        elif coortype == 'spherical':
            self.sph = np.array([c1, c2, c3])
        else:
            raise "Wrong coordinate type"

    @property
    def cart(self):
        return self.__cart

    @cart.setter
    def cart(self, cart_coor):
        self.__cart = cart_coor
        assert self.r > 0, "Wrong coordinates, at least one coordinate must be greater than 0"

    @property
    def x(self):
        return self.__cart[0]

    @property
    def y(self):
        return self.__cart[1]

    @property
    def z(self):
        return self.__cart[2]

    @property
    def sph(self):
        return self.r, self.theta, self.phi

    @sph.setter
    def sph(self, sph_coor):
        """Set spherical coordinates"""
        # set proper boundaries for angles
        sph_coor_new = list(sph_coor)
        for angle, i in zip(sph_coor[1:], (1, 2)):
            lim = i*np.pi
            if angle < 0:
                angle += -lim * (angle // lim)
            elif angle > lim:
                angle -= lim * (angle // lim)
            sph_coor_new[i] = angle

        r, theta, phi = sph_coor_new

        assert r > 0, "r <= 0"
        assert phi >= 0 and phi <= 2 * np.pi, "Zła wartość kąta phi"
        assert theta >= 0 and theta <= np.pi, "Zła wartość kąta theta"

        x = r * np.cos(phi) * np.sin(theta)
        y = r * np.sin(phi)* np.sin(theta)
        z = r * np.cos(theta)
        self.__cart = np.array((x, y, z))

    @property
    def r(self):
        return np.sqrt(self.x**2 + self.y**2 + self.z**2)

    @property
    def phi(self):
        if self.x + self.y == 0:  # bo osobliwość
            return 0
        try:
            phi = np.arctan(self.y/self.x)
        except (RuntimeWarning, ZeroDivisionError):
            sign = np.sign(self.y)
            if sign:
                phi = np.pi + sign * np.pi / 2
            else:
                phi = 0
        return phi

    @property
    def theta(self):
        try:
            theta = np.arccos(self.z/self.r)
        except (RuntimeWarning, ZeroDivisionError):
            theta = 0
        return theta


#class TransformationMatrix():
#    """
#    Matrix used to transform plane wave into spherical.
#    Dimensions: 3l x 3l
#    """
#    def __init__(self, l, omega, S, cn, csn, rhos, rho):
#        pass


class Wave():
    """
    Main class for a wave function.

    Parameters
    ----------
    P: str
        wave type: \n
        - "L" fala podłużna
        - "M" fala poprzeczna o polaryzacji p
        - "N" fala poprzeczna o polaryzacji s
    m: int
        degree
    l: int
        order
    omega: float
        frequency
    c: float
        velocity
    """

    def __init__(self, P='L', m=0, l=0, omega=1, c=speed_of_light):
        """Initialize a wave function"""
        check_degree_and_order(m, l)
        assert P in ('L', 'M', 'N'), "P should take one of the values ('L', 'M', 'N')"
        assert omega > 0, "omega should be greater than 0"
        assert c > 0, "c should be greater than 0"

        self.P = P
        self.l = l
        self.m = m
        self.omega = omega
        self.c = c
        self.wave_type = {'L': self.__podluzna,
                          'M': self.__poprzeczna_p,
                          'N': self.__poprzeczna_s}

#    def __call__(self, R=Vector3D):
#        """Compute function in a position given by R vector."""
#        assert type(R) == Vector3D, "R should by of type Vector3D"
#        return None  # będzie zwracać wektor

#%%

class SphericalWave(Wave):
    """
    Main class for a  spherical wave function.

    Parameters
    ----------
    P: str
        wave type: \n
        - "L" fala podłużna
        - "M" fala poprzeczna o polaryzacji p
        - "N" fala poprzeczna o polaryzacji s
    m: int
        degree
    l: int
        order
    omega: float
        frequency
    c: float
        velocity
    """
    def __podluzna(self, sph_func, R):
        """
        Compute longitudal spherical eigenfunction
        """
        q = self.omega / self.c
        return (spherical_gradient(self.l, q*R.r, sph_func)
                * _sph_harm(self.m, self.l, R.theta, R.phi)
                + sph_func(self.l, q*R.r)
                * sph_harm_gradient(self.m, self.l, q*R.r, R.theta, R.phi)
                ) / q

    def __poprzeczna_p(self, sph_func, R):
        """
        Compute transverse spherical eigenfunction with p-polarization
        """
        q = self.omega / self.c
        return sph_func(self.l, q*R.r) * vsh3(self.m, self.l, R.theta, R.phi)

    def __poprzeczna_s(self, sph_func, R):
        """
        Compute transverse spherical eigenfunction with s-polarization
        """
        q = self.omega / self.c
        r = q * R.r
        a = self.l * ( self.l + 1 )
        return (-a * sph_func(self.l, r) *
                vsh1(self.m, self.l, R.theta, R.phi) / r -
                (sph_func(self.l, r, derivative=True) +
                 sph_func(self.l, r) / r
                 ) * vsh2(self.m, self.l, R.theta, R.phi)
                ) / np.sqrt(a) / q

#%%

class IncomingWave(Wave):
    pass

class OutgoingWave(Wave):
    pass

class RegularWave(Wave):
    pass

class IrregularWave(Wave):
    pass


class GreenFuncSphere():
    """
    Liczy funkcję Greena dla jednorodnej sfery zanurzonej
    w jednorodnym ośrodku.

    Argumenty
        S    :   średnica sfery
        rho  :   gęstość ośrodka
        rhoS :   gęstość sfery
        r1   :   ?
        r2   :   ?

    Powiązanie funkcji z oznaczeniami użytymi w opisie
        R    :   regularwave
        I    :   irregularwave
    """
    def __init__(self, S, rho, rhoS, r1, r2):
        pass



if __name__ == '__main__':
    pass