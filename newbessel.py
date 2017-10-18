# -*- coding: utf-8 -*-
"""
Z modułu scipy.special importujemy:
    sph_harm            -   harmoniki sferyczne
    spherical_jn        -   sferyczna funkcja Bessela pierwszego rodzaju
    spherical_yn        -   sferyczna funkcja Bessela drugiego rodzaju

"""

import numpy as np


class SphericalBessel():
    """
    Redefine spherical Bessel function for z < 0.
    Parameters:
        kind:
            ('first', 'second')     -   kind of function
    """

    def __init__(self, kind='first'):
        assert kind in ('first', 'second'), "Wrong function kind"
        self.kind = kind
        self.__do = {'first': self.__first, 'second': self.__second}

    def __call__(self, l, z):
        assert l%1 == 0 and l >= 0, "l must be a natural number"
        return self.__do[self.kind](l, z)

    @staticmethod
    def __first(l, z):
        from scipy.special import spherical_jn as sph_jn
        j = sph_jn(l, np.abs(z))
        try:
            j[np.sign(z) < 0] *= (-1) ** l
            return j
        except TypeError:  # pojawia się, gdy z jest liczbą
            sign = -1 if z < 0 else 1
            return j * sign ** l

    @staticmethod
    def __second(l, z):
        from scipy.special import spherical_yn as sph_yn
        y = sph_yn(l, np.abs(z))
        try:
            y[np.sign(z) < 0] *= (-1) ** (l + 1)
            return y
        except TypeError:  # pojawia się, gdy z jest liczbą
            sign = -1 if z < 0 else 1
            return y * sign ** (l + 1)


spherical_jn = SphericalBessel('first')
spherical_yn = SphericalBessel('second')


if __name__ == '__main__':
    pass