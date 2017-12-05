#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  9 16:56:01 2017

@author: monika
"""

import mathtools
import unittest
import numpy as np


class KnownValues(unittest.TestCase):

    spherical_hn_values = (
            ((0,  0),  np.nan + 1j*np.inf),
            ((0,  1),  0.8414709848078965 - 1j*0.5403023058681397),
#            ((0, -1),  0.8414709848078965 + 1j*0.5403023058681397),
            ((1,  0),  np.nan + 1j*np.inf),
            ((1,  1),  0.3011686789397567 - 1j*1.3817732906760362),
#            ((1, -5),  0.0950894080791707 + 1j*0.1804383675140986),
            ((2,  0),  np.nan + 1j*np.inf),
            ((2,  1),  0.0620350520113738 - 1j*3.6050175661599689),
            ((2, 64), -0.0146518827918317 + 1j*0.0054444375794004),
            ((3,  0),  np.nan + 1j*np.inf),
            ((3, 19),  0.0474010963848815 + 1j*0.0238794700013334),
#            ((3, -7),  0.0016120468591568 + 1j*0.1527286711607054),
            ((4,  0),  np.nan + 1j*np.inf),
            ((4, 11), -0.0574173060215915 - 1j*0.0757278401103418),
#            ((4,-15),  0.0025836762513213 - 1j*0.0681746685875722),
            ((5,  0),  np.nan + 1j*np.inf),
#            ((5,-34), -0.0294079535548832 - 1j*0.0034141438670383),
            ((5, 89), -0.0040334675621289 - 1j*0.0104984538988348)
            )

    def test_spherical_hn(self):
        """spherical_hn powinno dać sprawdzone rezultaty"""
        for args, value in self.spherical_hn_values:
            result = mathtools.spherical_hn(*args)
            for r, v in ((result.real, value.real), (result.imag, value.imag)):
                if np.isnan(r):  # or np.isinf(r):
                    self.assertTrue(np.isnan(v))
                elif np.isinf(r):
                    self.assertTrue(np.isinf(v))
                else:
                    self.assertAlmostEqual(v, r)


class IsItWorking(unittest.TestCase):

    l_values = list(range(10))
    m_values = list(range(-10, 10))
    r_values = list(range(-100, 100, 5))

    def test_spherical_hn(l, r, derivative=False):
        pass

if __name__ == '__main__':
    unittest.main()
