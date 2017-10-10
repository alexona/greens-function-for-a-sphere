#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  9 16:56:01 2017

@author: monika
"""

import greenfuncsphere
import unittest
import numpy as np


class KnownValues(unittest.TestCase):

    spherical_hn_values = (((0,  0),  np.nan + 1j*np.nan),
                           ((0,  1),  0.8414709848078965 - 1j*0.5403023058681397),
                           ((0, -1),  0.8414709848078965 + 1j*0.5403023058681397),
                           ((1,  0),  np.nan + 1j*np.nan),
                           ((1,  1),  0.3011686789397567 - 1j*1.3817732906760362),
                           ((1, -5),  0.0950894080791707 + 1j*0.1804383675140986),
                           ((2,  0),  np.nan + 1j*np.nan),
                           ((2,  1),  0.0620350520113738 - 1j*3.6050175661599689),
                           ((2, 64), -0.0146518827918317 + 1j*0.0054444375794004),
                           ((3,  0),  np.nan + 1j*np.nan),
                           ((3, 19),  0.0474010963848815 + 1j*0.0238794700013334),
                           ((3, -7),  0.0016120468591568 + 1j*0.1527286711607054),
                           ((4,  0),  np.nan + 1j*np.nan),
                           ((4, 11), -0.0574173060215915 - 1j*0.0757278401103418),
                           ((4,-15),  0.0025836762513213 - 1j*0.0681746685875722),
                           ((5,  0),  np.nan + 1j*np.nan),
                           ((5,-34), -0.0294079535548832 - 1j*0.0034141438670383),
                           ((5, 89), -0.0040334675621289 - 1j*0.0104984538988348))

    sph_harm_diff_theta_values = ()


    def test_spherical_hn(self):
        """spherical_hn powinno dać sprawdzone rezultaty"""
        for args, value in self.spherical_hn_values:
            result = greenfuncsphere.spherical_hn(*args)
            for r, v in ((result.real, value.real), (result.imag, value.imag)):
                if np.isnan(r):  # or np.isinf(r):
                    self.assertTrue(np.isnan(v))
                else:
                    self.assertAlmostEqual(v, r)


#    def test_sph_harm_diff_theta(self):
#        """sph_harm_diff_theta powinno dać sprawdzone rezultaty"""
#        for integer, numeral in self.known_values:
#            result = roman1.from_roman(numeral)
#            self.assertEqual(integer, result)

#
#class ToRomanBadInput(unittest.TestCase):
#
#    def test_too_large(self):
#        '''to_roman should fail with large input'''
#        self.assertRaises(roman1.OutOfRangeError, roman1.to_roman, 4000)
#
#    def test_zero(self):
#        '''to_roman should fail with 0 input'''
#        self.assertRaises(roman1.OutOfRangeError, roman1.to_roman, 0)
#
#    def test_negative(self):
#        '''to_roman should fail with negative input'''
#        self.assertRaises(roman1.OutOfRangeError, roman1.to_roman, -1)
#
#    def test_non_integer(self):
#        '''to_roman should fail with non-integer input'''
#        self.assertRaises(roman1.NotIntegerError, roman1.to_roman, 0.5)
#
#
#class RoundtripCheck(unittest.TestCase):
#
#    def test_roundtrip(self):
#        '''from_roman(to_roman(n))==n for all n'''
#        for integer in range(1, 4000):
#            numeral = roman1.to_roman(integer)
#            result = roman1.from_roman(numeral)
#            self.assertEqual(result, integer)
#
#
#class FromRomanBadInput(unittest.TestCase):
#
#    def test_too_many_repeated_numerals(self):
#        '''from_roman should fail with too many repeated numerals'''
#        for s in ('MMMM', 'DD', 'CCCC', 'LL', 'XXXX', 'VV', 'IIII'):
#            self.assertRaises(roman1.InvalidRomanNumeralError,
#                              roman1.from_roman, s)
#
#    def test_repeated_pairs(self):
#        '''from_roman should fail with repeated pairs of numerals'''
#        for s in ('CMCM', 'CDCD', 'XCXC', 'XLXL', 'IXIX', 'IVIV'):
#            self.assertRaises(roman1.InvalidRomanNumeralError,
#                              roman1.from_roman, s)
#
#    def test_malformed_antecedents(self):
#        '''from_roman should fail with malformed antecedents'''
#        for s in ('IIMXCC', 'VX', 'DCM', 'CMM', 'IXIV',
#                  'MCMC', 'XCX', 'IVI', 'LM', 'LD', 'LC'):
#            self.assertRaises(roman1.InvalidRomanNumeralError,
#                              roman1.from_roman, s)

if __name__ == '__main__':
    unittest.main()
