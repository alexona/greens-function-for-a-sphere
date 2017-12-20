#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 20 10:21:32 2017

@author: monika
"""

import greensphere
import unittest
import numpy as np


def check_vsh3(l, m, dec=6, mneg=False, it=25):
    """Sprawdza rownosc vsh3 i vsh3_definiotion"""
#        l = np.random.randint(1, 10, 1)
#        m = np.random.randint(-l*mneg, l, 1)
    data = []
    check = []
    for i in range(it):
        theta, phi = np.round(np.random.random(2), 3)
        expr = (np.nan_to_num(np.round(
                greensphere.vsh3(m, l, theta, phi), dec)))
        defi = (np.nan_to_num(np.round(
                greensphere.vsh3_definition(m, l, theta, phi), dec)))
        if not (expr == defi).all():
            check.append(True)
            data.append('{},{}, {:.3f}, {:.3f}\n{}\n{}\n'.format(
                    l, m, theta, phi, expr, defi))
        else:
            check.append(False)
    for res in data:
        print(res)
    return check


class CheckFunctionVSH3(unittest.TestCase):

    def test_vsh3_mpos(self):
        res = 0
        for l in range(10):
            for m in range(l):
                res = check_vsh3(l, m)
                print('it:', len(res), '\twrong:', sum(res))
        self.assertTrue(sum(res) == 0)

    def test_vsh3_mneg(self):
        for l in range(10):
            for m in range(-l, 0):
                res = check_vsh3(l, m, mneg=True)
                print('it:', len(res), '\twrong:', sum(res))
        self.assertTrue(sum(res) == 0)


# =============================================================================
#     def test_vsh3_mneg(self):
#         res = 0
#         for l in range(10):
#             for m in range(l):
#                 res += check_vsh3(l, m)
#         self.assertTrue(res == 0)
# =============================================================================
