# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 21:03:23 2019

@author: Julie
"""
# mymod.py
# bins = amplitude histogram bins
# num = amplitude histogram values
# p0 = [max(num), bins(num==max(num)), 2 * nanstd(coords),prctile(curr_amplis, 1)]
import numpy as np
import scipy.optimize as opt

def JF_gaussian_cut(x, a, x0, sigma, xcut):
    g = a * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2))
    g[x < xcut] = 0
    return g

def JF_fit(x,num,p1):
    popt = opt.curve_fit(JF_gaussian_cut, x, num, p0=p1,maxfev=10000) 
    return popt
