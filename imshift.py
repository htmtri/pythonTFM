# -*- coding: utf-8 -*-
"""
Created on Tue Aug 27 14:26:14 2019

@author: surface
"""

import numpy as np
#import matplotlib.pyplot as plt
#from scipy import signal
from normxcorr2 import normxcorr2

def imshift(im1,im2,rect):
    
    subimg = im1[int(rect[1]):int(rect[1]+rect[3]), int(rect[0]):int(rect[0]+rect[2])]
    
    cc1 = normxcorr2(subimg,im1,mode='full')
    cc2 = normxcorr2(subimg,im2,mode='full')
    
    idm1 = np.where(cc1 == np.amax(np.abs(cc1)))
    idm2 = np.where(cc2 == np.amax(np.abs(cc2)))
    
    #[xpeak1,ypeak1] = np.unravel_index(idm1,np.shape(cc1),order='F')
    #[xpeak2,ypeak2] = np.unravel_index(idm2,np.shape(cc2),order='F')
    
    xdrift = idm2[1]-idm1[1]
    ydrift = idm2[0]-idm1[0]
    
    return np.array([xdrift[0],ydrift[0]])