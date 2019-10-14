# -*- coding: utf-8 -*-
"""
Created on Fri Aug 23 16:56:48 2019

@author: surface
"""
import numpy as np
from scipy import signal


def im2double(im):
    info = np.iinfo(im.dtype) # Get the data type of the input image
    return im.astype(np.float) / info.max # Divide all values by the largest possible value in the datatype

def normalize(x):
    return x/np.sum(x)

def bpass(im_array,lnoise=0,lobject=0,threshold=0):
    
    if lnoise == 0 :
        gaussian_kernel = np.array([[1],[0]])
    else:
        gk = normalize(np.exp(-((np.arange(-np.ceil(5*lnoise),np.ceil(5*lnoise)+1))/(2*lnoise))**2))
        gaussian_kernel = np.vstack((gk,np.zeros(np.size(gk))))
    
    if lobject:
        bk = normalize(np.ones((1,np.size(np.arange(-np.ma.round(lobject),np.ma.round(lobject)+1)))))
        boxcar_kernel = np.vstack((bk,np.zeros(np.size(bk))))
        
    gconv = signal.convolve2d(np.transpose(im_array),np.transpose(gaussian_kernel),mode='same')
    gconv = signal.convolve2d(np.transpose(gconv),np.transpose(gaussian_kernel),mode='same')
    
    if lobject:
        bconv = signal.convolve2d(np.transpose(im_array),np.transpose(boxcar_kernel),mode='same')
        bconv = signal.convolve2d(np.transpose(bconv),np.transpose(boxcar_kernel),mode='same')
        filtered = gconv - bconv
    else:
        filtered = gconv
    
    lzero = np.amax((lobject,np.ceil(5*lnoise)))
    
    filtered[0:int(np.round(lzero)),:] = 0
    filtered[(-1 - int(np.round(lzero)) + 1):,:] = 0
    filtered[:,0:int(np.round(lzero))] = 0
    filtered[:,(-1 - int(np.round(lzero)) + 1):] = 0
    filtered[filtered < threshold] = 0
    return filtered