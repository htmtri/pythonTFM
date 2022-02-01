# -*- coding: utf-8 -*-
"""
Created on Fri Nov 20 20:35:11 2020

@author: htmtr
"""

import numpy as np
import matplotlib as mplt
import matplotlib.pyplot as plt
import matplotlib.path as mpltpath
from tkinter import filedialog
from tkinter import *
from roipoly import RoiPoly
from scipy.stats import mode
from scipy import fft
import cv2
from bpass import bpass
from imshift import imshift
import openpiv.tools as tools
import openpiv.pyprocess
import openpiv.scaling
import openpiv.validation
import openpiv.filters

root = Tk()

# root.directory = filedialog.askdirectory()

root.pc_name = filedialog.askopenfilename(initialdir='.',
                                          title="Select phase contrast img",
                                          filetypes=(("tiff files", "*.tiff"),
                                                     ("all files", "*.*")))

root.load_name = filedialog.askopenfilename(initialdir='.',
                                            title="Select loadimg",
                                            filetypes=(("tiff files", "*.tiff"),
                                                       ("all files", "*.*")))
root.nulf_name = filedialog.askopenfilename(initialdir='.',
                                            title="Select nulfimg",
                                            filetypes=(("tiff files", "*.tiff"), ("all files", "*.*")))

frame_c = plt.imread(root.pc_name)
org_b = plt.imread(root.nulf_name)
org_a = plt.imread(root.load_name)

root.withdraw()

mplt.use("TkAgg")

fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(
    32, 20), sharex='all', sharey='all')

ax[0].imshow(frame_c, CMAP='gray', origin='lower')
ax[0].axis('off')
ax[0].set_title('OG Cell Image', fontsize=20)

ax[1].imshow(org_a, CMAP='gray', origin='lower',
             vmin=0, vmax=0.03 * org_a.max())
ax[1].axis('off')
ax[1].set_title('OG Loaded beads image', fontsize=20)

ax[2].imshow(org_b, CMAP='gray', origin='lower',
             vmin=0, vmax=0.05 * org_b.max())
ax[2].axis('off')
ax[2].set_title('OG Relaxed beads image', fontsize=20)

fig.tight_layout()
plt.show(block=False)

c_uint8 = cv2.convertScaleAbs(frame_c, alpha=0.5)

showCrosshair = False
fromCenter = False

# drag from top left to bot right .press ESC to finish
rect = cv2.selectROI('Cell ROI', c_uint8, fromCenter, showCrosshair)
cv2.destroyWindow('Cell ROI')
recs = cv2.selectROI('Empty ROI', c_uint8, fromCenter, showCrosshair)
cv2.destroyWindow('Empty ROI')

frame_b = bpass(org_b, lnoise=0, lobject=7,
                threshold=0.05 * mode(org_b.flatten())[0])
frame_a = bpass(org_a, lnoise=0, lobject=7,
                threshold=0.05 * mode(org_a.flatten())[0])
drift = imshift(frame_a, frame_b, recs)

cellimg = frame_c[int(rect[1]):int(rect[1] + rect[3]),
                  int(rect[0]):int(rect[0] + rect[2])]
loadimg = frame_a[int(rect[1]):int(rect[1] + rect[3]),
                  int(rect[0]):int(rect[0] + rect[2])]
nulfimg = frame_b[int(rect[1] + drift[1]):int(rect[1] + rect[3] + drift[1]),
                  int(rect[0] + drift[0]):int(rect[0] + rect[2] + drift[0])]

fig = plt.figure()
plt.imshow(cellimg, cmap='gray', vmin=0, vmax=0.5 * cellimg.max())
plt.title("left click: line segment; right click or double click: close region.\n"
          "Please close all open GUI windows to proceed with PIV")
plt.show(block=False)

cell_roi = RoiPoly(color='r', fig=fig, show_fig=False)
# block=False leads to subsequent code gets run before cell_roi is properly setup
# block=True hangs Spyder (most likely because Spyder is a Qt GUI itself)

cimg = np.zeros([cellimg.shape[0], cellimg.shape[1], 3], dtype=np.uint8)
cimg[:, :, 0] = loadimg
cimg[:, :, 1] = nulfimg
plt.figure()
plt.imshow(cimg, vmin=0, vmax=cimg[:, :, 0].max())
plt.title("Overlay beads image")
plt.axis('off')
plt.show(block=True)


# plt.figure()
# plt.plot(cell_roi.x,cell_roi.y,color='r',linewidth=2.5)
# plt.plot(cell_roi.x + [cell_roi.x[0]],cell_roi.y + [cell_roi.y[0]],color='b')
# plt.show(block=False)
# cell_roi does not close boundary itself
cellx = np.array(cell_roi.x + [cell_roi.x[0]])
celly = np.array(cell_roi.y + [cell_roi.y[0]])

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(
    32, 20), sharex='all', sharey='all')
ax[0].imshow(cellimg, cmap='gray', vmin=0, vmax=0.5 * cellimg.max())
cell_roi.display_roi()
ax[0].axis('off')
ax[0].set_title('Cell Image with celltrace', fontsize=20)

ax[1].imshow(cimg, cmap='gray', vmin=0, vmax=0.5 * cellimg.max())
ax[1].plot(cellx, celly, color='r')
ax[1].axis('off')
ax[1].set_title('Beads with celltrace', fontsize=20)
plt.show(block=False)

u, v, sig2noise = openpiv.process.extended_search_area_piv(nulfimg.astype(np.int32),
                                                           loadimg.astype(
                                                               np.int32),
                                                           window_size=32,
                                                           overlap=16,
                                                           dt=0.02,
                                                           search_area_size=48,
                                                           subpixel_method='gaussian',
                                                           sig2noise_method='peak2peak')

xm, ym = openpiv.process.get_coordinates(
    image_size=loadimg.shape, window_size=32, overlap=16)
xm = np.flipud(xm)
ym = np.flipud(ym)

xgrid = xm.flatten()
ygrid = ym.flatten()
grid = list(zip(xgrid, ygrid))

u1, v1, mask = openpiv.validation.sig2noise_val(
    u, v, sig2noise, threshold=2)  # noise-filter
u2, v2, mask = openpiv.validation.local_median_val(u1, v1, 2, 2)  # smooth
uf, vf = openpiv.filters.replace_outliers(
    u2, v2, method='localmean', max_iter=10, tol=1e-3, kernel_size=2)
# x, y, u, v = openpiv.scaling.uniform(x, y, u, v, scaling_factor = 96.52 )
xdispRaw = uf.flatten()
ydispRaw = vf.flatten()

plt.figure()
plt.imshow(cimg, vmin=0, vmax=0.01 * cimg.max())
plt.quiver(xgrid, ygrid, xdispRaw, ydispRaw, color='c', units='xy')
plt.plot(cellx, celly, color='r')
plt.show()

##PARAM BLOCK
px_size = 0.161e-6
poisson = 0.49# Poisson's Ratio
h =  100e-6
E = 1500 #G = E/2(1+Poisson)
mu = E/(2*(1+poisson))

##################NAIVE_ALGO###############################
# Solving in the case of inf thickness of inf small deformation (h*k<<0)
u_ft = fft.fft2(uf*px_size) 
v_ft = fft.fft2(uf*px_size)
tx_ft = mu*u_ft/h
ty_ft = mu*v_ft/h
tx = fft.ifft2(tx_ft)
ty = fft.ifft2(ty_ft)
t = np.sqrt(tx**2+ty**2)

