# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 16:03:19 2019

@author: htmtri
"""

import numpy as np

#import os

#from tempfile import TemporaryFile

#import inspect
#src = inspect.getsource(tools)

import matplotlib.path as mpltPath
import matplotlib.pyplot as plt

### ORDER OF POLYGON BOUNDARY VERTICES WILL AFFECT THE RESULT IF RADIUS 
### OF BOUNDARY IS 0 (or closed=True). CONSIDER SETTING RADIUS TO 1E9

#polygon = ((-80, 0), (80, 0), (80, 60), (-80, 60))

cellx = np.array((-80,80,80,-80))
celly = np.array((0,0,60,60))

x, y = np.meshgrid(np.arange(-90, 100, 10), np.arange(-5, 70, 5))
points = list(zip(x.flatten(), y.flatten()))

polygon = np.column_stack((cellx,celly))
#polygon = tuple(map(tuple,np.column_stack((cellx,celly)))) 

path = mpltPath.Path(polygon, closed=True)
path = mpltPath.Path(polygon)
inside = path.contains_points(points,radius=1e-9)

path2 = mpltPath.Path(polygon[::-1], closed=True)
path2 = mpltPath.Path(polygon[::-1])
inside2 = path2.contains_points(points,radius=-1e-9)

fig, (ax, ax2) = plt.subplots(ncols=2, figsize=(6, 3))
# plot ccw path
patch = plt.Polygon(polygon, zorder=10, fill=False, lw=2)
ax.add_patch(patch)
ax.scatter(x.flatten(), y.flatten(), c=(~inside).astype(float), cmap="RdYlGn")
# plot cw path
patch2 = plt.Polygon(polygon[::-1], zorder=10, fill=False, lw=2)
ax2.add_patch(patch2)
ax2.scatter(x.flatten(), y.flatten(), c=inside2.astype(float), cmap="RdYlGn")

ax.set_title("ccw path")
ax2.set_title("cw path")
plt.show()