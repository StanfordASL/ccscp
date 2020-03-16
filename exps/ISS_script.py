import sys, os
sys.path.append('../src/utils/')
sys.path.append('../exps/Models/')

import numpy as np
import matplotlib as plt
from polygonal_obstacles import PolygonalObstacle as PolyObs
from viz import *
from ISS import get_ISS_zones


lims_btm, lims_up = np.array([5.,-3.5, 3.]), np.array([13., 8.5, 6.5])

keepin_zones, keepout_zones = get_ISS_zones()

# --------------------------------------------
plt.figure(1)
ax = plt.gca()
# --------------------------------------------
for obs in keepin_zones:
	center, widths = obs.c, 2*np.array([obs.dx,obs.dy,obs.dz])
	plot_rectangle(ax, center[:2], widths[:2], color='g')
for obs in keepout_zones:
	center, widths = obs.c, 2*np.array([obs.dx,obs.dy,obs.dz])
	plot_rectangle(ax, center[:2], widths[:2], color='r')


plt.xlim([lims_btm[0], lims_up[0]])
plt.ylim([lims_btm[1], lims_up[1]])
plt.draw()
plt.show()
# --------------------------------------------