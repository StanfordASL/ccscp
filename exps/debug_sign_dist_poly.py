#!/usr/bin/env python3

import sys, os
import time
import numpy as np
import matplotlib as plt

sys.path.append('../src/utils/')

# from parameters import *
from polygonal_obstacles import *
from viz import *


idx_nplt = 2 # dimension which is not plotted
idx = [0, 1]




class Cursor(object):
    def __init__(self, ax, obs):
        self.obs = obs 

        self.ax = ax

        # self.mouse_x = ax.axhline(color='k')  # the horiz line
        # self.mouse_y = ax.axvline(color='k')  # the vert line
        self.mouse_pt = ax.scatter(0.,0.,color='k')  # the vert line

        # self.pt_x = ax.axhline(color='k')  # the horiz line
        # self.pt_y = ax.axvline(color='k')  # the vert line
        self.obs_pt = ax.scatter(0.,0.,color='b')

        # convexified constraint
        self.n_obsconpts = 20
        self.obs_con,  = ax.plot(0.,0.,color='b')
        # self.obs_good, = ax.plot(0.,0.,color='g')
        # self.obs_bad,  = ax.plot(0.,0.,color='r')
        self.pt_test_pos = np.array([0.7,0.65,0.])
        self.pt_test = ax.scatter(self.pt_test_pos[idx[0]],
        						  self.pt_test_pos[idx[1]],
        						  color='k')

        # text location in axes coords
        self.txt = ax.text(0.2, 0.9, '', transform=ax.transAxes)
        self.txt2= ax.text(0.2, 0.5, '', transform=ax.transAxes)
        self.txt3= ax.text(0.2, 0.1, '', transform=ax.transAxes)

        self.z = 0.

    def mouse_move(self, event):
        if not event.inaxes:
            return

        x, y = event.xdata, event.ydata
        mouse_pos = np.array([x,y])
        self.mouse_pt.set_offsets(mouse_pos)

        x_p = np.zeros(3)
        x_p[idx[0]], x_p[idx[1]], x_p[idx_nplt] = x, y, self.z

        ds, pt_obs = signed_distance_with_closest_point_on_surface(x_p, self.obs)
        self.obs_pt.set_offsets(pt_obs[idx])

        # draw constraint
        self.draw_convexified_obs_constraint(x_p, ds, pt_obs)

        self.txt.set_text('obs_pos: x=%1.2f, y=%1.2f, z=%1.2f' % (pt_obs[0],pt_obs[1],pt_obs[2]))
        self.txt2.set_text('ds = %1.2f' % (ds))

        self.ax.figure.canvas.draw()

    def draw_convexified_obs_constraint(self, x_p, ds, pt_obs):
        obs_c, obs_w = obs.c, obs.widths
        
        # n_prev = (x_p-obs_c) / np.linalg.norm((x_p-obs_c),2)       # (2,)
        if ds>=0.:
            n_prev = (x_p-pt_obs) / np.linalg.norm((x_p-pt_obs),2)       # (2,)
        else:
            n_prev = -(x_p-pt_obs) / np.linalg.norm((x_p-pt_obs),2)       # (2,)

        
        xs = np.linspace(obs.c[idx[0]]-obs_w[idx[0]], 
        				 obs.c[idx[0]]+obs_w[idx[0]], 
        					num=self.n_obsconpts)
        z  = self.z
        
        
        # Ax <= b
        n1, n2, n3 = n_prev[idx[0]], n_prev[idx[1]], n_prev[idx_nplt]
        ys = -(ds - n_prev@x_p + n1*xs + n3*z) / n2

        self.obs_con.set_xdata(xs)
        self.obs_con.set_ydata(ys)

        # ys_good = -(ds + 0.1 - n_prev@x_p + n1*xs + n3*z) / n2
        # ys_bad  = -(ds - 0.1 - n_prev@x_p + n1*xs + n3*z) / n2
        # self.obs_good.set_xdata(xs)
        # self.obs_good.set_ydata(ys_good)
        # self.obs_bad.set_xdata(xs)
        # self.obs_bad.set_ydata(ys_bad)
        con_test_penalty = -(ds + n_prev@(self.pt_test_pos-x_p))
        if con_test_penalty<=0.:
            self.pt_test.set_color('g')
        else: # violated
            self.pt_test.set_color('r')

        # n_prev@(xp-xp)<=b ->>> penality <= 0
        obs_con_penality= ds
        self.txt3.set_text('n_prev@(xp-xp)-b = %1.2f' % (obs_con_penality))


center = np.array([0.1,-0.15,0.])
widths_vec = np.ones(3)
obs = PolygonalObstacle(center, widths_vec)

fig, ax = plt.subplots()

plot_rectangle(ax, center[idx], widths_vec[idx])
plt.xlim([-1.5*widths_vec[idx[0]],2.5*widths_vec[idx[0]]])
plt.ylim([-1.5*widths_vec[idx[1]],2.5*widths_vec[idx[1]]])

if idx[0] == 0:
	plt.xlabel('x')
elif idx[0] == 1:
	plt.xlabel('y')
else:
	plt.xlabel('z')
if idx[1] == 0:
	plt.ylabel('x')
elif idx[1] == 1:
	plt.ylabel('y')
else:
	plt.ylabel('z')


cursor = Cursor(ax, obs)
fig.canvas.mpl_connect('motion_notify_event', cursor.mouse_move)


plt.draw()
plt.show()