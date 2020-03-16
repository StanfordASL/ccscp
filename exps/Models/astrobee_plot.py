import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import art3d

import sys
sys.path.append('../src/utils/')
sys.path.append('../exps/Models/')
from polygonal_obstacles import PolygonalObstacle as PolyObs
from viz import *
from ISS import get_ISS_zones

figures_i = 0


def plot_obstacles(ax, obstacles, keepin_zones, poly_obs):
    for obs in obstacles:
        pos, radius = obs[0], obs[1]
        ax = plot_circle(ax, pos, radius)
    for obs in keepin_zones:
        center, widths = obs.c, 2*np.array([obs.dx,obs.dy,obs.dz])
        ax = plot_rectangle(ax, center[:2], widths[:2], color='g')
    for obs in poly_obs:
        center, widths = obs.c, 2*np.array([obs.dx,obs.dy,obs.dz])
        ax = plot_rectangle(ax, center[:2], widths[:2], color='r')
    return ax

def plot_circle(ax, pos, radius, color='k', alpha=1.):
    circle = plt.Circle(pos, radius=radius, color=color, fill=True, alpha=alpha)
    ax.add_artist(circle)
    return ax


# ----------------------------------------------------
def key_press_event(event):
    global figures_i
    fig = event.canvas.figure

    if event.key == 'q' or event.key == 'escape':
        plt.close(event.canvas.figure)
        return

    if event.key == 'right':
        figures_i = (figures_i + 1) % figures_N
    elif event.key == 'left':
        figures_i = (figures_i - 1) % figures_N

    fig.clear()
    ax = my_plot(fig, figures_i)
    ax = plot_obstacles(ax, obstacles, keepin_zones, poly_obs)
    plt.draw()



def my_plot(fig, figures_i):
    ax = fig.add_subplot(111)

    X_i = X[figures_i, :, :]
    U_i = U[figures_i, :, :]
    V_i = np.moveaxis(V[figures_i], -1, 0) # (N, n_x,n_x)
    K = X_i.shape[1]

    plt.figure(1)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')

    plot_mean_traj_with_gaussian_uncertainty_ellipses(X_i.T, V_i, 
                                                      color='blue', probability=0.9)
    # ax.plot(X_i[0, :], X_i[1, :], color='lightgrey', zorder=0)
    # ax.plot(X_i[0, :], X_i[1, :], 'o', color='lightgrey', zorder=0)

    ax.set_title("Iteration " + str(figures_i))

    return ax

def plot(X_in, U_in, V_in, model):
    global figures_N
    figures_N = X_in.shape[0]
    figures_i = figures_N - 1
    global X, U, V, obstacles, poly_obs
    X = X_in
    U = U_in
    V = V_in
    obstacles, poly_obs = model.obstacles, model.poly_obstacles
    global keepin_zones
    keepin_zones, keepout_zones = get_ISS_zones()

    fig = plt.figure(figsize=(10, 12))
    ax = my_plot(fig, figures_i)
    ax = plot_obstacles(ax, obstacles, keepin_zones, poly_obs)
    cid = fig.canvas.mpl_connect('key_press_event', key_press_event)


    # Plot controls
    # plt.figure(2)
    # for ui in range(U_in.shape[1]):
    #     plt.subplot(00ui)
    #     plt.plot(range(U_in[ui,:]),U_in[ui,:])

    U_end = U_in[-1,:,:]
    print(U_end.shape)
    fig, axs = plt.subplots(U_end.shape[0])
    for ui in range(U_end.shape[0]):
        axs[ui].plot(list(range(0,U_end.shape[1])),U_end[ui,:])
        axs[ui].set_title(str(ui) + "-th control of last SCP iter.")
    plt.draw()
    plt.figure(1)


    X_end = X_in[-1,:,:]
    print(X_end.shape)
    fig, axs = plt.subplots(X_end.shape[0])
    for xi in range(X_end.shape[0]):
        axs[xi].plot(list(range(0,X_end.shape[1])),X_end[xi,:])
        axs[xi].set_title(str(xi) + "-th state of last SCP iter.")
    plt.draw()
    plt.figure(1)


    lims_btm, lims_up = np.array([5.,-3.5, 3.]), np.array([13., 8.5, 6.5])
    plt.xlim([lims_btm[0], lims_up[0]])
    plt.ylim([lims_btm[1], lims_up[1]])
    plt.show()