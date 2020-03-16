import numpy as np
import sympy as sp

from numpy import concatenate

import sys
from stats import p_th_quantile_chi_squared, p_th_quantile_cdf_normal

sys.path.append('../exps/Models/')
sys.path.append('../src/utils/')

from polygonal_obstacles import *
from polygonal_obstacles import PolygonalObstacle as PolyObs
from ISS import get_ISS_zones

from Models.astrobee import Model

def monte_carlo(X, U, m, N_MC=100):
	n_x, n_u, N   = m.n_x, m.n_u, m.N
	K_lqrs        = m.K_fbs
	mean_eps, Sig_eps = np.zeros(n_x), m.Sig_w
	mass,Ix,Iy,Iz = m.mass, m.J[0,0], m.J[1,1], m.J[2,2]
	mJ_mean       = np.array([mass,Ix,Iy,Iz])
	mJ_var        = m.mJ_var

	x_init = m.x_init

	# sample parameters and disturbances
	params = np.random.multivariate_normal(mJ_mean,mJ_var,   N_MC)
	eps    = np.random.multivariate_normal(mean_eps,Sig_eps, (N, N_MC))
	eps    = np.swapaxes(np.swapaxes(eps,1,0),2,1) # (N_MC, N, n_x)

	Xs_true, Us_true = np.zeros((N_MC, n_x, N)), np.zeros((N_MC, n_u, N-1))

	for i in range(N_MC):
		Xs_true[i,:,:], Us_true[i,:,:] = m.simulate(x_init, X, U, params[i,:], eps[i,:,:])

	# check obs avoidance
	nb_in_obs = np.zeros(N)
	for k in range(N):
		for i in range(N_MC):
			ik_in_obs = False

			x_ki = Xs_true[i,:,k]
			# Spherical Obstacles
			for obs_i in range(len(m.obstacles)):
				val = m.check_obs_con(x_ki, 
										obs_i, obs_type='sphere')
				if not(ik_in_obs) and (val < 0.):
					ik_in_obs = True
					nb_in_obs[k] += 1
			# Rectangular Obstacles
			for obs_i in range(len(m.poly_obstacles)):
				val = m.check_obs_con(x_ki, 
										obs_i, obs_type='poly')
				if not(ik_in_obs) and (val < 0.):
					ik_in_obs = True
					nb_in_obs[k] += 1


	return Xs_true, Us_true, nb_in_obs
