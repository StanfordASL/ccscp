import osqp # solver

import numpy as np

from scipy import sparse
from scipy.sparse import vstack, hstack, eye

from copy import copy, deepcopy
from warnings import warn

import matplotlib.pyplot as plt
from ccscp.src.utils.viz import *

default_CCSCP_UP_config = {
    "A_all": [], # required
    "B_all": [], # required
    "return_grad": True,
    "reuse_ABgrads": True, # a call to propagate_uncertainty is always 
                           # preceded by the linearization of the dynamics.
}

class CCOCP:
    """
        CC-OCP: chance-constrained optimal control problem

        Defines a CC-OCP and adds the model specific constraints and objectives.

        Parser from model functions, to a CC-OCP problem formulation, to an OSQP problem.

        Initialization inputs:
            - m: model object (e.g. Astrobee Model)
            - verbose_osqp

        Inner Parameters Information            
        "last" denotes the last iteration around which we linearize

                    OSQP Optimization Problem Definition

                        min     1/2 x^T P x + q^T x
                        s.t.    l <= A x <= u

            with    x = [x(1);x(2),...;x(N);u(1);...;u(N-1);slack_vars],
                        where x(k) is (m.n_x), and u(k) is (m.n_u).
                        
        ********************************************************************************
        INPUTS: - model    - CCSCPModel      (controllers/ccscp/src/ccscp_controller.py)
                - problem  - PlanningProblem (core.py)
        ********************************************************************************
    """
    def __init__(self, model, problem, 
                       UP_config=default_CCSCP_UP_config,
                       verbose_osqp=False):
        # print('[CCOCP::__init__]: Nb. nodes =', problem.N)
        self.N, N = problem.N, problem.N

        # save variables
        m            = model
        self.model   = model
        self.problem = problem

        self.Q0 = problem.Q0

        # OSQP problem object
        self.osqp_prob = osqp.OSQP()

        # Variables:
        self.n_t     = (N-1) # slack_variables for penalization of trust region csontraints
        self.nb_vars = N*m.n_x+(N-1)*m.n_u+self.n_t

        # Solver Parameters
        self.params                = m.scp_config
        self.params['eps_dyn']     = 0.#1e-5
        self.params['padding_obs'] = 0.

        # Optimization Parameters:
        self.par = dict()
        self.par['omega']          = self.params["omega0"]     # penalization weight
        self.par['tr_radius']      = self.params["tr_radius0"] # trust region radius

        # Uncertainty Propagation parameters required for SCP
        self.UP_config = deepcopy(default_CCSCP_UP_config)
        self.UP_config.update(UP_config)

        # INITIALIZATION
        self.par['X_last'], self.par['U_last']           = self.get_XU_initial_trajectory()
        f_all, A_all, B_all                              = m.compute_dynamics(self.par['X_last'], 
                                                                              self.par['U_last'])
        self.UP_config["A_all"], self.UP_config["B_all"] = A_all, B_all
        UP_vals                                          = m.propagate_uncertainty(self.par['X_last'], 
                                                                                   self.par['U_last'], 
                                                                                   self.UP_config)
        self.par['f_all_last']   = f_all
        self.par['A_all_last']   = A_all
        self.par['B_all_last']   = B_all
        self.par['UP_vals_last'] = UP_vals

        # objective and constraints
        self.P, self.q         = self.get_objective_coeffs()
        self.A, self.l, self.u = self.get_all_constraints_coeffs()

        # Setup OSQP problem
        self.verbose_osqp = verbose_osqp
        self.osqp_prob    = osqp.OSQP()
        print("OSQP Problem size: ",
              "P =",self.P.shape,"q =",self.q.shape,
              "A =",self.A.shape,"l =",self.l.shape,"u =",self.u.shape)
        self.osqp_prob.setup(self.P, self.q, self.A, self.l, self.u, 
                            warm_start=True, verbose=self.verbose_osqp)

    def update_horizon(self, N):
        m = self.model

        self.N, self.problem.N = N, N
        self.n_t     = (N-1)
        self.nb_vars  = N*m.n_x+(N-1)*m.n_u+self.n_t

        self.par['X_last'], self.par['U_last'] = self.get_XU_initial_trajectory()
        f_all, A_all, B_all = m.compute_dynamics(self.par['X_last'], self.par['U_last'])
        self.UP_config["A_all"], self.UP_config["B_all"] = A_all, B_all
        UP_vals = m.propagate_uncertainty(self.par['X_last'], self.par['U_last'], self.UP_config)
        self.par['f_all_last']   = f_all
        self.par['A_all_last']   = A_all
        self.par['B_all_last']   = B_all
        self.par['UP_vals_last'] = UP_vals


        # objective and constraints
        self.P, self.q         = self.get_objective_coeffs()
        self.A, self.l, self.u = self.get_all_constraints_coeffs()
        # Setup OSQP problem
        self.osqp_prob  = osqp.OSQP()
        self.osqp_prob.setup(self.P, self.q, self.A, self.l, self.u, 
                            warm_start=True, verbose=self.verbose_osqp)

    def get_XU_initial_trajectory(self, init_method='straightline'):
        if self.problem.B_go_to_safe_set:
            X, U = self.model.initialize_trajectory(self.N, 
                                self.problem.x_init, self.problem.X_safe[0], 
                                init_method=init_method)
        else:
            X, U = self.model.initialize_trajectory(self.N, 
                                self.problem.x_init, self.problem.X_goal[0],
                                init_method=init_method)
        return X, U

    def get_objective_coeffs(self):
        m, problem = self.model, self.problem
        N          = self.problem.N
        n_vars, n_t = self.nb_vars, self.n_t
        

        Q,QN,qN,R,costs,costs_grads = m.get_objective_coeffs(self.par['X_last'], self.par['U_last'], 
                                                             self.problem.X_goal[0])

        # min     1/2 z^T P z + q^T z

        # Quadratic Objective
        Qt = np.zeros((n_t,n_t))      # trust region slack vars
        # R  = m.quadratic_cost_matrix_controls
        P  = sparse.block_diag([sparse.kron(eye(N-1), Q), 2.*QN,
                                sparse.kron(eye(N-1), R), Qt], format='csc')
        # Linear Objective
        q = np.zeros(self.nb_vars)
        # final constraint
        idx_xN, idx_xNn  = (N-1)*m.n_x, N*m.n_x
        q[idx_xN:idx_xNn] = qN
        # trust region
        q += self.get_trust_region_cost_linear()
        # reach/exploration linear cost
        for k in range(N-1):
            idx_xk  = k*m.n_x
            idx_xkn = k*m.n_x + m.n_x
            idx_uk  = N*m.n_x + k*m.n_u

            q[idx_xk :idx_xkn]      += costs_grads[k,:m.n_x] # x_{k}
            q[idx_uk :idx_uk+m.n_u] += costs_grads[k,m.n_x:(n_vars-n_t)] # u_{k}

        return P, q

    def get_all_constraints_coeffs(self):
        # Constraints: 
        #                  l <= A x <= u,     with
        #       x = [x(1);x(2),...;x(N);u(1);...;u(N-1)]
        m, problem = self.model, self.problem

        Aeq_x0,     leq_x0,     ueq_x0     = self.get_initial_constraints_coeffs(m)
        # Aeq_xf,     leq_xf,     ueq_xf     = self.get_final_constraints_coeffs(m)
        Aeq_xf,     leq_xf,     ueq_xf     = self.get_final_constraints_coeffs_convexified(m)
        Aeq_dyn,    leq_dyn,    ueq_dyn    = self.get_dynamics_constraints_coeffs(m)
        Aineq_lims, lineq_lims, uineq_lims = self.get_input_state_min_max_ineq_constraints_coeffs(m)
        Aineq_obs,  lineq_obs,  uineq_obs  = self.get_obs_avoidance_constraints_convexified_coeffs(m)
        A_slack_t,  l_slack_t,  u_slack_t  = self.get_trust_region_constraints_coeffs(m)

        self.x0_constraints_idx   = range(0,
                                          Aeq_x0.shape[0])
        self.xf_constraints_idx   = range(self.x0_constraints_idx[-1],
                                          self.x0_constraints_idx[-1]+Aeq_xf.shape[0])
        self.dyns_constraints_idx = range(self.xf_constraints_idx[-1],
                                          self.xf_constraints_idx[-1]+Aeq_dyn.shape[0])
        self.lims_constraints_idx = range(self.dyns_constraints_idx[-1],
                                          self.dyns_constraints_idx[-1]+Aineq_lims.shape[0])
        self.obs_constraints_idx  = range(self.lims_constraints_idx[-1],
                                          self.lims_constraints_idx[-1]+Aineq_obs.shape[0])
        if len(self.obs_constraints_idx)>0:
            self.trust_constraints_idx= range(self.obs_constraints_idx[-1],
                                              self.obs_constraints_idx[-1]+A_slack_t.shape[0])
        else:
            self.trust_constraints_idx= range(self.dyns_constraints_idx[-1],
                                              self.dyns_constraints_idx[-1]+A_slack_t.shape[0])

        A = sparse.vstack([Aeq_x0, Aeq_xf, Aeq_dyn, Aineq_lims, Aineq_obs, A_slack_t], format='csc')
        l = np.hstack(    [leq_x0, leq_xf, leq_dyn, lineq_lims, lineq_obs, l_slack_t])
        u = np.hstack(    [ueq_x0, ueq_xf, ueq_dyn, uineq_lims, uineq_obs, u_slack_t])

        return A, l, u


    """ ----------- TRUST REGION CONSTRAINTS ------------------
        min  {omega * max(x, 0)}, where x = |z-z^j|_1 - Delta_j
                     <=>
        min         { t }
        s.t. omega*(  |z-z^j| - Delta_j) <= t
                              0          <= t
    """
    def get_trust_region_cost_linear(self):
        #   min  sum_k {t_k}
        q_slack_t = np.ones(self.n_t)
        q_slack_t = np.concatenate((np.zeros(self.nb_vars-self.n_t),q_slack_t), axis=0)
        return q_slack_t

    def get_trust_region_constraints_coeffs(self, m):
        """     
            |z-z^j|_1 <= t/omega + Delta_j
                -t    <= 0
        """
        nb_vars, n_x, n_u, n_t, N = self.nb_vars, m.n_x, m.n_u, self.n_t, self.N
        X_j, U_j,                 = self.par['X_last'], self.par['U_last']
        omega_j, Delta_j          = self.par['omega'], self.par['tr_radius']

        """ array of permutations to reformulate 1-norm as linear constraints
        n_u = 1 -> [1],     [-1]
        n_u = 2 -> [1,1],   [-1,1],   [1,-1],   [-1,-1]
        n_u = 3 -> [1,1,1], [-1,1,1], [1,-1,1], [-1,-1,1], [1,1,-1], [-1,1,-1], [1,-1,-1], [-1,-1,-1]
        n_u = ... """
        mat_arr_weights_u = np.zeros((2**n_u, n_u))
        for ui in range(n_u):
            mat_arr_weights_u[:, ui] = np.array([(-1)**(j//(2**ui)) for j in range(2**n_u)])

        # |z-z^j| - Delta_j <= t/omega 
        n_c_k = 2**n_u      # number of cosntraints per timestep
        Aineq = np.zeros((n_t*n_c_k,nb_vars))
        lineq = -np.inf * np.ones(n_t*n_c_k)
        uineq = np.zeros(n_t*n_c_k)

        for k in range(n_t):
            idx_uk  = N*n_x + k*n_u
            idx_ukn = N*n_x + k*n_u + n_u
            idx_tk  = N*n_x + (N-1)*n_u   + k

            for ci in range(n_c_k):
                Aineq[k*n_c_k+ci, idx_uk:idx_ukn] = mat_arr_weights_u[ci,:]
                Aineq[k*n_c_k+ci, idx_tk        ] = -1./omega_j
                uineq[k*n_c_k+ci]                 = Delta_j + mat_arr_weights_u[ci,:]@U_j[:,k]

        # -t <= 0
        A_t = np.zeros((n_t,nb_vars))
        A_t[:,nb_vars-n_t:] = -np.eye(n_t)
        l_t = -np.inf * np.ones(n_t)
        u_t = np.zeros(n_t)


        A_slack_t = np.concatenate((Aineq, A_t), axis=0)
        l_slack_t = np.concatenate((lineq, l_t), axis=0)
        u_slack_t = np.concatenate((uineq, u_t), axis=0)
        return A_slack_t, l_slack_t, u_slack_t
    #   ----------- TRUST REGION CONSTRAINTS ------------------

    def get_initial_constraints_coeffs(self, m):
        n_vars, n_x, n_u, n_t, N = self.nb_vars, m.n_x, m.n_u, self.n_t, self.N

        Aeq = hstack([eye(n_x), np.zeros((n_x, (N-1)*n_x)), np.zeros((n_x, n_vars-N*n_x))])
        leq = self.problem.x_init
        ueq = leq
        return Aeq, leq, ueq

    def get_final_constraints_coeffs(self, m):
        n_vars, n_x, n_u, n_t, N = self.nb_vars, m.n_x, m.n_u, self.n_t, self.N
        problem, model           = self.problem, m

        # go to final set
        if problem.B_go_to_safe_set:
            Aineq_N, lineq_N, uineq_N = model.get_final_set_constraint(problem.X_safe,
                                            self.par['UP_vals_last'], self.UP_config)
        else:
            Aineq_N, lineq_N, uineq_N = model.get_final_set_constraint(problem.X_goal,
                                            self.par['UP_vals_last'], self.UP_config)

        Aineq = np.hstack([np.zeros((n_x, (N-1)*n_x)), Aineq_N, np.zeros((n_x, n_vars-N*n_x))])
        Aineq = sparse.csr_matrix(Aineq)
        lineq = lineq_N
        uineq = uineq_N
        return Aineq, lineq, uineq

    def get_final_constraints_coeffs_convexified(self, m):
        # same as [::get_final_constraints_coeffs], but with Jacobian of variance

        n_vars, n_x, n_u, n_t, N = self.nb_vars, m.n_x, m.n_u, self.n_t, self.N
        problem, model           = self.problem, m

        # go to final set
        Aineq = np.zeros((2*n_x,n_vars))
        if problem.B_go_to_safe_set:
            A_N, l_N, u_N = model.get_final_set_constraint_convexified(problem.X_safe,
                                                   self.par['X_last'], self.par['U_last'],
                                                   self.par['UP_vals_last'], self.UP_config)
        else:
            A_N, l_N, u_N = model.get_final_set_constraint_convexified(problem.X_goal,
                                                   self.par['X_last'], self.par['U_last'],
                                                   self.par['UP_vals_last'], self.UP_config)

        Aineq[:,        :(N*n_x)     ] = np.reshape(A_N[ :,      :, :n_x], (-1, N*n_x),     order='C')
        Aineq[:, (N*n_x):(n_vars-n_t)] = np.reshape(A_N[ :, :(N-1), n_x:], (-1, (N-1)*n_u), order='C')
        Aineq = sparse.csr_matrix(Aineq)
        lineq = l_N
        uineq = u_N
        return Aineq, lineq, uineq

    def get_input_state_min_max_ineq_constraints_coeffs(self, m):
        n_vars, n_x, n_u, n_t, N   = self.nb_vars, m.n_x, m.n_u, self.n_t, self.N
        p                          = self.problem
        x_min, x_max, u_min, u_max = p.x_min, p.x_max, p.u_min, p.u_max

        Aineq = np.zeros((2*(N*n_x+(N-1)*n_u),n_vars))
        Am, lm, um = m.state_input_constraints_convexified(self.par['X_last'], self.par['U_last'],
                                                           x_min, x_max, u_min, u_max,
                                                           self.par['UP_vals_last'], self.UP_config)
        #                                             nb_i   N      xu
        Aineq[:,        :(N*n_x)     ] = np.reshape(Am[ :,      :, :n_x], (-1, N*n_x),     order='C')
        Aineq[:, (N*n_x):(n_vars-n_t)] = np.reshape(Am[ :, :(N-1), n_x:], (-1, (N-1)*n_u), order='C')
        Aineq = sparse.csr_matrix(Aineq)
        lineq = lm
        uineq = um

        return Aineq, lineq, uineq

    def get_dynamics_constraints_coeffs(self, m):
        n_x, n_u, n_t, N = m.n_x, m.n_u, self.n_t, self.N
        Aeq = np.zeros(((N-1)*n_x, self.nb_vars))
        leq = np.zeros((N-1)*n_x)
        for k in range(N-1):
            X_kj     = self.par['X_last'][:, k]
            U_kj     = self.par['U_last'][:, k]
            f_dyn_kj = self.par['f_all_last'][:, k]
            A_kj     = np.reshape(self.par['A_all_last'][:, k], (n_x, n_x), order='F')
            B_kj     = np.reshape(self.par['B_all_last'][:, k], (n_x, n_u), order='F')

            idx_xk  = k*n_x
            idx_xkn = k*n_x + n_x
            idx_uk  = N*n_x + k*n_u
            # x_{k+1} = f_kj + A_kj@(x_{k}-xj_{k}) + B_kj@(u_{k}-uj_{k}
            Aeq[idx_xk:idx_xkn, idx_xk :idx_xkn]     = A_kj           # x_{k}
            Aeq[idx_xk:idx_xkn, idx_uk :idx_uk +n_u] = B_kj           # u_{k}
            Aeq[idx_xk:idx_xkn, idx_xkn:idx_xkn+n_x] = -np.eye(n_x)    # x_{k+1}
            leq[idx_xk:idx_xkn] = - ( f_dyn_kj - A_kj@X_kj - B_kj@U_kj )
            ueq                 = leq.copy()
        Aeq = sparse.csr_matrix(Aeq)
        leq -= self.params['eps_dyn']
        ueq += self.params['eps_dyn']
        return Aeq, leq, ueq


    def update_constraints(self, m):
        """ Problem in OSQP is formatted in the format:

            min     1/2 x^T P x + q^T x
            s.t.    l <= A x <= u

            Convention:        ### TODO USE INDICES AS GLOBAL VARIABLES !!!
                - x = (x(0),x(1),...,x(N),u(0),...,u(N-1))
                - the first nx constraints l[:nx], u[:nx] 
                    correspond to the initial conditions equality constraints.
                - the next nx constraints l[nx:2*nx], u[nx:2*nx] 
                    correspond to the final conditions equality constraints.
        """
        # self.update_dynamics_constraints(m)
        # self.update_obs_avoidance_constraints(m)
        # self.osqp_prob.update(Ax=self.A.data,l=self.l,u=self.u)
        self.P, self.q         = self.get_objective_coeffs()
        self.A, self.l, self.u = self.get_all_constraints_coeffs()
        self.osqp_prob.update(Ax=self.A.data,l=self.l,u=self.u)

        return False
        
    def update_dynamics_constraints_coeffs(self, m, dyn_constraints_idx):
        n_x, n_u, n_t, N = m.n_x, m.n_u, self.n_t, self.N
        id0 = dyn_constraints_idx[0]
        for k in range(N-1):
            X_kj     = self.par['X_last'][:, k]
            U_kj     = self.par['U_last'][:, k]
            f_dyn_kj = self.par['f_all_last'][:, k]
            A_kj     = np.reshape(self.par['A_all_last'][:, k], (n_x, n_x), order='F')
            B_kj     = np.reshape(self.par['B_all_last'][:, k], (n_x, n_u), order='F')

            idx_xk  = k*n_x
            idx_xkn = k*n_x + n_x
            idx_uk  = N*n_x + k*n_u

            idx_c_k  = (id0 + idx_xk)
            idx_c_kn = (id0 + idx_xkn)

            # x_{k+1} = f_kj + A_kj@(x_{k}-xj_{k}) + B_kj@(u_{k}-uj_{k}
            self.A[idx_c_k:idx_c_kn, idx_xk :idx_xkn]     = A_kj           # x_{k}
            self.A[idx_c_k:idx_c_kn, idx_uk :idx_uk +n_u] = B_kj           # u_{k}
            self.A[idx_c_k:idx_c_kn, idx_xkn:idx_xkn+n_x] = -eye(n_x)    # x_{k+1}
            self.l[idx_c_k:idx_c_kn] = - ( f_dyn_kj - A_kj@X_kj - B_kj@U_kj )
            self.u[idx_c_k:idx_c_kn] = self.l[idx_c_k:idx_c_kn]
        self.l -= self.params['eps_dyn']
        self.u += self.params['eps_dyn']

        return True

    def update_dynamics_constraints(self, m):
        f_all, A_all, B_all = m.compute_dynamics(self.par['X_last'], self.par['U_last'])
        self.par['f_all_last'] = f_all
        self.par['A_all_last'] = A_all
        self.par['B_all_last'] = B_all

        return self.update_dynamics_constraints_coeffs(m, self.dyns_constraints_idx)

    def update_obs_avoidance_constraints(self, m):
        Aineq, lineq, uineq = self.get_obs_avoidance_constraints_convexified_coeffs(m)

        idx_constraints = self.obs_constraints_idx
        self.A[idx_constraints, :] = Aineq
        self.l[idx_constraints]    = lineq
        self.u[idx_constraints]    = uineq

        return True

    def get_obs_avoidance_constraints_convexified_coeffs(self, m):
        n_vars, n_x, n_u, n_t, N = self.nb_vars, m.n_x, m.n_u, self.n_t, self.N
        m, problem               = self.model, self.problem

        # Spherical Obstacles
        n_obs = len(problem.sphere_obstacles)
        Aineq = np.zeros((N*n_obs, self.nb_vars))
        lineq = -np.inf * np.ones(N*n_obs)
        uineq = np.zeros(N*n_obs)
        for k in range(N):
            for obs_i, obs in enumerate(problem.sphere_obstacles):
                A_i, b_i = m.obs_avoidance_constraint_convexified(
                                    self.par['X_last'], self.par['U_last'], 
                                    self.par['UP_vals_last'], self.UP_config, 
                                    k, obs, obs_type='sphere')
                Aineq[k*n_obs+obs_i,        :(N*n_x)]      = np.reshape(A_i[:,     :n_x], (N*n_x),     order='C')
                Aineq[k*n_obs+obs_i, (N*n_x):(n_vars-n_t)] = np.reshape(A_i[:(N-1),n_x:], ((N-1)*n_u), order='C')
                uineq[k*n_obs+obs_i]                       = b_i

        # Rectangular Obstacles
        n_obs      = len(problem.poly_obstacles)
        Aineq_poly = np.zeros((N*n_obs, self.nb_vars))
        lineq_poly = -np.inf * np.ones(N*n_obs)
        uineq_poly = np.zeros(N*n_obs)
        for k in range(N):
            for obs_i, obs in enumerate(problem.poly_obstacles):
                A_i, b_i = m.obs_avoidance_constraint_convexified(
                                    self.par['X_last'], self.par['U_last'], 
                                    self.par['UP_vals_last'], self.UP_config,
                                    k, obs, obs_type='poly')
                Aineq_poly[k*n_obs+obs_i,        :(N*n_x)]      = np.reshape(A_i[:,     :n_x], (N*n_x),     order='C')
                Aineq_poly[k*n_obs+obs_i, (N*n_x):(n_vars-n_t)] = np.reshape(A_i[:(N-1),n_x:], ((N-1)*n_u), order='C')
                uineq_poly[k*n_obs+obs_i]                       = b_i

        Aineq = np.concatenate((Aineq, Aineq_poly), axis=0)
        lineq = np.concatenate((lineq, lineq_poly), axis=0)
        uineq = np.concatenate((uineq, uineq_poly), axis=0)

        Aineq = sparse.csr_matrix(Aineq)
        lineq = lineq - self.params['padding_obs']
        uineq = uineq + self.params['padding_obs']

        return Aineq, lineq, uineq

    def solve_OSQP(self):
        B_problem_solved = True

        self.res = self.osqp_prob.solve()
        if self.res.info.status != 'solved':
            warn("[solve_OSQP]: Problem unfeasible.")
            B_problem_solved = False

        return B_problem_solved

    def solve_ccscp(self, init_method='straightline'):
        m, problem, N = self.model, self.problem, self.N

        (tr_radius0, omega0, omegamax, epsilon, rho0, 
         rho1, beta_succ, beta_fail, gamma_fail, conv_thresh) = self.extract_scp_params()
        self.par['omega']     = self.params["omega0"]             
        self.par['tr_radius'] = self.params["tr_radius0"] 

        # Initialization
        self.par['X_last'], self.par['U_last'] = self.get_XU_initial_trajectory(init_method)
        X = self.par['X_last'].copy(); Xp = X.copy()
        U = self.par['U_last'].copy(); Up = U.copy()
        
        all_X, all_U, all_UPvals = [], [], []

        it = 0  
        B_success = False
        while it<self.params["NB_SCP_iter_max"] and \
              not(it!=0 and it!=1 and it!=2 and \
                  B_success and self.convergence_metric(X,U,Xp,Up)<conv_thresh) and \
              self.par['omega']<self.params["omegamax"]:
            # print('-' * 50)
            # print('Iteration ' + str(it))
            # print('-' * 50)

            B_success = False

            # Relinearize dynamics and propagate uncertainty
            self.par['X_last'], self.par['U_last'] = X.copy(), U.copy()
            f_all, A_all, B_all = m.compute_dynamics(self.par['X_last'], self.par['U_last'])
            self.UP_config["A_all"], self.UP_config["B_all"] = A_all, B_all
            UP_vals = m.propagate_uncertainty(self.par['X_last'], self.par['U_last'], self.UP_config)
            self.par['f_all_last']   = f_all
            self.par['A_all_last']   = A_all
            self.par['B_all_last']   = B_all
            self.par['UP_vals_last'] = UP_vals

            all_X.append(X.copy())
            all_U.append(U.copy())
            all_UPvals.append(UP_vals.copy())

            # Help to find feasible solution at first iteration (not always necessary)
            if it == 0:   self.params['padding_obs'] = 0.1
            # elif it == 1: self.params['padding_obs'] = 0.5
            else:         self.params['padding_obs'] = 0.

            self.update_constraints(m)

            self.B_solved_successfully = self.solve_OSQP()
            X_sol, U_sol               = self.get_XU_solution_OSQP()


            # PLOT
            # UP_config_plot = deepcopy(m.UP_config); UP_config_plot.update(self.UP_config)
            # Kfbs = []
            # if UP_config_plot['B_feedback']:
            #     Kfbs = m.get_feedback_gains(B_reuse_precomputed=True)
            # self.plot_inner_SCP_loop(X, U, 
            #                          UP_config_plot, 
            #                          UP_vals,
            #                          Kfbs,
            #                          self.problem.u_min, self.problem.u_max)


            if self.B_solved_successfully == False:
                print('[solve_ccscp] Failure to solve SCP iter #'+str(it))
                self.all_X = np.stack(all_X.copy())
                self.all_U = np.stack(all_U.copy())
                self.all_UPvals = np.stack(all_UPvals.copy())
                return False

            # check trust region
            Xp, Up = self.par['X_last'].copy(), self.par['U_last'].copy()
            # print('np.linalg.norm(X_sol-Xp,2) = ' + str(np.linalg.norm(X_sol-Xp,2)))
            if np.linalg.norm(X_sol-Xp,2) < self.par['tr_radius']:
                rho = self.accuracy_ratio(m, X_sol,U_sol,Xp,Up)

                if rho > rho1:
                    print('Reject solution.')
                    self.par['tr_radius'] = beta_fail * self.par['tr_radius']
                    self.par['omega']     = self.par['omega']
                    B_success = False

                else:
                    # print('-' * 50,'\nAccept solution.\n','-' * 50)
                    # print('Accept solution.')
                    X, U = X_sol.copy(), U_sol.copy()
                    B_success = True
                    if rho < rho0:
                        self.par['tr_radius'] = np.minimum(beta_succ*self.par['tr_radius'], tr_radius0)
                    else:
                        self.par['tr_radius'] = self.par['tr_radius']
            else:
                print('Reject solution (Outside trust region)')
                print('norm(x_sol-xp,2)=',np.linalg.norm(X_sol-Xp,2))
                self.par['omega'] = gamma_fail * self.par['omega']
                B_success         = False

            it += 1

        self.all_X, self.all_U, self.all_UPvals = np.stack(all_X), np.stack(all_U), np.stack(all_UPvals)

        print('[solve_ccscp] Success: '+str(B_success)+', Nb of iterations: '+str(it))
        return B_success


    def get_XU_solution_OSQP(self):
        nb_vars, n_x, n_u, n_t, N = self.nb_vars, self.model.n_x, self.model.n_u, self.n_t, self.N

        X_sol = np.reshape(self.res.x[:N*n_x],            (n_x, N),   order='F')
        U_sol = np.reshape(self.res.x[N*n_x:nb_vars-n_t], (n_u, N-1), order='F')
        return X_sol, U_sol

    def get_XU_solution_CCSCP(self):
        if self.B_solved_successfully:
            return self.get_XU_solution_OSQP()
        else:
            return self.all_X[-1], self.all_U[-1]



    def convergence_metric(self, X, U, Xp, Up):
        conv_metric = ((np.linalg.norm(X-Xp,2)/np.linalg.norm(Xp,2)) + 
                       (np.linalg.norm(U-Up,2)/np.linalg.norm(Up,2)))
        print('Convergence Metric: '+"{0:.2f}".format(100.*conv_metric)+'%')
        return conv_metric

    def accuracy_ratio(self, m, X, U, Xp, Up):
        m = self.model

        num, den = 0.0, 0.0
        for k in range(self.N-1):
            x_k, u_k = X[:,k],  U[:,k]
            x_p, u_p = Xp[:,k], Up[:,k]

            f_dyn_k, _, _ = m.compute_dynamics(x_k, u_k)
            f_dyn_p =            self.par['f_all_last'][:, k]
            A_dyn_p = np.reshape(self.par['A_all_last'][:, k], (m.n_x, m.n_x), order='F')
            B_dyn_p = np.reshape(self.par['B_all_last'][:, k], (m.n_x, m.n_u), order='F')

            linearized = f_dyn_p + A_dyn_p@(x_k-x_p) + B_dyn_p@(u_k-u_p)
            num += np.dot((f_dyn_k - linearized),(f_dyn_k - linearized))
            den += np.dot(linearized,linearized)

        accuracy_ratio = num/den
        return accuracy_ratio

    def extract_scp_params(self):
        (tr_radius0, omega0, 
         omegamax,   epsilon, 
         rho0,       rho1, 
         beta_succ,  beta_fail, 
         gamma_fail, conv_thresh) = (self.params["tr_radius0"], self.params["omega0"], 
                                     self.params["omegamax"], self.params["epsilon"], 
                                     self.params["rho0"], self.params["rho1"], 
                                     self.params["beta_succ"], self.params["beta_fail"], 
                                     self.params["gamma_fail"], self.params["convergence_threshold"])
        return (tr_radius0, omega0, omegamax, epsilon, rho0, 
                rho1, beta_succ, beta_fail, gamma_fail, conv_thresh)


    def plot_inner_SCP_loop(self, X_sol, U_sol, UP_config, UP_vals, K_fbs, u_min, u_max):
        T = X_sol.shape[1]

        if UP_config['B_feedback']:
            Deltas_u = np.einsum('tux,xt->tu', K_fbs, UP_vals['Deltas'][:,:(T-1)])

        plt.figure(figsize=[15,3])
        plt.subplot(1, 3, 1)
        plt.plot(X_sol[0,:], X_sol[1,:])
        plot_mean_traj_with_uncertainty(X_sol.T, UP_vals, UP_config)

        # plt.xlim([0.2,0.6])
        plt.xlim([0.,0.4])
        plt.ylim([-0.5,-0.1])

        plt.subplot(1, 3, 2)
        # # plt.scatter(U_sol[0,0], U_sol[1,0])
        # # plt.plot(U_sol[0,:], U_sol[1,:])
        for i in [0,1]:
            plt.plot(np.arange(T-1), U_sol[i,:])
            if UP_config['B_feedback']:
                plt.fill_between(np.arange(T-1), U_sol[i,:]-Deltas_u[:,i], U_sol[i,:]+Deltas_u[:,i], color='b', alpha=0.2)
        plt.grid(True)
        # # plt.xlim([-0.29,0.29])  
        plt.axhline(y=u_min[i], color='r') 
        plt.axhline(y=u_max[i], color='r') 
        plt.ylim([2*-0.29,2*0.29])

        plt.subplot(1, 3, 3)
        for i in [2]:
            plt.plot(np.arange(T-1), U_sol[i,:])
            if UP_config['B_feedback']:
                plt.fill_between(np.arange(T-1), U_sol[i,:]-Deltas_u[:,i], U_sol[i,:]+Deltas_u[:,i], color='b', alpha=0.2)
            plt.axhline(y=u_min[i], color='r') 
            plt.axhline(y=u_max[i], color='r') 

        plt.show()
