import osqp # solver

import numpy as np

from scipy import sparse
from scipy.sparse import vstack, hstack, eye

from warnings import warn


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
    """
    def __init__(self, m, verbose_osqp=False):        
        print('[CCOCP::__init__]: Nb. nodes =', m.N)
        self.N, N = m.N, m.N

        # Variables:
        self.n_t     = (N-1) # slack_variables for penalization of trust region csontraints
        self.nb_vars = N*m.n_x+(N-1)*m.n_u+self.n_t

        # Optimization Parameters:
        self.par               = dict()
        self.par['X_last']     = np.empty(shape=[m.n_x,       N  ])
        self.par['U_last']     = np.empty(shape=[m.n_u,       N-1])
        self.par['f_all_last'] = np.empty(shape=[m.n_x,       N-1])
        self.par['A_all_last'] = np.empty(shape=[m.n_x*m.n_x, N-1])
        self.par['B_all_last'] = np.empty(shape=[m.n_x*m.n_u, N-1])

        # Solver Parameters
        self.params                = m.scp_params
        self.params['eps_dyn']     = 1e-5
        self.params['padding_obs'] = 0.
        self.par['omega']          = self.params["omega0"]     # penalization weight
        self.par['tr_radius']      = self.params["tr_radius0"] # trust region radius

        # ----------------------------------------------------------------------
        # INITIALIZATION
        self.par['X_last'], self.par['U_last'] = m.initialize_trajectory(N)
        f_all, A_all, B_all    = m.compute_dynamics(   self.par['X_last'], self.par['U_last'])
        Vars_all, Vars_dxu_all = m.propagate_variances(self.par['X_last'], self.par['U_last'],
                                                       A_all, B_all)
        self.par['f_all_last']        = f_all
        self.par['A_all_last']        = A_all
        self.par['B_all_last']        = B_all
        self.par['Vars_all_last']     = Vars_all
        self.par['Vars_dxu_all_last'] = Vars_dxu_all

        # objective and constraints
        self.P, self.q         = self.get_objective_coeffs(m)
        self.A, self.l, self.u = self.get_all_constraints_coeffs(m)

        # Setup OSQP problem
        self.verbose_osqp = verbose_osqp
        self.prob         = osqp.OSQP()
        self.prob.setup(self.P, self.q, self.A, self.l, self.u, 
                        warm_start=True, 
                        verbose=self.verbose_osqp)
        print("OSQP Problem size: ",
              "P =",self.P.shape,"q =",self.q.shape,
              "A =",self.A.shape,"l =",self.l.shape,"u =",self.u.shape)


    def get_objective_coeffs(self, m):
        # min     1/2 z^T P x + q^T z
        N, n_t = self.N, self.n_t

        # Quadratic Objective
        Q  = m.quadratic_cost_matrix_state
        QN = np.zeros((m.n_x,m.n_x))
        Qt = np.zeros((n_t,n_t))      # trust region slack vars
        R  = m.quadratic_cost_matrix_controls
        P  = sparse.block_diag([sparse.kron(eye(N-1), Q), QN,
                                sparse.kron(eye(N-1), R), Qt], format='csc')
        # Linear Objective
        q = np.zeros(self.nb_vars)
        q += self.get_trust_region_cost_linear()

        return P, q

    def get_all_constraints_coeffs(self, m):
        # Constraints: 
        #                  l <= A x <= u,     with
        #       x = [x(1);x(2),...;x(N);u(1);...;u(N-1)]
        Aeq_x0,     leq_x0,     ueq_x0     = self.get_initial_constraints_coeffs(m)
        Aeq_xf,     leq_xf,     ueq_xf     = self.get_final_constraints_coeffs(m)
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
        self.trust_constraints_idx= range(self.obs_constraints_idx[-1],
                                          self.obs_constraints_idx[-1]+A_slack_t.shape[0])

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
        leq = m.x_init
        ueq = leq
        return Aeq, leq, ueq

    def get_final_constraints_coeffs(self, m):
        n_vars, n_x, n_u, n_t, N = self.nb_vars, m.n_x, m.n_u, self.n_t, self.N

        Aeq = hstack([np.zeros((n_x, (N-1)*n_x)), eye(n_x), np.zeros((n_x, n_vars-N*n_x))])
        leq = m.x_final - 5e-2
        ueq = m.x_final + 5e-2
        return Aeq, leq, ueq

    def get_input_state_min_max_ineq_constraints_coeffs(self, m):
        n_vars, n_x, n_u, n_t, N = self.nb_vars, m.n_x, m.n_u, self.n_t, self.N

        Aineq = np.zeros((N*n_x+(N-1)*n_u,n_vars))
        Am, lm, um = m.state_input_constraints_convexified(self.par['X_last'], self.par['U_last'],
                                                  B_uncertainty=True, 
                                                  Sigmas=self.par['Vars_all_last'], 
                                                  Sigmas_dxu=self.par['Vars_dxu_all_last'])
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
        # self.prob.update(Ax=self.A.data,l=self.l,u=self.u)
        P, q    = self.get_objective_coeffs(m)
        A, l, u = self.get_all_constraints_coeffs(m)
        self.P, self.q = P, q
        self.A, self.l, self.u = A, l, u
        self.prob.update(Ax=self.A.data,l=self.l,u=self.u)

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

        # Spherical Obstacles
        n_obs = len(m.obstacles)
        Aineq = np.zeros((N*n_obs, self.nb_vars))
        lineq = -np.inf * np.ones(N*n_obs)
        uineq = np.zeros(N*n_obs)
        for k in range(N):
            Var_k     = self.par['Vars_all_last'][:,:,k]
            Var_dxu_k = self.par['Vars_dxu_all_last'][:,:,:,:,k]
            for obs_i in range(n_obs):
                A_i, b_i = m.obs_avoidance_constraint_convexified(self.par['X_last'], self.par['U_last'], 
                                    obs_i, k, B_uncertainty=True, Sigma_k=Var_k, Sigma_dxu_k=Var_dxu_k,
                                    obs_type='sphere') # [N,n_xu]
                Aineq[k*n_obs+obs_i,        :(N*n_x)]      = np.reshape(A_i[:,     :n_x], (N*n_x),     order='C')
                Aineq[k*n_obs+obs_i, (N*n_x):(n_vars-n_t)] = np.reshape(A_i[:(N-1),n_x:], ((N-1)*n_u), order='C')
                uineq[k*n_obs+obs_i]                       = b_i

        # Rectangular Obstacles
        n_obs      = len(m.poly_obstacles)
        Aineq_poly = np.zeros((N*n_obs, self.nb_vars))
        lineq_poly = -np.inf * np.ones(N*n_obs)
        uineq_poly = np.zeros(N*n_obs)
        for k in range(N):
            Var_k     = self.par['Vars_all_last'][:,:,k]
            Var_dxu_k = self.par['Vars_dxu_all_last'][:,:,:,:,k]
            for obs_i in range(n_obs):
                A_i, b_i = m.obs_avoidance_constraint_convexified(self.par['X_last'], self.par['U_last'], 
                                    obs_i, k, B_uncertainty=True, Sigma_k=Var_k, Sigma_dxu_k=Var_dxu_k,
                                    obs_type='poly') # [N,n_xu]
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

        self.res = self.prob.solve()
        if self.res.info.status != 'solved':
            warn("[solve_OSQP]: Problem unfeasible.")
            B_problem_solved = False

        return B_problem_solved

    def solve_ccscp(self, m):
        N = self.N

        (tr_radius0, omega0, omegamax, epsilon, rho0, 
         rho1, beta_succ, beta_fail, gamma_fail, conv_thresh) = self.extract_scp_params()
        self.par['omega']     = self.params["omega0"]             
        self.par['tr_radius'] = self.params["tr_radius0"] 

        # Initialization
        self.par['X_last'], self.par['U_last'] = m.initialize_trajectory(N)
        X = self.par['X_last'].copy(); Xp = X.copy()
        U = self.par['U_last'].copy(); Up = U.copy()
        
        all_X, all_U, all_V = [], [], []

        it = 0  
        B_success = False
        while it<self.params["NB_SCP_iter_max"] and \
              not(it!=0 and B_success and self.convergence_metric(X,U,Xp,Up)<conv_thresh) and \
              self.par['omega']<self.params["omegamax"]:
            print('\n'+'=' * 50)
            print('Iteration ' + str(it))
            print('-' * 50)

            B_success = False

            self.par['X_last'], self.par['U_last'] = X.copy(), U.copy()
            f_all, A_all, B_all    = m.compute_dynamics(   self.par['X_last'], self.par['U_last'])
            Vars_all, Vars_dxu_all = m.propagate_variances(self.par['X_last'], self.par['U_last'],
                                                            A_all, B_all)
            self.par['f_all_last']        = f_all
            self.par['A_all_last']        = A_all
            self.par['B_all_last']        = B_all
            self.par['Vars_all_last']     = Vars_all
            self.par['Vars_dxu_all_last'] = Vars_dxu_all

            all_X.append(X.copy())
            all_U.append(U.copy())
            all_V.append(Vars_all.copy())

            # Help to find feasible solution at first iteration (not always necessary)
            if it == 0: self.params['padding_obs'] = 0.1
            else:       self.params['padding_obs'] = 0.

            self.update_constraints(m)

            self.B_solved_successfully = self.solve_OSQP()
            X_sol, U_sol               = self.get_XU_solution_OSQP(m)

            if self.B_solved_successfully == False:
                print('[solve_ccscp] Failure to solve SCP iter #'+str(it))
                self.all_X = np.stack(all_X.copy())
                self.all_U = np.stack(all_U.copy())
                self.all_V = np.stack(all_V.copy())
                return False

            # check trust region
            Xp, Up = self.par['X_last'].copy(), self.par['U_last'].copy()
            print('np.linalg.norm(X_sol-Xp,2) = ' + str(np.linalg.norm(X_sol-Xp,2)))
            if np.linalg.norm(X_sol-Xp,2) < self.par['tr_radius']:
                rho = self.accuracy_ratio(m, X_sol,U_sol,Xp,Up)

                if rho > rho1:
                    print('Reject solution.')
                    self.par['tr_radius'] = beta_fail * self.par['tr_radius']
                    self.par['omega']     = self.par['omega']
                    B_success = False

                else:
                    print('-' * 50)
                    print('Accept solution.')
                    print('-' * 50)
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

        self.all_X, self.all_U, self.all_V = np.stack(all_X), np.stack(all_U), np.stack(all_V)

        print('[solve_ccscp] Success: '+str(B_success)+', Nb of iterations: '+str(it))
        return True


    def get_XU_solution_OSQP(self, m):
        nb_vars, n_x, n_u, n_t, N = self.nb_vars, m.n_x, m.n_u, self.n_t, self.N
        X_sol = np.reshape(self.res.x[:N*n_x],            (n_x, N),   order='F')
        U_sol = np.reshape(self.res.x[N*n_x:nb_vars-n_t], (n_u, N-1), order='F')
        return X_sol, U_sol

    def get_XU_solution_CCSCP(self, m):
        return self.get_XU_solution_OSQP(m)



    def convergence_metric(self, X, U, Xp, Up):
        conv_metric = ((np.linalg.norm(X-Xp,2)/np.linalg.norm(Xp,2)) + 
                       (np.linalg.norm(U-Up,2)/np.linalg.norm(Up,2)))
        # print('Convergence Metric: '+"{0:.2f}".format(100.*conv_metric)+'%')
        return conv_metric

    def accuracy_ratio(self, m, X, U, Xp, Up):
        num, den = 0.0, 0.0
        for k in range(self.N-1):
            x_k, u_k = X[:,k],  U[:,k]
            x_p, u_p = Xp[:,k], Up[:,k]

            f_dyn_k = np.squeeze(m.f_dt(x_k, u_k))
            f_dyn_p, A_dyn_p, B_dyn_p = m.get_dynamics(x_p, u_p)
            f_dyn_p = (self.par['f_all_last'][:, k])

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

    def check_obs_avoidance_constraints_satisfied(self, m, X):
        N     = self.N
        n_obs = len(m.obstacles)
        B_inside_obs = False
        for k in range(N):
            x_k = X[:,k]
            for obs_i in range(n_obs):
                penalty = m.obs_avoidance_constraint(x_k, obs_i)
                if penalty>1e-9:
                    B_inside_obs = True
                    print("robot inside obstacle ",obs_i,"at k=",k,": penalty=",penalty)

        return not(B_inside_obs)
