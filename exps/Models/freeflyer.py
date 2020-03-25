import sys
sys.path.append('../src/utils/')

import numpy as np
import sympy as sp

from stats import p_th_quantile_chi_squared, p_th_quantile_cdf_normal

from polygonal_obstacles import *
from polygonal_obstacles import PolygonalObstacle as PolyObs

# 6-states, 3-controls freeflyer system
# Nonlinear dynamics with center of mass offset (see get_equations_dt)

class Model:
    n_x = 6
    n_u = 3

    # robot constants
    robot_radius     = 0.001#0.05
    mass             = 14.4
    J                = 0.1083
    hard_limit_vel   = 0.20         # m/s
    hard_limit_accel = 0.02         # m/s^2
    hard_limit_omega = 10*3.14/180  # rad/s
    hard_limit_alpha = 10*3.14/180  # rad/s^2

    # constraints / limits
    x_max = [100., 100., 20.*3.14, hard_limit_vel,hard_limit_vel,hard_limit_omega]   # m     # state limits          
    u_max = [mass*hard_limit_accel,mass*hard_limit_accel,J*hard_limit_alpha]
    x_min = np.negative(x_max)
    u_min = np.negative(u_max)

    # problem 
    x_init  = np.array([0.4,-0.3,0.01,0,0,1e-9]) + 1e-6
    x_final = np.array([-0.3,0.4,0.01,0,0,1e-9]) + 1e-6
    N        = 41               # nb discretization steps
    tf_guess = 41.              # s
    dt       = tf_guess / (N-1) # discretization time

    # Uncertainty / chance constraints
    Sig_w = 0.2*np.diag([1e-7,1e-7,1e-6,1e-7,1e-7,1e-6])
    prob  = 0.9         # probability threshold for chance constraints
    n_p   = 2           # number of positional dimensions for obs. avoid.
    p_quant_chi_sqrt = np.sqrt(p_th_quantile_chi_squared(0.9, n_p))

    # Feedback controller parameters
    B_feedback = False

    quadratic_cost_matrix_controls = np.eye(n_u)
    quadratic_cost_matrix_state    = np.zeros((n_x,n_x))

    # CC-SCP Parameters
    scp_params  = {
        "tr_radius0":            100.,
        "omega0":                100.,
        "omegamax":              1.0e10,
        "epsilon":               1.0e-6,
        "rho0":                  0.4,
        "rho1":                  1.5, 
        "beta_succ":             2.,
        "beta_fail":             0.5,
        "gamma_fail":            5.,
        "convergence_threshold": 1e-1,
        "NB_SCP_iter_max":       30
    }


    def __init__(self):
        print('[freeflyer::__init__] Initializing freeflyer.')

        # get symbolic dynamics
        (self.f_dt, self.A_dt, self.B_dt,
            self.f_A_dx, self.f_A_du, 
            self.f_B_dx, self.f_B_du)  = self.get_equations_dt()

        # cylindrical obstacles [(x,y),r]
        self.obstacles = [
            [[0.0,0.175], 0.15],
            [[-0.2,-0.1], 0.08],
            [[-0.35,0.1], 0.02],
        ]
        print('[freeflyer::__init__] nb of obstacles=',len(self.obstacles))

        self.poly_obstacles = []
        center, width = np.array([0.1,-0.15,0.]), 2*0.01*np.array([1.,1.,1e-4])
        self.poly_obstacles.append(PolyObs(center,width))


    def get_equations_dt(self):
        n_x, n_u = self.n_x, self.n_u

        m = self.mass
        J = self.J

        f = sp.zeros(6, 1)

        x = sp.Matrix(sp.symbols('x y theta xd yd thetad', real=True))
        u = sp.Matrix(sp.symbols('Fx Fy M', real=True))
        r, theta, v, thetadot = x[0:2,0], x[2,0], x[3:5,0], x[5,0]
        F, M                  = u[0:2,0], u[2,0]

        # No center of mass offset, Control of forces in World frame
        # f[0, 0] = x[3, 0]
        # f[1, 0] = x[4, 0]
        # f[2, 0] = x[5, 0]
        # f[3, 0] = (1./m) * u[0, 0]
        # f[4, 0] = (1./m) * u[1, 0]
        # f[5, 0] = (1./J) * u[2, 0]


        # Center of mass offset, Control of forces in Body frame
        p0       = np.array([0.05,0.05])
        p0_sympy = sp.Matrix([[p0[0]],[p0[1]]])
        R        = sp.Matrix([[sp.cos(theta),-sp.sin(theta)],[sp.sin(theta),sp.cos(theta)]])

        f[0, 0]     = x[3, 0]       # xdot
        f[1, 0]     = x[4, 0]       # ydot
        f[2, 0]     = x[5, 0]       # thetadot
        thetadotdot = (1. / J) * M
        f[3:5, 0]   = (1. / m) * R * (F - thetadotdot*sp.Matrix([[-p0[1]],[p0[0]]]) + thetadot**2*p0_sympy)
        f[5,   0]   = thetadotdot

        # pass to discrete time
        f = x + self.dt * f

        f = sp.simplify(f)
        A = sp.simplify(f.jacobian(x))
        B = sp.simplify(f.jacobian(u))

        A_col = A.reshape(n_x*n_x, 1)
        B_col = B.reshape(n_x*n_u, 1)

        A_dx = sp.simplify(A_col.jacobian(x))
        A_du = sp.simplify(A_col.jacobian(u))
        B_dx = sp.simplify(B_col.jacobian(x))
        B_du = sp.simplify(B_col.jacobian(u))

        f_func = sp.lambdify((x, u), f,    'numpy')
        A_func = sp.lambdify((x, u), A,    'numpy')
        B_func = sp.lambdify((x, u), B,    'numpy')
        A_dx_f = sp.lambdify((x, u), A_dx, 'numpy')
        A_du_f = sp.lambdify((x, u), A_du, 'numpy')
        B_dx_f = sp.lambdify((x, u), B_dx, 'numpy')
        B_du_f = sp.lambdify((x, u), B_du, 'numpy')

        return (f_func, A_func, B_func,
                A_dx_f, A_du_f, B_dx_f, B_du_f)

    def get_dynamics(self, x_k, u_k):
        """ 
            In discrete time, for one timestep k. 
            f() denotes dynamics:   x_{k+1} = f(x_k, u_k)

            Returns f(x_k, u_k) and its Jacobians (df/dx, df/du) 
            linearized around a state-control tuple (x_k, u_k)          
        """
        return self.f_dt(x_k, u_k), self.A_dt(x_k, u_k), self.B_dt(x_k, u_k)

    def get_dynamics_2nd_derivatives(self, x_k, u_k):
        """ 
            Returns 2nd-order Jacobians (df/dx^2, df/dxdu, df/du^2, df/dudx) 
            linearized around a state-control tuple (x_k, u_k)       
        """
        A_dx_k, A_du_k = self.f_A_dx(x_k, u_k), self.f_A_du(x_k, u_k)
        B_dx_k, B_du_k = self.f_B_dx(x_k, u_k), self.f_B_du(x_k, u_k)
        return A_dx_k, A_du_k, B_dx_k, B_du_k


    def compute_dynamics(self, X, U):
        """
            In discrete time, for all trajectory, s.t.

            x_{k+1} = f(x_k, u_k)  =>  linearized:
            x_{k+1} ~ f(xj_k,uj_k) + A(xj_k,uj_k)*(x_k-xj_k) + B(xj_k,uj_k)*(u_k-uj_k)

            Inputs:  - X     - states at each time  [n_x, N] (np.array)
                     - U     - controls ...         [n_u, N] (np.array)
                                (around which dynamics are linearized)
            Outputs: - f_all - vector of dynamics f(x_k,u_k) : [n_x,     N-1]
                     - A_all - vector of jacobians df/dx     : [n_x*n_x, N-1]
                     - B_all - vector of jacobians df/du     : [n_x*n_u, N-1]
        """
        N = X.shape[1]

        f_all = np.zeros([self.n_x         , N-1])
        A_all = np.zeros([self.n_x*self.n_x, N-1])
        B_all = np.zeros([self.n_x*self.n_u, N-1])

        for k in range(N-1):
            x_k = X[:,k]
            u_k = U[:,k]

            f_dyn_k, A_dyn_k, B_dyn_k = self.get_dynamics(x_k, u_k)

            f_all[:,k] = np.squeeze(f_dyn_k)
            A_all[:,k] = (A_dyn_k).flatten(order='F')
            B_all[:,k] = (B_dyn_k).flatten(order='F')

        return f_all, A_all, B_all

    def propagate_variances(self, X, U, A_all, B_all):
        """
        Outputs: - Vars     - Variances along trajectory 
                                    [n_x,n_x, N]
                 - Vars_dxu - Gradients of vars. along traj 
                                [n_x,n_x, N,n_xu, N]
                                where last dimension is time,
        """
        if self.B_feedback:
            raise NotImplementedError('Variance with feedback not implemented.')

        n_x, n_u = self.n_x, self.n_u
        Sig_w    = self.Sig_w

        N = X.shape[1]

        Vars     = np.zeros([n_x,n_x,              N])
        Vars_dxu = np.zeros([n_x,n_x, N, n_x+n_u,  N]) # [xdim^2, dxu_k, k of Sigma_{k}] 

        for k in range(N-1):
            x_k,   u_k       = X[:,k], U[:,k]
            Sig_k, Sig_dxu_k = Vars[:,:,k], Vars_dxu[:,:,:(k-1),:,k]

            # _, A_k, B_k                      = self.get_dynamics(x_k, u_k)
            A_k = np.reshape(A_all[:, k], (n_x, n_x), order='F')
            B_k = np.reshape(B_all[:, k], (n_x, n_u), order='F')
            (A_dx_k, A_du_k, B_dx_k, B_du_k) = self.get_dynamics_2nd_derivatives(x_k, u_k)

            A_dx_k = np.reshape(A_dx_k, (n_x,n_x,n_x))
            A_du_k = np.reshape(A_du_k, (n_x,n_x,n_u))
            B_dx_k = np.reshape(B_dx_k, (n_x,n_u,n_x))
            B_du_k = np.reshape(B_du_k, (n_x,n_u,n_u))
            A_dxu_k = np.concatenate((A_dx_k,A_du_k), axis=2)

            id_xk  = k*(n_x+n_u)
            id_xkn = k*(n_x+n_u) + (n_x+n_u)

            # next variance
            Sig_next = A_k@Sig_k@(A_k.T) + Sig_w
            # derivatives w.r.t. previous states and controls
            Sig_dxup_next = np.tensordot(A_k, 
                                np.tensordot(A_k,Sig_dxu_k, (1,0)), (1,1))
            Sig_dxuk_next = 2.*(A_k@Sig_k@np.swapaxes(A_dxu_k,1,0))

            Vars[:,:, k+1] = Sig_next

            Vars_dxu[:,:, :(k-1),:, k+1] = Sig_dxup_next
            Vars_dxu[:,:,      k,:, k+1] = Sig_dxuk_next

        return Vars, Vars_dxu


    def initialize_trajectory(self, N):
        """
            Straight-line initialization of the trajectory
        """
        X = np.empty(shape=[self.n_x, N])
        U = np.empty(shape=[self.n_u, N-1])

        for k in range(N):
            alpha1 = ( (N-1) - k ) / (N-1)
            alpha2 =       k       / (N-1)
            X[:, k] = self.x_init * alpha1 + self.x_final * alpha2 + 1e-4

        # Avoids zeros when linearizing some functions (dynamics), 
        # which could make the system uncontrollable => unfeasibility.
        # Note that this is not always necessary, and only for certain systems.
        U[:, :] = 1e-5

        return X, U


    def state_input_constraints_convexified(self, X_j, U_j, 
                                                  B_uncertainty=True,
                                                  Sigmas=None, Sigmas_dxu=None):
        """ 
        Inputs: - X_j         : state around which the constraint is linearized [n_x, N]
                - U_j         : control ...                                     [n_u, N-1]
                - Sigma_k     : Variances [xdim,xdim, N] along trajectory
                - Sigma_dxu_k : Derivative of Variances [n_x,n_x, N,n_xu, N],
        Outpus: Coeficients s.t. l <= A * [X,U] <= u
                - A : [n, N,n_xu]
                - l : (n,)
                - u : (n,)  with n : number of ineqalities = N*n_xu
        """  
        n_x, n_u, N                = self.n_x, self.n_u, Sigmas.shape[2]
        x_min, x_max, u_min, u_max = self.x_min, self.x_max, self.u_min, self.u_max

        XUj = np.concatenate((X_j, np.concatenate((U_j,np.zeros((n_u,1))), axis=1)),
                             axis=0)   # (n_xu, N)
        A = np.zeros([N*n_x+(N-1)*n_u, N, n_x+n_u])
        l = np.zeros([N*n_x+(N-1)*n_u])
        u = np.zeros([N*n_x+(N-1)*n_u])

        # deterministic part
        for k in range(N):
            for i in range(n_x):
                idx = k*(n_x+n_u) + i

                A[idx, k, i] = 1.
                l[idx] = x_min[i]
                u[idx] = x_max[i]

            for i in range(n_u):
                idx = k*(n_x+n_u) + n_x + i

                if k<U_j.shape[1]:
                    A[idx, k, n_x+i] = 1.
                    l[idx] = u_min[i]
                    u[idx] = u_max[i]

        # with uncertainy
        if B_uncertainty:
            if self.B_feedback:
                raise NotImplementedError('Variance with feedback not implemented.')

            delta_x = (1-self.prob)/(2.*n_x) # min and max constraints
            Phi_x   = p_th_quantile_cdf_normal(1-delta_x)

            for k in range(N):
                Sk, Sk_dxu = Sigmas[:,:,k], Sigmas_dxu[:,:,:,:,k]

                if (Sk.sum()>1e-6):
                    for i in range(n_x):
                        idx = k*(n_x+n_u) + i

                        aSa     = np.sqrt(Sk[i,i])
                        aSa_dxu = Sk_dxu[i,i,:,:]

                        A[idx, :,:] += Phi_x * ( 1./(2.*aSa) ) * aSa_dxu

                        asadxu_sum = np.einsum('nd,dn->',aSa_dxu,XUj)

                        l[idx] += Phi_x * ( aSa + ( 1./(2.*aSa) )*asadxu_sum )
                        u[idx] += Phi_x * (-aSa + ( 1./(2.*aSa) )*asadxu_sum )

        return A, l, u

    def obs_avoidance_constraint_convexified(self, X_j, U_j, obs_i, k,
                                B_uncertainty=True, Sigma_k=None, Sigma_dxu_k=None,
                                obs_type='sphere'):
        """ 
            Returns convexified obstacle avoidance chance constraints coefficients.

            Inputs: - X_j         : state around which the constraint is linearized [n_x, N]
                    - U_j         : control ...                                     [n_u, N-1]
                    - obs_i       : idx of obstacle
                    - Sigma_k     : Variance [xdim,xdim] at which obs constraint is evaluated
                    - Sigma_dxu_k : Derivative of Variance [n_x,n_x, N,n_xu],
                    - obs_type    : Type of obstacles, can be 'sphere' or 'poly'
            Outpus: Coeficients s.t. A * [X,U] <= b
                    - A : [N, n_xu]
                    - b : scalar

            Returns the constraints coefficients of the i-th obstacle 
            constraint g_i(x_k) <= 0 linearized at the state x_kj = X_j[:,k]
                s.t.            A * x_k <= b

                                  dist > (bot_radius+obs_radius) 
                    => ||x_k-obs_pos|| > (bot_radius+obs_radius)
                linearized =>
                    dist_prev + n_prev*(x_k-x_p) > (bot_radius+obs_radius)
                    n_prev*x_k  > -dist_prev + n_prev*x_p + (bot_radius+obs_radius)
                  -(n_prev*x_k) <  dist_prev - n_prev*x_p - (bot_radius+obs_radius))
        """  
        assert(X_j.ndim==2 and U_j.ndim==2 and k<=X_j.shape[1] and k<=U_j.shape[1])
        if obs_type == 'sphere':
            assert(obs_i>=0 and obs_i<len(self.obstacles))

        n_p, n_x, n_u, N = 2, self.n_x, self.n_u, X_j.shape[1]
        x_p              = X_j[:n_p, k]

        if obs_type=='sphere':
            obs = self.obstacles[obs_i]
            pos, radius = obs[0][0:n_p], obs[1]
                
            dist_prev = np.linalg.norm(x_p-pos,2)
            n_prev    = (x_p-pos) / dist_prev       # (2,)

            # deterministic part
            b = (dist_prev-radius) - n_prev@x_p - self.robot_radius

        elif obs_type=='poly':
            obs = self.poly_obstacles[obs_i]

            pos3d          = np.array([x_p[0], x_p[1], 0.])
            dist_prev, pos = signed_distance_with_closest_point_on_surface(pos3d, obs)
            pos            = pos[:n_p]

            # n_prev = (x_p-pos) / dist_prev       # (2,)
            n_prev = (x_p-obs.c[:n_p]) / np.linalg.norm((x_p-obs.c[:n_p]),2)       # (2,)

            # deterministic part
            b = dist_prev - n_prev@x_p - (self.robot_radius)

        else:
            raise NotImplementedError('Unknown obstacle type.')

        # deterministic part
        A         = np.zeros([N, n_x+n_u])
        A[k,:n_p] = -n_prev

        # with uncertainy
        if B_uncertainty:
            Phi   = self.p_quant_chi_sqrt

            S, S_dxu = Sigma_k[:n_p,:n_p], Sigma_dxu_k[:n_p,:n_p,:,:]
            if (S.sum()>1e-6):
                n_dxkj = np.eye(n_p)
                for i in range(n_p):
                    n_dxkj[i,:] += -(x_p[i]-pos[i])*(x_p-pos) / (dist_prev**2)
                n_dxkj /= dist_prev

                n_S_n      = np.sqrt(n_prev.T@S@n_prev)
                nSndxu     = np.einsum('x,y,xynd->nd', n_prev,n_prev,S_dxu) # (N,n_xu)

                n_S_ndx = n_prev.T@S@n_dxkj
                A[k,:n_p] += Phi * ( 1./(2.*n_S_n) ) * (2*n_S_ndx)
                A[:,:]    += Phi * ( 1./(2.*n_S_n) ) * nSndxu

                nSndxu_sum = (nSndxu[:,:n_x].flatten())      @ ((X_j.T).flatten()) + (
                             (nSndxu[:(N-1),n_x:].flatten()) @ ((U_j.T).flatten())   )
                b += Phi * (- n_S_n
                            + ( 1./(2.*n_S_n) ) * (
                                    (2*n_S_ndx) @ x_p
                                    + nSndxu_sum  )
                           )

        return A, b