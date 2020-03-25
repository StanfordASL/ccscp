import sys
sys.path.append('../src/utils/')

import numpy as np
import sympy as sp

from stats import p_th_quantile_chi_squared, p_th_quantile_cdf_normal

# Obstacles
from polygonal_obstacles import *
from polygonal_obstacles import PolygonalObstacle as PolyObs
from ISS import get_ISS_zones


class Model:
    # number of dimensions
    n_x      = 13
    n_u      = 6
    n_params = 4 # number of parameters (mass, diag inertias)

    # robot constants 
    robot_radius = 0.05
    mass         = 7.2
    J_norm       = 0.07
    J            = J_norm*np.eye(3)
    Jinv         = np.linalg.inv(J)
    mJ_var       = np.diag([3.2**2, 0.015**2, 0.015**2, 0.015**2]) # variances of parameters

    hard_limit_vel   = 0.4          # m/s
    hard_limit_accel = 0.1          # m/s^2
    hard_limit_omega = 45*3.14/180. # rad/s
    hard_limit_alpha = 50*3.14/180. # rad/s^2

    # constraints / limits
    x_max = np.array([100.,100.,100.,    hard_limit_vel,hard_limit_vel,hard_limit_vel,   100.,100.,100.,100.,   hard_limit_omega,hard_limit_omega,hard_limit_omega])
    u_max = np.array([mass*hard_limit_accel,mass*hard_limit_accel,mass*hard_limit_accel,  J_norm*hard_limit_alpha,J_norm*hard_limit_alpha,J_norm*hard_limit_alpha])
    x_min = np.negative(x_max)
    u_min = np.negative(u_max)

    # problem 
    s13     = np.sqrt(1./3.)
    x_init  = np.array([ 9.2,0.0,5.0,  1e-4,1e-4,1e-4,  s13,0.,s13,s13,     0,0,0])  + 1e-5
    x_final = np.array([11.3,6.0,4.5,  1e-4,1e-4,1e-4,  -0.5,0.5,-0.5,0.5,  0,0,0])  + 1e-4

    N        = 38               # nb discretization steps
    tf_guess = 90.              # s
    dt       = tf_guess / (N-1) # discretization time

    # Uncertainty / chance constraints
    Sig_w = np.diag([1e-7,1e-7,1e-7,3e-6,3e-6,3e-6, 1e-7,1e-7,1e-7,1e-7, 1e-7,1e-7,1e-7])
    prob  = 0.90                # probability threshold for chance constraints
    n_p   = 3                   # number of positional dimensions for obs. avoid.
    p_quant_chi_sqrt = np.sqrt(p_th_quantile_chi_squared(0.9, n_p))

    # OCP quadratic cost matrices
    quadratic_cost_matrix_controls = 10.*np.eye(n_u)
    quadratic_cost_matrix_state    = np.zeros((n_x,n_x))

    # Feedback controller parameters
    B_feedback = True
    Q_lqr      = 1e-1*np.eye(n_x)
    R_lqr      = np.diag([20.,20.,20., 5e2,5e2,5e2])
    K_fbs      = np.zeros((N, n_u, n_x))

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
        "NB_SCP_iter_max":       20
    }

    def __init__(self, mJ_true=np.array([None])):
        print('[astrobee.py::__init__] Initializing Astrobee.')
        
        # true dynamics
        if mJ_true.any()!=None:
            self.mass = mJ_true[0]
            self.J    = np.diag(mJ_true[1:])
            self.Jinv = np.linalg.inv(self.J)

        # get symbolic dynamics
        (self.f_dt, self.A_dt, self.B_dt,
            self.f_A_dx, self.f_A_du, 
            self.f_B_dx, self.f_B_du,
            self.f_dw, 
            self.f_w_dx, self.f_w_du)  = self.get_equations_dt()

        # spherical obstacles [(x,y,z),r]
        self.obstacles = [
            [[11.3,3.8,4.8], 0.3],
            [[10.5,5.5,5.5], 0.4],
        ]

        print('Initializing the ISS.')
        keepin_zones, keepout_zones = get_ISS_zones()
        self.poly_obstacles = keepout_zones
        # additional obstacles
        center, width = np.array([10.8,0.,5.]), 0.85*np.ones(3)
        self.poly_obstacles.append(PolyObs(center,width))
        center, width = np.array([11.2,1.75,4.85]), np.array([0.5,0.6,0.65])
        self.poly_obstacles.append(PolyObs(center,width))

        print('nb spherical obs:', len(self.obstacles))
        print('nb polygonal obs:', len(self.poly_obstacles))
        # --------------------


    def get_equations_dt(self):
        n_x, n_u, n_p = self.n_x, self.n_u, self.n_params

        f = sp.zeros(self.n_x, 1)
        x = sp.Matrix(sp.symbols('x y z vx vy vz qw qx qy qz wx wy wz', real=True))
        u = sp.Matrix(sp.symbols('Fx Fy Fz Mx My Mz', real=True))
        p = sp.Matrix(sp.symbols('m, Ix, Iy, Iz', real=True))

        r, v, w         = x[0:3,0], x[3:6,0], x[10:13,0]
        qw, qx, qy, qz  = x[6:10,0]
        wx, wy, wz      = x[10:13,0]
        F, M            = u[0:3,0], u[3:6,0]
        m, Ix, Iy, Iz   = p[0], p[1], p[2], p[3]

        J    = sp.diag(Ix,Iy,Iz)
        Jinv = sp.diag(1./Ix,1./Iy,1./Iz)

        f[0] = x[3, 0]
        f[1] = x[4, 0]
        f[2] = x[5, 0]
        f[3] = (1./m) * u[0, 0]
        f[4] = (1./m) * u[1, 0]
        f[5] = (1./m) * u[2, 0]   
        f[6] = 1/2*(-wx*qx - wy*qy - wz*qz)
        f[7] = 1/2*( wx*qw - wz*qy + wy*qz)
        f[8] = 1/2*( wy*qw + wz*qx - wx*qz)
        f[9] = 1/2*( wz*qw - wy*qx + wx*qy)
        f[10:13,0] = Jinv@(M[:] - np.cross(w[:],(J@w)[:]))

        # pass to discrete time
        f = x + self.dt * f

        f  = sp.simplify(f)
        A  = sp.simplify(f.jacobian(x))
        B  = sp.simplify(f.jacobian(u))
        fw = sp.simplify(f.jacobian(p))

        A_col =  A.reshape(n_x*n_x, 1)
        B_col =  B.reshape(n_x*n_u, 1)
        fwcol = fw.reshape(n_x*n_p, 1)

        A_dx = sp.simplify(A_col.jacobian(x))
        A_du = sp.simplify(A_col.jacobian(u))
        B_dx = sp.simplify(B_col.jacobian(x))
        B_du = sp.simplify(B_col.jacobian(u))
        fwdx = sp.simplify(fwcol.jacobian(x))
        fwdu = sp.simplify(fwcol.jacobian(u))

        # replace with nominal parameters
        f    = sp.simplify(f.subs(   [(m,self.mass), (Ix,self.J[0,0]), (Iy,self.J[1,1]), (Iz,self.J[2,2])]) )
        A    = sp.simplify(A.subs(   [(m,self.mass), (Ix,self.J[0,0]), (Iy,self.J[1,1]), (Iz,self.J[2,2])]) )
        B    = sp.simplify(B.subs(   [(m,self.mass), (Ix,self.J[0,0]), (Iy,self.J[1,1]), (Iz,self.J[2,2])]) )
        fw   = sp.simplify(fw.subs(  [(m,self.mass), (Ix,self.J[0,0]), (Iy,self.J[1,1]), (Iz,self.J[2,2])]) )
        A_dx = sp.simplify(A_dx.subs([(m,self.mass), (Ix,self.J[0,0]), (Iy,self.J[1,1]), (Iz,self.J[2,2])]) )
        A_du = sp.simplify(A_du.subs([(m,self.mass), (Ix,self.J[0,0]), (Iy,self.J[1,1]), (Iz,self.J[2,2])]) )
        B_dx = sp.simplify(B_dx.subs([(m,self.mass), (Ix,self.J[0,0]), (Iy,self.J[1,1]), (Iz,self.J[2,2])]) )
        B_du = sp.simplify(B_du.subs([(m,self.mass), (Ix,self.J[0,0]), (Iy,self.J[1,1]), (Iz,self.J[2,2])]) )
        fwdx = sp.simplify(fwdx.subs([(m,self.mass), (Ix,self.J[0,0]), (Iy,self.J[1,1]), (Iz,self.J[2,2])]) )
        fwdu = sp.simplify(fwdu.subs([(m,self.mass), (Ix,self.J[0,0]), (Iy,self.J[1,1]), (Iz,self.J[2,2])]) )

        f_func = sp.lambdify((x, u), f,  'numpy')
        A_func = sp.lambdify((x, u), A,  'numpy')
        B_func = sp.lambdify((x, u), B,  'numpy')
        fwfunc = sp.lambdify((x, u), fw, 'numpy')

        A_dx_f = sp.lambdify((x, u), A_dx, 'numpy')
        A_du_f = sp.lambdify((x, u), A_du, 'numpy')
        B_dx_f = sp.lambdify((x, u), B_dx, 'numpy')
        B_du_f = sp.lambdify((x, u), B_du, 'numpy')

        fwdx_f = sp.lambdify((x, u), fwdx, 'numpy')
        fwdu_f = sp.lambdify((x, u), fwdu, 'numpy')

        return (f_func, A_func, B_func,
                A_dx_f, A_du_f, B_dx_f, B_du_f,
                fwfunc, fwdx_f, fwdu_f)

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

    def get_dynamics_params_grads(self, x_k, u_k):
        """ 
            Returns 1st & 2nd order Jacobians w.r.t. paramaters
            (df/dparam, df/dparamdx, df/dparamdu), linearized 
            around a state-control tuple (x_k, u_k)          
        """
        f_dw_k, f_w_dx_k, f_w_du_k = self.f_dw(x_k, u_k), self.f_w_dx(x_k, u_k), self.f_w_du(x_k, u_k)

        return f_dw_k, f_w_dx_k, f_w_du_k

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

    # Linearized LQR controller
    def Riccati_equation(self, A, B, Qcost, Rcost, P):
        AtP           = A.T@P
        AtPA          = AtP@A
        AtPB          = AtP@B
        RplusBtPB_inv = np.linalg.inv(Rcost+B.T@P@B)
        return (Qcost + AtPA - AtPB @ RplusBtPB_inv @ (AtPB.T))
    def compute_lqr_feedback_gain(self, A, B, Qcost, Rcost, niter=5):
        P = Qcost
        for i in range(niter):
            P = self.Riccati_equation(A, B, Qcost, Rcost, P)
        Klqr = -np.linalg.inv(Rcost + B.T @ P @ B) @ (B.T @ P @ A )
        return Klqr
    def compute_feedback_gains(self, A_all, B_all):
        Qcost, Rcost = self.Q_lqr, self.R_lqr
        n_x, n_u, N = self.n_x, self.n_u, A_all.shape[1]

        for k in range(N):
            Alin_k = np.reshape(A_all[:, k], (n_x, n_x), order='F')
            Blin_k = np.reshape(B_all[:, k], (n_x, n_u), order='F')
            self.K_fbs[k,:,:] = self.compute_lqr_feedback_gain(Alin_k, Blin_k, 
                                                               Qcost,  Rcost)
        return self.K_fbs


    def propagate_variances(self, X, U, A_all, B_all):
        """
        Outputs: - Vars     - Variances along trajectory 
                                    [n_x,n_x, N]
                 - Vars_dxu - Gradients of vars. along traj 
                                [n_x,n_x, N,n_xu, N]
                                where last dimension is time
        """
        if self.B_feedback:
            K_fbs = self.compute_feedback_gains(A_all, B_all)

        n_x, n_u, n_params = self.n_x, self.n_u, self.n_params
        Sig_w, mJ_var      = self.Sig_w, self.mJ_var

        N = X.shape[1]

        Vars     = np.zeros([n_x,n_x,              N])
        Vars_dxu = np.zeros([n_x,n_x, N, n_x+n_u,  N]) # [xdim^2, dxu_k, k of Sigma_{k}] 

        for k in range(N-1):
            x_k,   u_k       = X[:,k], U[:,k]
            Sig_k, Sig_dxu_k = Vars[:,:,k], Vars_dxu[:,:,:(k-1),:,k]

            A_k = np.reshape(A_all[:, k], (n_x, n_x), order='F')
            B_k = np.reshape(B_all[:, k], (n_x, n_u), order='F')
            (A_dx_k, A_du_k, B_dx_k, B_du_k) = self.get_dynamics_2nd_derivatives(x_k, u_k)
            f_dw_k, f_w_dx_k, f_w_du_k       = self.get_dynamics_params_grads(x_k, u_k)

            A_dx_k     = np.reshape(A_dx_k, (n_x,n_x,n_x))
            A_du_k     = np.reshape(A_du_k, (n_x,n_x,n_u))
            B_dx_k     = np.reshape(B_dx_k, (n_x,n_u,n_x))
            B_du_k     = np.reshape(B_du_k, (n_x,n_u,n_u))
            A_dxu_k    = np.concatenate((A_dx_k,A_du_k), axis=2)
            f_w_dx_k   = np.reshape(f_w_dx_k, (n_x,n_params,n_x))
            f_w_du_k   = np.reshape(f_w_du_k, (n_x,n_params,n_u))
            f_dw_dxu_k = np.concatenate((f_w_dx_k,f_w_du_k), axis=2)

            if not(self.B_feedback):
                Sig_next = A_k@Sig_k@(A_k.T) + Sig_w + f_dw_k@mJ_var@(f_dw_k.T)
            else:
                K_fb = K_fbs[k,:,:]
                Sig_KT = Sig_k @ K_fb.T
                # Covariance matrix between state x and input u
                # Sig_xu_k = [sig         Sig_KT;
                #             Sig_KT.T  K_fb@Sig_KT]
                Sig_xu_k = np.concatenate((np.concatenate((Sig_k,    Sig_KT     ),axis=1),
                                           np.concatenate((Sig_KT.T, K_fb@Sig_KT),axis=1)) ,axis=0)
                AB_k = np.concatenate((A_k, B_k), axis=1)
                Sig_next = AB_k@Sig_xu_k@(AB_k.T) + Sig_w + f_dw_k@mJ_var@(f_dw_k.T)

            # derivatives w.r.t. previous states and controls
            Sig_dxup_next = np.tensordot(A_k, 
                                np.tensordot(A_k,Sig_dxu_k, (1,0)), (1,1))
            # derivatives w.r.t. current state and control 
            Sig_dxuk_next = 2.*(A_k@Sig_k@np.swapaxes(A_dxu_k,1,0) + 
                                np.tensordot(f_dw_k,
                                    np.tensordot(f_dw_dxu_k,mJ_var, (1,1)), (1,2))
                                )
            # Save results
            Vars[:,:, k+1]               = Sig_next
            Vars_dxu[:,:, :(k-1),:, k+1] = Sig_dxup_next
            Vars_dxu[:,:,      k,:, k+1] = Sig_dxuk_next

        return Vars, Vars_dxu


    def initialize_trajectory(self, N):
        """
            Straight-line initialization of the trajectory.

            Inputs:  - N : ()
            Outputs: - X : (n_x,  N ) - linearly interpolated from x_init to x_final
                     - U : (n_x, N-1) - zero controls
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
        U[:, :] = 1e-7

        return X, U


    def state_input_constraints_convexified(self, X_j, U_j, 
                                                  B_uncertainty=True,
                                                  Sigmas=None, Sigmas_dxu=None):
        """ 
            Returns convexified state-input chance constraints coefficients.

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
            delta_x = (1-self.prob)/(2.*n_x)  # min and max constraints
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

            if self.B_feedback:
                delta_u = (1-self.prob)/(2.*n_u)  # min and max constraints
                Phi_u   = p_th_quantile_cdf_normal(1-delta_u)

                for k in range(N):
                    K_fb       = self.K_fbs[k,:,:]
                    Sk, Sk_dxu = Sigmas[:,:,k], Sigmas_dxu[:,:,:,:,k]

                    Sk     = K_fb@Sk@(K_fb.T)
                    Sk_dxu = np.tensordot(K_fb, np.tensordot(K_fb,Sk_dxu, (1,0)), (1,1))

                    if (Sk.sum()>1e-6):
                        for i in range(n_u):
                            idx = k*(n_x+n_u) + n_x + i

                            aSa     = np.sqrt(Sk[i,i])
                            aSa_dxu = Sk_dxu[i,i,:,:]

                            A[idx, :,:] += Phi_u * ( 1./(2.*aSa) ) * aSa_dxu

                            asadxu_sum = np.einsum('nd,dn->',aSa_dxu,XUj)

                            l[idx] += Phi_u * ( aSa + ( 1./(2.*aSa) )*asadxu_sum )
                            u[idx] += Phi_u * (-aSa + ( 1./(2.*aSa) )*asadxu_sum )

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
        if obs_type == 'poly':
            assert(obs_i>=0 and obs_i<len(self.poly_obstacles))

        n_p, n_x, n_u, N = self.n_p, self.n_x, self.n_u, X_j.shape[1]
        x_p              = X_j[:n_p, k]

        if obs_type=='sphere':
            obs = self.obstacles[obs_i]
            pos, radius = obs[0][0:n_p], obs[1]

            dist_prev = np.linalg.norm(x_p-pos,2)
            n_prev    = (x_p-pos) / dist_prev

            # deterministic part
            b = dist_prev - n_prev@x_p - (radius+self.robot_radius) 

        elif obs_type=='poly':
            obs = self.poly_obstacles[obs_i]

            dist_prev, pos = signed_distance_with_closest_point_on_surface(x_p, obs)
            n_prev = (x_p-obs.c[:n_p]) / np.linalg.norm((x_p-obs.c[:n_p]),2)
            # if dist_prev>=0.:
            #     n_prev = (x_p-pos) / np.linalg.norm((x_p-pos),2)
            # else:
            #     n_prev = -(x_p-pos) / np.linalg.norm((x_p-pos),2)

            # deterministic part
            b = dist_prev - n_prev@x_p - (self.robot_radius) 
        else:
            raise NotImplementedError('Unknown obstacle type.')
            

        # deterministic part
        A         = np.zeros([N, n_x+n_u])
        A[k,:n_p] = -n_prev

        # with uncertainy
        if B_uncertainty:
            Phi = self.p_quant_chi_sqrt

            S, S_dxu = Sigma_k[:n_p,:n_p], Sigma_dxu_k[:n_p,:n_p,:,:]
            if (S.sum()>1e-6):
                n_dxkj = np.eye(n_p)
                for i in range(n_p):
                    n_dxkj[i,:] += -(x_p[i]-pos[i])*(x_p-pos) / (dist_prev**2)
                n_dxkj /= dist_prev

                n_S_n      = np.sqrt(n_prev.T@S@n_prev)
                nSndxu     = np.einsum('x,y,xynd->nd', n_prev,n_prev,S_dxu) # (N,n_xu)
                nSndxu     = np.squeeze((n_prev@S_dxu.T)@n_prev).T

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

    # ------------------------------------------------------------------
    # Other functions used for monte-carlo simualtions
    # 
    def check_obs_con(self, x_k, obs_i, obs_type='sphere'):
        n_p, n_x, n_u = 3, self.n_x, self.n_u
        x_p = x_k[:n_p]

        if obs_type=='sphere':
            obs = self.obstacles[obs_i]
            pos, radius = obs[0][0:n_p], obs[1]
            dist_prev = np.linalg.norm(x_p-pos,2) - (radius+self.robot_radius) 
            return dist_prev

        elif obs_type=='poly':
            obs = self.poly_obstacles[obs_i]
            return (signed_distance(x_p, obs)-self.robot_radius) 

        else:
            raise NotImplementedError('Unknown obstacle type.')
            

    def simulate(self, x_init, X_nom, U_nom, mJ_true, epsilons):
        n_x, n_u, N = self.n_x, self.n_u, self.N
        K_fbs = self.K_fbs

        X, U = np.zeros((n_x, N)), np.zeros((n_u, N-1))
        X[:,0] = x_init
        for k in range(N-1):
            U[:,k] = U_nom[:,k]
            if self.B_feedback:
                U[:,k] = U[:,k] + K_fbs[k,:,:]@(X[:,k]-X_nom[:,k])
            U[:,k] = np.clip(U[:,k], self.u_min, self.u_max)

            X[:,k+1] = self.f_dt_truemodel(X[:,k], U[:,k], mJ_true) + epsilons[:,k]
        return X, U


    def f_dt_truemodel(self, x, u, mJ_true):
        p = np.abs(mJ_true)
        r, v, w         = x[0:3], x[3:6], x[10:13]
        qw, qx, qy, qz  = x[6:10]
        wx, wy, wz      = x[10:13]
        F, M            = u[0:3], u[3:6]
        m, Ix, Iy, Iz   = p[0], p[1], p[2], p[3]

        f    = np.zeros(self.n_x)
        f[0] = x[3]
        f[1] = x[4]
        f[2] = x[5]
        f[3] = (1./m) * u[0]
        f[4] = (1./m) * u[1]
        f[5] = (1./m) * u[2]   
        f[6] = 1/2*(-wx*qx - wy*qy - wz*qz)
        f[7] = 1/2*( wx*qw - wz*qy + wy*qz)
        f[8] = 1/2*( wy*qw + wz*qx - wx*qz)
        f[9] = 1/2*( wz*qw - wy*qx + wx*qy)

        J    = np.diag([Ix,Iy,Iz])
        Jinv = np.diag([1./Ix,1./Iy,1./Iz])
        f[10:13] = Jinv@(M[:] - np.cross(w[:],(J@w)[:]))

        # pass to discrete time
        return (x + self.dt * f)