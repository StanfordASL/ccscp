import numpy as np
np.set_printoptions(precision=3, threshold=10**7)
import time
from copy import copy, deepcopy
from warnings import warn

from ccscp.src.utils.stats import p_th_quantile_chi_squared, p_th_quantile_cdf_normal

from ccscp.src.utils.polygonal_obstacles import *
from ccscp.src.utils.polygonal_obstacles import PolygonalObstacle as PolyObs

from ccscp.src.ccscp.cc_ocp import CCOCP

# CC-SCP Parameters
default_scp_config  = {
    "tr_radius0":            10.,
    "omega0":                1e5,
    "omegamax":              1.0e10,
    "epsilon":               1.0e-6,
    "rho0":                  0.4,
    "rho1":                  1.5, 
    "beta_succ":             2.,
    "beta_fail":             0.5,
    "gamma_fail":            5.,
    "convergence_threshold": 3e-1,#2e-1,
    "NB_SCP_iter_max":       5, #30, #20
}

# Parameters for uncertainty propagation
# Note: this is typically overwritten by controller UP config.
default_UP_config = {
    "shape":            "ellipsoidal", # "rectangular"
    "method":           "Gaussian",    # "MC-Bayesian" # "MC-frequentist"
    "reuse_ABgrads":   False,
    "reuse_presampled_dynamics":  False,
    "N_MC":           100,
    "return_grad":    True,
    "return_parts":   True,
    "safety_prob":    0.9,    # probability level [%] of chance constraints
    "B_feedback":     False,
}

class CCSCPModel:
  # wrapper of ALPaCA dynamics to provide functions for CCSCP
  def __init__(self, model, scp_config=default_scp_config,
                            UP_config=default_UP_config):
    """
    Inputs: - model: AdaptiveDynamics (in Models) which has functions:
                  - predict_mean
                  - predict_mean_linearized

    class must have functions:
        - @initialize_trajectory
        - @compute_dynamics
        - @obs_avoidance_constraint
        - @obs_avoidance_constraint_convexified
    """
    print('[CCSCPModel::__init__] Initializing model, using ', model)
    self.scp_config = scp_config
    self.UP_config  = UP_config

    self.model = model

    self.final_constraint_feasible = True

    # self.dt = model.dt
    # warn('ToDo: get dt from (lin_uncert/ALPaCA/NN) model, but it is not saved from dataset yet...')

    # number of positional dimensions for obs. avoid.
    self.nb_pos_dim = model.nb_pos_dim

    self.n_x = model.y_dim
    self.n_u = model.u_dim

  def reset(self):
    self.model.reset()

  def get_feedback_gains(self, X=np.zeros(0), U=np.zeros(0),
                               B_reuse_precomputed=True):
    if B_reuse_precomputed==False and not(X.ndim==2) and not(U.ndim==2):
      raise KeyError('[CCSCPModel::get_feedback_gains] no trajectory provided.')
    if not(self.UP_config["B_feedback"]):
      print('[CCSCPModel::get_feedback_gains] B_feedback = False,' + 
                                    'controller and model do not match!')
      return np.zeros((0,self.n_u,self.n_x))

    if B_reuse_precomputed:
      return self.model.K_fbs
    else:
      _, A_all, B_all = self.compute_dynamics(X,U)
      return self.model.compute_feedback_gains(A_all, B_all)

  def incorporate_transition(self, x, u, xp):
    self.model.incorporate_transition(x,u,xp)
  def get_uncertainty_params_norm(self):
    return self.model.get_uncertainty_params_norm()

  def initialize_trajectory(self, N, x_init, x_final,
                                  init_method='straightline'):
    """
      Inputs:  - N : (); - x_init : (n_x,); - x_final : (n_x,)
      Outputs: - X : (n_x,  N ) - linearly interpolated from x_init to x_final
               - U : (n_x, N-1) - zero controls
    """
    self.final_constraint_feasible = True

    X = np.empty(shape=[self.n_x, N])
    U = np.empty(shape=[self.n_u, N-1])

    if init_method=='straightline':
      print('[ccscp_model::initialize_traj] using straight-line initialization.')
      for k in range(N):
        alpha1 = ( (N-1) - k ) / (N-1)
        alpha2 =       k       / (N-1)
        X[:, k] = x_init * alpha1 + x_final * alpha2 + 1e-6

    elif init_method=='left':
      print('[ccscp_model::initialize_traj] using left initialization.')

      for k in range(N):
        alpha1 = ( (N-1) - k ) / (N-1)
        alpha2 =       k       / (N-1)
        X[:, k] = x_init * alpha1 + x_final * alpha2 + 1e-6

        x_intermediate = np.array([0.5,3])
        Nhalf = int(N/2)
        for k in range(Nhalf):
            alpha1 = ( (Nhalf-1) - k ) / (Nhalf-1)
            alpha2 =       k       / (Nhalf-1)
            X[:2, k] = x_init[:2] * alpha1 + x_intermediate * alpha2 + 1e-6
        for k in range(Nhalf,N):
            alpha1 = ( (N-1) - k ) / (Nhalf)
            alpha2 =       (k-Nhalf)       / (Nhalf)
            X[:2, k] = x_intermediate * alpha1 + x_final[:2] * alpha2 + 1e-6

    else:
      print('[ccscp_model::initialize_traj] unknown initialization method, using straight-line')
      for k in range(N):
        alpha1 = ( (N-1) - k ) / (N-1)
        alpha2 =       k       / (N-1)
        X[:, k] = x_init * alpha1 + x_final * alpha2 + 1e-6

    # avoids zeros when linearizing some functions, which could make
    # the system uncontrollable => no feasible solution
    U[:, :] = 1e-5 * np.random.rand(self.n_u, N-1)

    return X, U

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

    # single timestep: X \in \R^{xdim}, U \in \R^{udim}
    if (X.ndim==1 and X.size==self.n_x and U.ndim==1 and U.size==self.n_u):
      f_dyn_k          = self.model.predict_mean(X, U)
      A_dyn_k, B_dyn_k = self.model.predict_mean_linearized(X, U)
      return f_dyn_k, A_dyn_k, B_dyn_k

    # trajectory: X is [N, xdim], U is X is [N, udim]
    else:
      N = X.shape[1]

      f_all = np.zeros([self.n_x         , N-1])
      A_all = np.zeros([self.n_x*self.n_x, N-1])
      B_all = np.zeros([self.n_x*self.n_u, N-1])

      for k in range(N-1):
        x_k = X[:,k]
        u_k = U[:,k]

        f_dyn_k          = self.model.predict_mean(x_k, u_k)
        A_dyn_k, B_dyn_k = self.model.predict_mean_linearized(x_k, u_k)

        f_all[:,k] =  f_dyn_k
        A_all[:,k] = (A_dyn_k).flatten(order='F')
        B_all[:,k] = (B_dyn_k).flatten(order='F')

    return f_all, A_all, B_all

  def propagate_uncertainty(self, X, U, UP_args):
    """
    Inputs: X             : states at each time  (n_x, N)
            U             : controls ...         (n_u, N)
            UP_args  : Dict of options/arguments for uncertainty propagation (UP)
                  - shape         : "rectangular" / "ellipsoidal"
                  - method        : "Gaussian" (moments matching) / "MC-Bayesian" (monte-carlo)
                  - reuse_ABgrads : if True -> reuse A_all and B_all below
                  - A_all         : vector of jacobians df/dx (n_x*n_x, N-1)
                  - B_all         : vector of jacobians df/du (n_x*n_u, N-1)
                  - reuse_presampled_dynamics : if True, won't resample 
                                  uncertain parameters for monte-carlo
                  - N_MC          : nb of monte-carlo samples for MC set 
                                    computation
                  - return_grad   : if True, returns jacobians of confset shapes
                  - return_parts  : if True and using MC, 
                                      returns MC particles (N_MC, n_x, N)
    Outputs: UP_vals : Dict with output, depending on UP_pars:
              method == "Gaussian" 
                  - Vars : Variances along trajectory (n_x,n_x, N)
                if return_grad
                  - Vars_dxu : Jacobians of Vars (n_x,n_x, N,n_xu, N) (last dim is time)
              method == "MC-Bayesian"
                shape == "ellipsoidal"
                  - Qs : Ellipsoidal shapes along trajectory (n_x,n_x, N)
                  if return_grad
                    - Qs_dxu : Jacobians of Qs (n_x,n_x, N,n_xu, N)
                shape == "rectangular"
                  - TODO
    """
    # Copy model parameters, and update different ones with those that are given
    UP_pars = deepcopy(self.UP_config)
    UP_pars.update(UP_args)

    UP_vals = dict()

    # Dynamics jacobians always used (for variance prop, or feedback controller computation)
    if not(UP_pars["reuse_ABgrads"] and 
           "A_all" in UP_pars and "B_all" in UP_pars):
      _, UP_pars["A_all"], UP_pars["B_all"] = self.compute_dynamics(X, U)

    if UP_pars["method"] == "Gaussian":
      UP_vals["Vars"], Vars_dxu = self.model.propagate_variances(X, U, 
                                                  UP_pars["A_all"], UP_pars["B_all"], 
                                                  B_feedback=UP_pars["B_feedback"])
      if UP_pars["return_grad"]: UP_vals["Vars_dxu"] = Vars_dxu


    elif UP_pars["method"]=="MC-Bayesian" or UP_pars["method"]=="MC-frequentist":
      if UP_pars["method"]=="MC-Bayesian":
        QDs, QDs_dxu, UP_vals["parts"] = self.model.predict_credsets_monteCarlo(X, U, 
                                                    UP_pars["A_all"], UP_pars["B_all"], 
                                                    UP_pars["shape"],
                                                    N_MC=UP_pars["N_MC"], prob=UP_pars["safety_prob"],
                                                    B_feedback=UP_pars["B_feedback"])
      if UP_pars["method"]=="MC-frequentist":
        QDs, QDs_dxu, UP_vals["parts"] = self.model.predict_confsets_monteCarlo(X, U, 
                                                    UP_pars["A_all"], UP_pars["B_all"], 
                                                    UP_pars["shape"],
                                                    N_MC=UP_pars["N_MC"], prob=UP_pars["safety_prob"],
                                                    B_feedback=UP_pars["B_feedback"])


      if UP_pars["shape"] == "ellipsoidal":
        UP_vals["Qs"] = QDs
        if UP_pars["return_grad"]: UP_vals["Qs_dxu"] = QDs_dxu
      elif UP_pars["shape"] == "rectangular":
        UP_vals["Deltas"] = QDs
        if UP_pars["return_grad"]: UP_vals["Deltas_dxu"] = QDs_dxu
      else: 
        raise NotImplementedError('[ccscp_model::propagate_uncertainty] Unknown MC set shape.')

    else:
      raise NotImplementedError('[ccscp_model::propagate_uncertainty] ' + 
                                'Unknown UP method.')

    return UP_vals


  def get_objective_coeffs(self, means, actions, x_goal):
    """
    Returns the coefficients describing the exploration objective, such that
      cost = xN^T*QN*xN + sum_{k=0}^{N-1} { xk^T*Q*xk + uk^T*R*uk }
              + sum_{k=0}^{N-1} { costs[k] + costs_grads[k,:] * [xk;uk] }
    Inputs:  mus         : [xdim, T+1]
             actions     : [udim,  T ]
    Outputs: Q           : [xdim,xdim]
             QN          : [xdim,xdim]
             R           : [udim,udim]
             costs       : [T]
             costs_grads : [T, xdim+udim]
    """
    n_x, n_u, T = self.n_x, self.n_u, means.shape[1]

    Q, QN, R = self.model.get_quadratic_costs()
    qN       = np.zeros(n_x)

    costs       = np.zeros(T)
    costs_grads = np.zeros((T,n_x+n_u))

    return Q, QN, qN, R, costs, costs_grads

  def check_UP_dicts_for_constraints(self, UP_vals, UP_config):
    # performs checks before evaluating chance constraints
    if "B_feedback"  not in UP_config: raise KeyError('Feedback boolean missing.')
    if "safety_prob" not in UP_config: raise KeyError('Probability level (%) missing.')
    if UP_config["method"] == "Gaussian":
      if "Vars"      not in UP_vals:   raise KeyError('Variances missing.')
      if "Vars_dxu"  not in UP_vals:   raise KeyError('Variances Jacobians missing.')

    elif ((UP_config["method"]=="MC-Bayesian" or UP_config["method"]=="MC-frequentist") and 
           UP_config["shape"] == "ellipsoidal"):
        if "Qs"      not in UP_vals:   raise KeyError('Ellips. shape matrices missing.')
        if "Qs_dxu"  not in UP_vals:   raise KeyError('Ellipses Jacobians missing.')

    elif ((UP_config["method"]=="MC-Bayesian" or UP_config["method"]=="MC-frequentist") and 
           UP_config["shape"] == "rectangular"):
        if "Deltas"      not in UP_vals:   raise KeyError('Rectangles Deltas missing.')
        if "Deltas_dxu"  not in UP_vals:   raise KeyError('Deltas Jacobians missing.')

    else:
      raise NotImplementedError('[ccscp_model::check_UP_dicts_for_constraints] ' + 
                                ' Unknown Uncertainty Propagation method.')
    return True


  def get_final_set_constraint(self, Xset_final, UP_vals, UP_args):
    """ 
    Inputs: - X_final : ball [pos, rads] centered at pos=(xdim) and of radius rads=(xdim)
            - UP_vals : dict of vals. 
                If UP_args["method"] == "Gaussian"
                  > Vars :     Variances along trajectory (n_x,n_x, N)
                  > Vars_dxu : Gradients of vars (n_x,n_x, N,n_xu, N) (last dim is time)
                Else 
                  >  !!!!!!!!!! TODO !!!!!!!!!!
                  >  !!!!!!!!!! TODO !!!!!!!!!!
            - UP_args : dict of parameters to handle uncertainty propagation
    Returns (A, lineq, uineq) such that
              lineq <= A xN <= uineq
            for a final goal constraints, accounting for uncertainty
    """
    UP_pars = deepcopy(self.UP_config)
    UP_pars.update(UP_args)
    self.check_UP_dicts_for_constraints(UP_vals, UP_pars)


    xgoal, deltas_xgoal = Xset_final[0], Xset_final[1]

    # pad by uncertainty
    Deltas_N = np.zeros((self.n_x))
    if self.UP_config["method"] == "Gaussian":
      delta_xN = ((1.-self.UP_config["safety_prob"]) / (2.*self.model.n_x)) # min and max constraints
      Phi_xN   = p_th_quantile_cdf_normal(1.-delta_xN)
      Deltas_N = Phi_xN * np.sqrt(np.diag(UP_vals["Vars"][:,:,-1]))


    elif ((UP_pars["method"]=="MC-Bayesian" or UP_pars["method"]=="MC-frequentist") and 
           UP_pars["shape"] == "ellipsoidal"):
        Deltas_N = np.sqrt(np.diag(UP_vals["Qs"][:,:,-1]))

    elif ((UP_pars["method"]=="MC-Bayesian" or UP_pars["method"]=="MC-frequentist") and 
           UP_pars["shape"] == "rectangular"):
        Deltas_N = UP_vals["Deltas"][:,-1]

    else:
      raise NotImplementedError('[ccscp_model::get_final_set_constraint] ' + 
                                'Unknown UP method.')


    deltas_xgoal = deltas_xgoal - Deltas_N

    # if uncertainty too high, replace by equality.
    if (deltas_xgoal < 1e-7).any():

      self.final_constraint_feasible = False

      deltas_xgoal[deltas_xgoal < 1e-7] = 1e-6#0.

    Aineq = np.eye(self.n_x)
    lineq = xgoal - deltas_xgoal
    uineq = xgoal + deltas_xgoal

    return Aineq, lineq, uineq

  def get_final_set_constraint_convexified(self, Xset_final,
                                                 X_j, U_j, 
                                                 UP_vals, UP_args):
    """ 
    Inputs: - X_j     : state around which the constraint is linearized [n_x, N]
            - U_j     : control ...                                     [n_u, N-1]
            - UP_vals : dict of vals, see description in [self::get_final_set_constraint]
            - UP_args : dict of pars, see description in [self::get_final_set_constraint]
    Outpus: Coeficients s.t. l <= A * [X,U] <= u
            - A : [n, N,n_xu]
            - l : (n,)
            - u : (n,)  with n : number of ineqalities = N*n_xu
    """  
    UP_pars = deepcopy(self.UP_config)
    UP_pars.update(UP_args)
    self.check_UP_dicts_for_constraints(UP_vals, UP_pars)
    
    m, n_x, n_u, N      = self.model, self.n_x, self.n_u, X_j.shape[1]
    xgoal, deltas_xgoal = Xset_final[0], Xset_final[1]

    XUj = np.concatenate((X_j, np.concatenate((U_j,np.zeros((n_u,1))), axis=1)),
                         axis=0)   # (n_xu, N)

    A = np.zeros([2*n_x, N, n_x+n_u])
    l = -np.inf*np.ones(2*n_x)
    u = np.zeros(2*n_x)

    # deterministic part
    A[:n_x,-1,:n_x] =  np.eye(n_x)
    A[n_x:,-1,:n_x] = -np.eye(n_x)
    u[:n_x] =   xgoal + deltas_xgoal
    u[n_x:] = -(xgoal - deltas_xgoal)

    # with uncertainty
    if ((UP_pars["method"]=="MC-Bayesian" or UP_pars["method"]=="MC-frequentist") and 
         UP_pars["shape"] == "rectangular"):
      # similar code as in [compute_paddings_state_input_constraints_rectangular]
      Aki, bki, bki_dxuSum = np.zeros([n_x, N, n_x+n_u]), np.zeros(n_x), np.zeros(n_x)
      D, D_dxu = UP_vals["Deltas"][:,-1], UP_vals["Deltas_dxu"][:,:,:,-1]
      if (D.sum()>1e-6):
        Aki, bki, bki_dxuSum = self.compute_padding_dimwise_cc_rectangular_separateTerms(XUj[:n_x, -1], XUj, D, D_dxu)

    else:
      raise NotImplementedError('[ccscp_model::get_final_set_constraint_convexified] ' + 
                                ' Unknown Uncertainty Propagation method.')

    # if uncertainty too high, replace by equality.
    if ((   D   > deltas_xgoal ).any() ):# or 
        # ( bki + bki_dxuSum > deltas_xgoal ).any()):
      self.final_constraint_feasible = False
      # print('v_pad is too high') 

      A[:n_x,:,:] +=   Aki
      u[:n_x]      =  (xgoal+1e-2) - bki_dxuSum

      A[n_x:,:,:] +=  Aki
      u[n_x:]      = -(xgoal-1e-2) - bki_dxuSum

    else:
      self.final_constraint_feasible = True
      A[:n_x,:,:] +=   Aki
      u[:n_x]     += -(bki + bki_dxuSum)

      A[n_x:,:,:] +=  Aki
      u[n_x:]     += -(bki + bki_dxuSum)

    return A, l, u



  def state_input_constraints_convexified(self, X_j, U_j, 
                                                x_min, x_max, u_min, u_max,
                                                UP_vals, UP_args):
    """ 
    Inputs: - X_j     : state around which the constraint is linearized [n_x, N]
            - U_j     : control ...                                     [n_u, N-1]
            - UP_vals : dict of vals, see description in [self::get_final_set_constraint]
            - UP_args : dict of pars, see description in [self::get_final_set_constraint]
    Outpus: Coeficients s.t. l <= A * [X,U] <= u
            - A : [n, N,n_xu]
            - l : (n,)
            - u : (n,)  with n : number of ineqalities = N*n_xu
    """  
    UP_pars = deepcopy(self.UP_config)
    UP_pars.update(UP_args)
    self.check_UP_dicts_for_constraints(UP_vals, UP_pars)
    
    m           = self.model
    n_x, n_u, N = self.n_x, self.n_u, X_j.shape[1]

    XUj = np.concatenate((X_j, np.concatenate((U_j,np.zeros((n_u,1))), axis=1)),
                         axis=0)   # (n_xu, N)

    # A*z < u, first: xu_max, second: xu_min
    NCs = (N*n_x+(N-1)*n_u)
    A =  np.zeros(     [2*NCs, N, n_x+n_u])
    l = -np.inf*np.ones(2*NCs)
    u =  np.zeros(     [2*NCs])

    # deterministic part
    for k in range(N):
      for i in range(n_x):
        idx = k*(n_x+n_u) + i

        A[idx, k, i] = 1.
        u[idx]       = x_max[i]

        A[idx+NCs, k, i] = -1.
        u[idx+NCs]       = -x_min[i]

      for i in range(n_u):
        if k<U_j.shape[1]:
          idx = k*(n_x+n_u) + n_x + i

          A[idx,     k, n_x+i] = 1.
          u[idx]               = u_max[i]

          A[idx+NCs, k, n_x+i] = -1.
          u[idx+NCs]           = -u_min[i]

    # with uncertainty
    if UP_pars["method"] == "Gaussian":
      A_pad, v_pad = self.compute_paddings_state_input_constraints_Gaussian(XUj, UP_vals)

    elif ((UP_pars["method"]=="MC-Bayesian" or UP_pars["method"]=="MC-frequentist") and 
           UP_pars["shape"] == "ellipsoidal"):
      A_pad, v_pad = self.compute_paddings_state_input_constraints_ellipsoidal(XUj, UP_vals)

    elif ((UP_pars["method"]=="MC-Bayesian" or UP_pars["method"]=="MC-frequentist") and 
           UP_pars["shape"] == "rectangular"):
      A_pad, v_pad = self.compute_paddings_state_input_constraints_rectangular(XUj, UP_vals)

    else:
      raise NotImplementedError('[ccscp_model::state_input_constraints_convexified] ' + 
                                ' Unknown Uncertainty Propagation method.')

    # if (v_pad > (u-l)/2. ).any():
    #   print('state or input paddning is too high')
    # v_pad = np.clip(v_pad, v_pad, (u-l)/2.)

    A[:NCs,:,:] +=  A_pad
    u[:NCs]     +=  v_pad
    A[NCs:,:,:] +=  A_pad
    u[NCs:]     +=  v_pad

    return A, l, u

  def compute_paddings_state_input_constraints_Gaussian(self, XUj, UP_vals):
    """
    For the state-input constraints Pr(xmin < xk <= xmax) > prob, k=0,..N
                                    Pr(umin < uk <= umax) > prob, k=0,..N-1,
          computes the paddings due to the variance/chance-constraints, such that
      (xmin,umin) + v_pad <= (xk_mean,uk_mean) + A_pad @ XUj <= (xmax,umax) - v_pad 
    """
    n_x, n_u, N = self.n_x, self.n_u, UP_vals["Vars"].shape[-1]

    A_pad = np.zeros([N*n_x+(N-1)*n_u, N, n_x+n_u])
    v_pad = np.zeros([N*n_x+(N-1)*n_u])

    # compute quantiles
    delta_x = (1-self.UP_config["safety_prob"])/(2.*n_x)    # min and max constraints
    Phi_x   = p_th_quantile_cdf_normal(1-delta_x)
    if self.UP_config["B_feedback"]:
      delta_u = (1-self.UP_config["safety_prob"])/(2.*n_u)  # min and max constraints
      Phi_u   = p_th_quantile_cdf_normal(1-delta_u)

    for k in range(N):
      # State constraints
      Var, Var_dxu = UP_vals["Vars"][:,:,k], UP_vals["Vars_dxu"][:,:,:,:,k]

      if (Var.sum()>1e-6):
        idx_k, idx_kn = k*(n_x+n_u), k*(n_x+n_u) + n_x

        Aki, bki = self.compute_padding_dimwise_cc_Gaussian(XUj[:n_x, k], XUj, 
                                                            Var, Var_dxu, Phi_x)
        A_pad[idx_k:idx_kn, :,:] +=  Aki
        v_pad[idx_k:idx_kn]      += -bki

      # Control constraints
      if not(self.UP_config["B_feedback"]) or k==N-1:
        continue

      K_fb      = self.model.K_fbs[k,:,:]
      Var_u     = K_fb @ Var @ (K_fb.T)
      Var_u_dxu = np.tensordot(K_fb, np.tensordot(K_fb,Var_dxu, (1,0)), (1,1))

      if (Var_u.sum()>1e-6):
        idx_k, idx_kn = k*(n_x+n_u)+n_x, k*(n_x+n_u) + n_x+n_u

        Aki, bki = self.compute_padding_dimwise_cc_Gaussian(XUj[n_x:, k], 
                                                            XUj, Var_u, Var_u_dxu, Phi_u)
        A_pad[idx_k:idx_kn, :,:] +=  Aki
        v_pad[idx_k:idx_kn]      += -bki

    return A_pad, v_pad

  def compute_padding_dimwise_cc_Gaussian(self, xj, XUj, Var_x, Var_x_dXUj, Phi_xj):
    """
        Pr(xi <= 0) > prob <=> fi(XU) <= 0 , reformulated with Gaussian approximation
                                             with prob. characterized by Phi_x (for each dim)

      Inputs: - xj      : (n_d,)       - mean of x
              - XUj     : (n_x+n_u, N) - mean (state/controls) around which linearized
              - Var_x   : (n_d, n_d)   - variance of x, evaluated at (Xj,Uj)
              - Var_x_dXUj : (n_d,n_d, N,n_xu) - Jacobian of Var_x evaluated at (Xj,Uj)
              - Phi_x   : ()           - quantile, see ECC paper
      Output: - A : (n_d, N,n_xu) = d/dxu(fi)(Xj,Uj),                       i = 1,..n_d
              - b : (n_d)         = fi(Xj,Uj) - d/dxu(fi)(Xj,Uj) @ (Xj,Uj), i = 1,..n_d
    """
    n_xu, N = XUj.shape
    n_d     = xj.shape[0]
    A, b    = np.zeros((n_d, N,n_xu)), np.zeros(n_d)

    for i in range(n_d):
        aSa        = np.sqrt(Var_x[i,i])
        aSa_dxu    = Var_x_dXUj[i,i,:,:]
        asadxu_sum = np.einsum('nd,dn->',aSa_dxu,XUj)
        
        A[i,:,:] = Phi_xj * ( 1./(2.*aSa) ) * aSa_dxu
        b[i]     = Phi_xj * (aSa - ( 1./(2.*aSa) )*asadxu_sum )
    return A, b
  def compute_paddings_state_input_constraints_ellipsoidal(self, XUj, UP_vals):
    # see description of self::compute_paddings_state_input_constraints_Gaussian()
    n_x, n_u, N  = self.n_x, self.n_u, UP_vals["Qs"].shape[-1]
    A_pad, v_pad = np.zeros([N*n_x+(N-1)*n_u, N, n_x+n_u]), np.zeros([N*n_x+(N-1)*n_u])

    for k in range(N):
      # State constraints
      Q, Q_dxu = UP_vals["Qs"][:,:,k], UP_vals["Qs_dxu"][:,:,:,:,k]
      if (Q.sum()>1e-6):
        idx_k, idx_kn = k*(n_x+n_u), k*(n_x+n_u) + n_x
        Aki, bki = self.compute_padding_dimwise_cc_ellipsoidal(XUj[:n_x, k], XUj, Q, Q_dxu)
        A_pad[idx_k:idx_kn, :,:] +=  Aki
        v_pad[idx_k:idx_kn]      += -bki
      # Control constraints
      if not(self.UP_config["B_feedback"]) or k==N-1: continue
      K_fb    = self.model.K_fbs[k,:,:]
      Q_u     = K_fb @ Q @ (K_fb.T)
      Q_u_dxu = np.tensordot(K_fb, np.tensordot(K_fb,Q_dxu, (1,0)), (1,1))
      if (Q_u.sum()>1e-6):
        idx_k, idx_kn = k*(n_x+n_u)+n_x, k*(n_x+n_u) + n_x+n_u
        Aki, bki = self.compute_padding_dimwise_cc_ellipsoidal(XUj[n_x:, k], XUj, Q_u, Q_u_dxu)
        A_pad[idx_k:idx_kn, :,:] +=  Aki
        v_pad[idx_k:idx_kn]      += -bki
    return A_pad, v_pad
  def compute_paddings_state_input_constraints_rectangular(self, XUj, UP_vals):
    # see description of self::compute_paddings_state_input_constraints_Gaussian()
    n_x, n_u, N  = self.n_x, self.n_u, UP_vals["Deltas"].shape[-1]
    A_pad, v_pad = np.zeros([N*n_x+(N-1)*n_u, N, n_x+n_u]), np.zeros([N*n_x+(N-1)*n_u])

    for k in range(N):
      # State constraints
      D, D_dxu = UP_vals["Deltas"][:,k], UP_vals["Deltas_dxu"][:,:,:,k]
      if (D.sum()>1e-6):
        idx_k, idx_kn = k*(n_x+n_u), k*(n_x+n_u) + n_x
        Aki, bki = self.compute_padding_dimwise_cc_rectangular(XUj[:n_x, k], XUj, D, D_dxu)
        A_pad[idx_k:idx_kn, :,:] +=  Aki
        v_pad[idx_k:idx_kn]      += -bki
      # Control constraints
      if not(self.UP_config["B_feedback"]) or k==(N-1): 
        continue
      K_fb    = self.model.K_fbs[k,:,:]
      # minus sign : D_u would be negative! 
      D_u     = -K_fb @ D 
      D_u_dxu = -np.tensordot(K_fb, D_dxu, (1,0))
      if (np.linalg.norm(D_u.sum())>1e-6):
        idx_k, idx_kn = k*(n_x+n_u)+n_x, k*(n_x+n_u) + n_x+n_u
        Aki, bki = self.compute_padding_dimwise_cc_rectangular(XUj[n_x:, k], XUj, D_u, D_u_dxu)
        A_pad[idx_k:idx_kn, :,:] +=  Aki
        v_pad[idx_k:idx_kn]      += -bki
    return A_pad, v_pad
  def compute_padding_dimwise_cc_ellipsoidal(self, xj, XUj, Q, Q_dXUj):
    # see description of self::compute_padding_dimwise_cc_Gaussian()
    n_xu, N = XUj.shape
    n_d     = xj.shape[0]
    A, b    = np.zeros((n_d, N,n_xu)), np.zeros(n_d)
    for i in range(n_d):
        aQa        = np.sqrt(Q[i,i])
        aQa_dxu    = Q_dXUj[i,i,:,:]
        aQadxu_sum = np.einsum('nd,dn->',aQa_dxu,XUj)
        
        A[i,:,:] =  1./(2.*aQa) * aQa_dxu
        b[i]     = aQa - ( 1./(2.*aQa) )*aQadxu_sum 
    return A, b
  def compute_padding_dimwise_cc_rectangular(self, xj, XUj, Delta, Delta_dXUj):
    # see description of self::compute_padding_dimwise_cc_Gaussian()
    n_xu, N = XUj.shape
    n_d     = xj.shape[0]
    A, b    = np.zeros((n_d, N,n_xu)), np.zeros(n_d)
    for i in range(n_d):
        aDa, aDa_dxu = Delta[i], Delta_dXUj[i,:,:]
        aDadxu_sum   = np.einsum('nd,dn->',aDa_dxu,XUj)
        
        A[i,:,:] = aDa_dxu
        b[i]     = aDa - aDadxu_sum 
    return A, b
  def compute_padding_dimwise_cc_rectangular_separateTerms(self, xj, XUj, Delta, Delta_dXUj):
    # see description of self::compute_padding_dimwise_cc_Gaussian()
    n_xu, N = XUj.shape
    n_d     = xj.shape[0]
    A, b, bdxu_sum    = np.zeros((n_d, N,n_xu)), np.zeros(n_d), np.zeros(n_d)
    for i in range(n_d):
        aDa, aDa_dxu = Delta[i], Delta_dXUj[i,:,:]
        aDadxu_sum   = np.einsum('nd,dn->',aDa_dxu,XUj)
        
        A[i,:,:]    = aDa_dxu
        b[i]        = aDa 
        bdxu_sum[i] = -aDadxu_sum 
    return A, b, bdxu_sum

  def obs_avoidance_constraint_convexified(self, X_j, U_j, UP_vals, UP_args,
                                                k, obs, obs_type='sphere'):
    """ 
      Returns convexified obstacle avoidance chance constraints coefficients.

      Inputs: - X_j      : state around which the constraint is linearized [n_x, N]
              - U_j      : control ...                                     [n_u, N-1]
              - UP_vals  : dict of vals, see description in [self::get_final_set_constraint]
              - UP_args  : dict of pars, see description in [self::get_final_set_constraint]
              - k        : time index of constraint
              - obs      : obstacle
              - obs_type : Type of obstacles, can be 'sphere' or 'poly'
      Outpus: Coeficients s.t. A * [X,U] <= b
              - A : [N, n_xu]
              - b : scalar

      Returns the constraints coefficients of the obstacle (for obs)
      constraint g(x_k) <= 0 linearized at the state x_kj = X_j[:,k]
          s.t.            A * x_k <= b

                            dist > (bot_radius+obs_radius) 
              => ||x_k-obs_pos|| > (bot_radius+obs_radius)
          linearized =>
              dist_prev + n_prev*(x_k-x_p) > (bot_radius+obs_radius)
              n_prev*x_k  > -dist_prev + n_prev*x_p + (bot_radius+obs_radius)
            -(n_prev*x_k) <  dist_prev - n_prev*x_p - (bot_radius+obs_radius))
    """  
    assert(X_j.ndim==2 and U_j.ndim==2 and k<=X_j.shape[1] and k<=U_j.shape[1])
    UP_pars = deepcopy(self.UP_config)
    UP_pars.update(UP_args)
    self.check_UP_dicts_for_constraints(UP_vals, UP_pars)

    n_p, n_x, n_u, N = self.nb_pos_dim, self.n_x, self.n_u, X_j.shape[1]
    x_p              = X_j[:n_p, k]

    if obs_type=='sphere':
      pos, radius = obs[0][0:n_p], obs[1]
          
      dist_prev = np.linalg.norm(x_p-pos,2)
      n_prev    = (x_p-pos) / dist_prev

      # hessian
      n_dxkj = np.eye(n_p)
      for i in range(n_p):
          n_dxkj[i,:] += -(x_p[i]-pos[i])*(x_p-pos) / (dist_prev**2)
      n_dxkj /= dist_prev

      dist_prev = dist_prev-radius

    elif obs_type=='poly':
      pos3d = x_p[:n_p] #np.array([x_p[0], x_p[1], 0.])
      if n_p==2: 
        pos3d = np.append(pos3d, obs.c[2])
      dist_prev, pos = signed_distance_with_closest_point_on_surface(pos3d, obs)
      pos            = pos[:n_p]

      n_prev = (x_p-obs.c[:n_p]) / np.linalg.norm((x_p-obs.c[:n_p]),2)

      # hessian
      n_dxkj = np.eye(n_p)
      for i in range(n_p):
          n_dxkj[i,:] += -(x_p[i]-pos[i])*(x_p-pos) / (dist_prev**2)
      n_dxkj /= dist_prev

    else:
      raise NotImplementedError('Unknown obstacle type.')

    # padding by deterministic distance
    A  = np.zeros([N, n_x+n_u])
    b  = dist_prev - self.model.robot_radius

    # evaluate chance constraint
    if UP_pars["method"] == "Gaussian":
      Var, Var_dxu = UP_vals["Vars"][:n_p,:n_p,k], UP_vals["Vars_dxu"][:n_p,:n_p,:,:,k]
      A_uncert, b_uncert = self.compute_padding_linear_cc_Gaussian(k, -n_prev, -n_dxkj,
                                              x_p, X_j, U_j, Var, Var_dxu, 
                                              UP_pars["safety_prob"])

    elif ((UP_pars["method"]=="MC-Bayesian" or UP_pars["method"]=="MC-frequentist") and 
           UP_pars["shape"] == "ellipsoidal"):
      Q, Q_dxu = UP_vals["Qs"][:n_p,:n_p,k], UP_vals["Qs_dxu"][:n_p,:n_p,:,:,k]
      A_uncert, b_uncert = self.compute_padding_linear_cc_ellipsoidal(k, -n_prev, -n_dxkj,
                                                                      x_p, X_j, U_j, Q, Q_dxu)
    elif ((UP_pars["method"]=="MC-Bayesian" or UP_pars["method"]=="MC-frequentist") and 
           UP_pars["shape"] == "rectangular"):
      D, D_dxu = UP_vals["Deltas"][:n_p,k], UP_vals["Deltas_dxu"][:n_p,:,:,k]
      A_uncert, b_uncert = self.compute_padding_linear_cc_rectangular(k, -n_prev, -n_dxkj,
                                                                      x_p, X_j, U_j, D, D_dxu)

    else:
      raise NotImplementedError('[ccscp_model::obs_avoidance_constraint_convexified] ' + 
                                ' Unknown Uncertainty Propagation method.')

    A, b = A + A_uncert, b - b_uncert

    return A, b


  def compute_padding_linear_cc_Gaussian(self, k, a, a_dx, 
                                               xmean_j, Xj, Uj, Var_x, Var_x_dXUj, 
                                               prob):
    """
      Deterministic reformulation of the padding from the chance constraint
         Pr(a.T @ x <= 0.) > prob,     where x ~ N(xmean, Var_x)
                                        and Var_x depends on X, U,
                                (X,U are the means, xmean is k-th elem of X)
                   <=>
          f(X,U) = a.T @ xmean + PhiInv(prob) * sqrt(a.T @ Var_x @ a) <= 0.

      Returns linearized coeffs (A, b) s.t.
            f(X,U) ~= f(Xj,Uj) + d/dxu(f)(Xj,Uj) @ (X-Xj, U-Uj)
        or, f(X,U) ~= f(Xj,Uj) - d/dxu(f)(Xj,Uj) @ (Xj,Uj) + d/dxu(f)(Xj,Uj) @ (X,U)
                      \--------------- b ----------------/   \----- A -----/

      Inputs: - k       : ()           - index of xmean in Xj
              - a       : (n_p,),     (n_p < n_x)
              - a_dx    : (n_p, n_p)
              - xmean_j : (n_p,)       - mean of x, evaluated at Xj
              - Xj      : (n_x,  N )   - mean around which linearized
              - Uj      : (n_u, N-1)   - mean around which linearized
              - Var_x   : (n_p, n_p)   - variance of x, evaluated at (Xj,Uj)
              - Var_x_dXUj : (n_p,n_p, N,n_xu) - Jacobian of Var_x evaluated at (Xj,Uj)
              - prob    : ()
      Output: - A       : (N,n_xu)
              - b       : ()
    """
    n_a, n_x, n_u, N = a.shape[0], self.n_x, self.n_u, Xj.shape[1]
    A, b             = np.zeros([N, n_x+n_u]), 0.

    # deterministic part
    A[k,:n_a] = a
    b         = - a @ xmean_j

    # uncertain part
    Phi = np.sqrt(p_th_quantile_chi_squared(prob, self.nb_pos_dim))

    S, S_dxu = Var_x, Var_x_dXUj
    if (S.sum()>1e-6):

      a_S_a  = np.sqrt(a.T@S@a)
      aSadxu = np.einsum('x,y,xynd->nd', a,a,S_dxu) # (N,n_xu)

      a_S_adx = a.T @ S @ a_dx
      A[k,:n_a] += Phi * ( 1./(2.*a_S_a) ) * (2*a_S_adx)
      A[:,:]    += Phi * ( 1./(2.*a_S_a) ) * aSadxu

      aSadxu_sum = (aSadxu[:,:n_x].flatten())      @ ((Xj.T).flatten()) + (
                   (aSadxu[:(N-1),n_x:].flatten()) @ ((Uj.T).flatten())   )
      b += Phi * (+ a_S_a
                  - ( 1./(2.*a_S_a) ) * (
                          (2*a_S_adx) @ xmean_j
                          + aSadxu_sum  )
                 )
    return A, b
  def compute_padding_linear_cc_ellipsoidal(self, k, a, a_dx, 
                                                  xmean_j, Xj, Uj, Q_x, Q_x_dXUj):
    # see description of self::compute_padding_linear_cc_Gaussian()
    n_a, n_x, n_u, N = a.shape[0], self.n_x, self.n_u, Xj.shape[1]
    A, b             = np.zeros([N, n_x+n_u]), 0.

    # deterministic part
    A[k,:n_a] = a
    b         = - a @ xmean_j

    # uncertain part
    if (Q_x.sum()>1e-6):

      a_Q_a  = np.sqrt(a.T@Q_x@a)
      aQadxu = np.einsum('x,y,xynd->nd', a,a,Q_x_dXUj) # (N,n_xu)

      a_Q_adx = a.T @ Q_x @ a_dx
      A[k,:n_a] += 1./(2.*a_Q_a) * (2*a_Q_adx)
      A[:,:]    += 1./(2.*a_Q_a) * aQadxu

      aQadxu_sum = (aQadxu[:,:n_x].flatten())      @ ((Xj.T).flatten()) + (
                   (aQadxu[:(N-1),n_x:].flatten()) @ ((Uj.T).flatten())   )
      b += (+ a_Q_a
            - ( 1./(2.*a_Q_a) ) * (
                    (2*a_Q_adx) @ xmean_j
                    + aQadxu_sum  )       )
    return A, b
  def compute_padding_linear_cc_rectangular(self, k, a, a_dx, 
                                                  xmean_j, Xj, Uj, D_x, D_x_dXUj):
    # see description of self::compute_padding_linear_cc_Gaussian()
    n_a, n_x, n_u, N = a.shape[0], self.n_x, self.n_u, Xj.shape[1]
    A, b             = np.zeros([N, n_x+n_u]), 0.

    # deterministic part
    A[k,:n_a] = a
    b         = - a @ xmean_j

    # uncertain part
    if (D_x.sum()>1e-6):
      # outer-ellipsoid of rectangle
      Q_x    = D_x**2 * np.eye(n_a)
      a_Q_a  = np.sqrt(a.T@Q_x@a)
      # aQadxu = np.einsum('x,x,xnd->nd', a,a, D_x_dXUj**2) # (N,n_xu)
      aQadxu = np.einsum('x,x,xnd->nd', 2*a**2, D_x, D_x_dXUj) # (N,n_xu)

      a_Q_adx = a.T @ Q_x @ a_dx
      A[k,:n_a] += 1./(2.*a_Q_a) * (2*a_Q_adx)
      A[:,:]    += 1./(2.*a_Q_a) * aQadxu

      aQadxu_sum = (aQadxu[:,:n_x].flatten())      @ ((Xj.T).flatten()) + (
                   (aQadxu[:(N-1),n_x:].flatten()) @ ((Uj.T).flatten())   )
      b += (+ a_Q_a
            - ( 1./(2.*a_Q_a) ) * (
                    (2*a_Q_adx) @ xmean_j
                    + aQadxu_sum  )       )
    return A, b

