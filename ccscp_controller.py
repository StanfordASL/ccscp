import numpy as np
np.set_printoptions(precision=3, threshold=10**7)
import time
from copy import copy, deepcopy
from warnings import warn


from ccscp.src.ccscp.cc_ocp import CCOCP
from ccscp.ccscp_model import CCSCPModel

default_controller_config = {
    "verbose": True,
}

# Parameters for uncertainty propagation
default_UP_controller_config = {
    # "shape":            "ellipsoidal", # "rectangular"
    # "method":           "Gaussian",    # "MC-Bayesian" # "MC-frequentist"
    "shape":            "rectangular",
    # "shape":            "ellipsoidal",
    "method":           "MC-frequentist", #"MC-Bayesian",
    "reuse_ABgrads":    False,
    "reuse_presampled_dynamics":  False,
    "N_MC":             100,
    "return_grad":      True,
    "return_parts":     True,
    "B_feedback":       False,
    "safety_prob":      0.9 # probability level [%] of chance constraints
}

class CCSCPController:
  def __init__(self, model, problem, 
                     config=default_controller_config):
    """ 
    Inputs: - model:   AdaptiveDynamics (in Models)
            - problem: PlanningProblem  (in Core)
    """
    self.config = copy(default_controller_config)
    self.config.update(config)
    self.UP_config = copy(default_UP_controller_config)

    self.problem = problem
    self.model   = CCSCPModel(model, UP_config=self.UP_config)

    # Create chance-constrained optimal control problem (CC OCP)
    self.ocp = CCOCP(self.model, self.problem)
    self.success = False

    self.reset()


  def reset(self):
    self.model.reset()
    # [xdim, T+1], [udim, T]
    self.Xtraj, self.Utraj = None, None

  def incorporate_transition(self, x, u, xp):
      self.model.incorporate_transition(x,u,xp)

  def get_uncertainty_params_norm(self):
      return self.model.get_uncertainty_params_norm()

  def get_info_gain_traj(self, Xtraj, Utraj):
      info_costs, _ = self.model.model.cost_exploration_convexified(Xtraj,Utraj)
      return np.linalg.norm(np.sum(info_costs))

  def set_horizon(self, N):
    self.ocp.update_horizon(N)


  def plan(self, N, init_method='straightline'):
      self.optimized_feasible = self.optimize(init_method=init_method)

      Xtraj, Utraj       = self.get_optimized_trajectory()
      UP_vals, UP_config = self.get_optimized_uncertainty()

      return Xtraj, Utraj, UP_vals, UP_config

  def optimize(self, init_method='straightline', verbose=False):
    self.success = self.ocp.solve_ccscp(init_method)
    if verbose:
      print('\n[ccscp_controller::optimize()] Optimized,\n success=',self.success, 
            ', final=',self.model.final_constraint_feasible, 
            ', SCP nb=',len(self.ocp.all_X))
    self.success = self.success and (len(self.ocp.all_X)<self.model.scp_config['NB_SCP_iter_max'])
    self.success = self.success and self.model.final_constraint_feasible
    if (verbose and not(self.model.final_constraint_feasible) and 
                    (len(self.ocp.all_X)<self.model.scp_config['NB_SCP_iter_max'])):
      print('[ccscp_controller::optimize] SCP converged, but padding was too high.')

    if self.success:
      self.Xtraj, self.Utraj = self.ocp.get_XU_solution_CCSCP()

      # Check again if final constraint feasible with solution
      UP_vals, _ = self.get_optimized_uncertainty()
      if self.problem.B_go_to_safe_set:
        A,l,u = self.model.get_final_set_constraint(self.problem.X_safe, UP_vals, self.ocp.UP_config)
      else:
        A,l,u = self.model.get_final_set_constraint(self.problem.X_goal, UP_vals, self.ocp.UP_config)
      self.success = self.success and self.model.final_constraint_feasible
      if not(self.model.final_constraint_feasible):
        print('[ccscp_controller::optimize] After re-evaluating uncertainty, padding too high.')
      if not((A @ self.Xtraj[:,-1] <= u).all() and (l <= A @ self.Xtraj[:,-1]).all()):
        self.success = False 
        if verbose:
          print('[ccscp_controller::optimize] After re-evaluating uncertainty, Xend unfeasible.')

      # return success
      return self.success

    else:
      # print('\n**** Returning initial trajectory ****\n')
      self.Xtraj, self.Utraj = self.ocp.get_XU_solution_CCSCP()
      # self.Xtraj, self.Utraj = self.ocp.get_XU_initial_trajectory()
      # print(self.Xtraj)
      return self.success

  def get_optimized_trajectory(self):
    # [xdim, T+1], [udim, T]
    return self.Xtraj, self.Utraj

  def get_optimized_uncertainty(self):
    X, U = self.Xtraj, self.Utraj
    UP_vals = self.model.propagate_uncertainty(X, U, self.UP_config)
    return UP_vals, self.UP_config

  def get_feedback_gains(self, X=np.zeros(0), U=np.zeros(0),
                               B_reuse_precomputed=True):
    if B_reuse_precomputed==False and not(X.ndim==2) and not(U.ndim==2):
      raise KeyError('[ccscp_controller::get_feedback_gains] no trajectory provided.')

    if self.model.UP_config["B_feedback"]:
      return self.model.get_feedback_gains(X=X, U=U, 
                                           B_reuse_precomputed=B_reuse_precomputed)
    else:
      return np.zeros((0,self.model.n_u,self.model.n_x))

  # def get_optimized_conf_sets(self, B_reuse_presampled_dynamics=False):
  #   X, U, Q0          = self.Xtraj, self.Utraj, self.problem.Q0
  #   QDs, parts, K_fbs = self.model.compute_prob_conf_sets(X, U, Q0, prob=0.9,
  #                             B_reuse_presampled_dynamics=B_reuse_presampled_dynamics)
  #   return QDs, parts, K_fbs