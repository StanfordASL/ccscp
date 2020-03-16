import sys, os
sys.path.append('./Models/')
sys.path.append('../src/ccscp/')
sys.path.append('../src/utils/')
import astrobee_plot
from cc_ocp import *
from Models.astrobee      import Model
from astro_iss_plot import plot
from astrobee_mc import monte_carlo

# Initialize Astrobe model
m = Model()
N = m.N

# Create chance-constrained problem
problem = CCOCP(m)

# Solve problem using CCSCP
problem.solve_ccscp(m)

# (Optional) Verify with Monte-Carlo
X_sol, U_sol = problem.get_XU_solution_CCSCP(m)
Xs_true, Us_true, nb_in_obs = monte_carlo(X_sol, U_sol, m, N_MC=1)

# Plot results
plot(problem.all_X, problem.all_U, problem.all_V, m, Xs_true, Us_true)

# View solutions at each SCP iteration
# astrobee_plot.plot(problem.all_X, problem.all_U, problem.all_V, m)