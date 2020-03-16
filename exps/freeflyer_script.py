import sys, os
sys.path.append('./Models/')
sys.path.append('../src/ccscp/')
sys.path.append('../src/utils/')

import time

from cc_ocp import *
from Models.freeflyer      import Model
from Models.freeflyer_plot import plot



## Initialize model from freeflyerr class
m = Model()
N = m.N

# Create chance-constrained problem
problem = CCOCP(m)

## Solve problem using GuSTO which calls osqp
problem.solve_ccscp(m)

start = time.time()
problem.solve_ccscp(m)
end = time.time()
print("\n\nelapsed time = ",end-start,"\n\n")

# Plot results
plot(problem.all_X, problem.all_U, problem.all_V, m)