import matplotlib.pyplot as plt
import numpy as np

# plotting of shapes
import matplotlib.patches as patches
import matplotlib.transforms as transforms
from scipy.stats import chi2


def plot_state_traj(states, idx=[0,1], color='C0', linestyle='-', alpha=1.0, label=None):
    """
    Plots each x-y trajectory on a plane according to idx:
    """
    traj = states
    plt.plot(traj[:,idx[0]],traj[:,idx[1]], 
             color=color, linestyle=linestyle, alpha=alpha, label=label)

def plot_mean_traj_with_gaussian_uncertainty_ellipses(mus, Sigmas, idx=[0,1], probability=0.9,
                                                        additional_radius=0., 
                                                        alpha=None, color="b", label=None):
    if mus.shape[0] != len(Sigmas):
        raise ValueError("mus (%d), Sigmas (%d) must have same nb. of elements" %(mus.shape[0],len(Sigmas)))

    T = len(Sigmas)
    ax = plt.gca()

    plot_state_traj(mus, idx=idx, color=color, label=label)

    if alpha == None:
        alpha = 0.1 * min(1., 30/mus.shape[0])
    for k in range(T):
        mu    = mus[k,idx]
        Sigma = Sigmas[k][np.ix_(idx,idx)]
        plot_gaussian_confidence_ellipse(ax, mu, Sigma, 
                                            probability=probability, 
                                            additional_radius=additional_radius,
                                            alpha=alpha, color=color)

def plot_mean_var(means, variances, prob=0.9, color='C0'):
    # means     - (N,)
    # variances - (N,)
    stds = np.sqrt(variances)
    sqrt_chi_squared_quantile_val = np.sqrt(chi2.ppf(prob, 1))

    H = means.shape[0]

    plt.fill_between(np.arange(0,H), means - sqrt_chi_squared_quantile_val*stds, 
                                     means + sqrt_chi_squared_quantile_val*stds, 
                                     color=color, alpha=0.2)
    plt.plot(np.arange(0,H),means, color=color)

def plot_gaussian_confidence_ellipse(ax, mu, Sigma, probability=0.9, additional_radius=0., alpha=0.1, color="b"):
    n_dofs = mu.shape[0]
    chi_squared_quantile_val = chi2.ppf(probability, n_dofs)
    Q = chi_squared_quantile_val * Sigma
    plot_ellipse(ax, mu, Q, additional_radius=additional_radius, color=color, alpha=alpha)

def plot_ellipse(ax, mu, Q, additional_radius=0., color='blue', alpha=0.1):
    """
    Based on
    http://stackoverflow.com/questions/17952171/not-sure-how-to-fit-data-with-a-gaussian-python.
    """

    # Compute eigenvalues and associated eigenvectors
    vals, vecs = np.linalg.eigh(Q)

    # Compute "tilt" of ellipse using first eigenvector
    x, y = vecs[:, 0]
    theta = np.degrees(np.arctan2(y, x))

    # Eigenvalues give length of ellipse along each eigenvector
    w, h =  2. * (np.sqrt(vals) + additional_radius)
    ellipse = patches.Ellipse(mu, w, h, theta, color=color, alpha=alpha)  # color="k")
    ellipse.set_clip_box(ax.bbox)
    ax.add_artist(ellipse) 

def plot_rectangle(ax, center, widths, additional_w=0., color='blue', alpha=0.1, noFaceColor=False):
    """
    Plots a rectangle with a given center and total widths
    arguments:  - center    - (2,) center
                - widths    - (2,) total widths  from and to edges of rectangle
    """
    if center.shape[0] != 2 or widths.shape[0] != 2:
        assert False, 'plot_rectangle function can only plot in 2d.'
    facecolor = color
    if noFaceColor:
        facecolor = None

    deltas = [widths[0]+additional_w, widths[1]+additional_w]
    bottom_left = (center[0] - deltas[0]/2., center[1] - deltas[1]/2.)
    rect = patches.Rectangle((bottom_left[0],bottom_left[1]),deltas[0],deltas[1], \
                                linewidth=1,edgecolor=color,facecolor=facecolor,alpha=alpha)
    ax.add_patch(rect)
    return ax
