import numpy as np
from scipy.stats import chi2, norm

def p_th_quantile_cdf_normal(probability):
    return norm.ppf(probability)

def p_th_quantile_chi_squared(probability, n_dofs):
    return chi2.ppf(probability, n_dofs)
    
def sample_from_multivariate_gaussian(mu, Sigma, N):
    return np.random.multivariate_normal(mu, Sigma, (N))

def is_in_gaussian_confidence_ellipsoid(x, mu, Sigma, probability):
    if mu.size != Sigma.shape[0] or x.size != mu.size:
        raise ValueError("x (%d), mu (%d) and Sigma (%d,%d) must be the same size" %(x.size, mu.size, Sigma.shape[0], Sigma.shape[1]))
        
    chi_squared_quantile_val = p_th_quantile_chi_squared(probability, mu.size)
    Q = chi_squared_quantile_val * Sigma
    return is_in_ellipse(x, mu, Q)
    
def is_in_ellipse(x, mu, Q):
    return (x-mu).T @ np.linalg.solve(Q, (x-mu)) < 1.

def count_nb_in_ellipse(Xs, mu, Q):
    if mu.size != Q.shape[0] or Xs.shape[1] != mu.size:
        raise ValueError("Xs (%d,%d), mu (%d) and Q (%d,%d) must be the same size" %(Xs.shape[0], Xs.shape[1], mu.size, Q.shape[0], Q.shape[1]))
    
    nb_in_ellipse = 0.
    for xi in Xs:
        if is_in_ellipse(xi, mu, Q) == True:
            nb_in_ellipse += 1
    return nb_in_ellipse

def count_nb_in_ellipse_and_get_idx_inside(Xs, mu, Q):
    if mu.size != Q.shape[0] or Xs.shape[1] != mu.size:
        raise ValueError("Xs (%d,%d), mu (%d) and Q (%d,%d) must be the same size" %(Xs.shape[0], Xs.shape[1], mu.size, Q.shape[0], Q.shape[1]))
    
    nb_in_ellipse = 0.
    indices_inside = []
    for idx,xi in enumerate(Xs):
        if is_in_ellipse(xi, mu, Q) == True:
            indices_inside.append(idx)
            nb_in_ellipse += 1
    return nb_in_ellipse, indices_inside

def percentage_in_ellipse(Xs, mu, Q):
    if mu.size != Q.shape[0] or Xs.shape[1] != mu.size:
        raise ValueError("Xs (%d,%d), mu (%d) and Q (%d,%d) must be the same size" %(Xs.shape[0], Xs.shape[1], mu.size, Q.shape[0], Q.shape[1]))
        
    return count_nb_in_ellipse(Xs, mu, Q) / Xs.shape[0]

def sample_pts_unit_ball(dim, NB_pts):
    """
    Uniformly samples points in a d-dimensional sphere (in a ball)
    Points characterized by    ||x||_2 < 1
    arguments:  dim    - nb of dimensions
                NB_pts - nb of points
    output:     pts    - points sampled uniformly in ball [xdim x NB_pts]
    Reference: http://extremelearning.com.au/how-to-generate-uniformly-random-points-on-n-spheres-and-n-balls/
    """
    us    = np.random.normal(0,1,(dim,NB_pts))
    norms = np.linalg.norm(us, 2, axis=0)
    rs    = np.random.random(NB_pts)**(1.0/dim)
    pts   = rs*us / norms
    return pts
    
def sample_pts_in_ellispoid(mu, Q, NB_pts):
    """
    Uniformly samples points in an ellipsoid, specified as
            (xi-mu)^T Q^{-1} (xi-mu) <= 1
    arguments: mu - mean [dim]
                Q - Q [dim x dim]
    output:     pts - points sampled uniformly in ellipsoid [xdim x NB_pts]
    """
    xs = sample_pts_unit_ball(mu.shape[0], NB_pts)
    E  = np.linalg.cholesky(Q)
    ys = (np.array(E@xs).T + mu).T
    return ys