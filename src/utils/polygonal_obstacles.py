import numpy as np
from numpy.linalg import norm as norm
from numpy import sqrt

""" ---------------------------------
    ToDo: Replace this custom-made signed
          distance function for polytopic
          objects with faster library,
          e.g. FCL.:
          
    github.com/BerkeleyAutomation/python-fcl/
          (ongoing work, please contact
            thomas.lew@stanford.edu 
           for more information.)
"""

class PolygonalObstacleClass:
  def __init__(self, c, dx,dy,dz,
                     n_vec, p_vec):
    # center
    self.c = c  # (3,)  

    # half widths
    self.dx = dx
    self.dy = dy
    self.dz = dz
    self.widths = 2.*np.array([dx,dy,dz])

    # normals
    self.n_vec = n_vec # (3,)

    # positions of corners
    self.p_vec = p_vec # (3,)

def PolygonalObstacle(center, widths_vec):
  # center     : (3,)
  # widths_vec : (3,) width along each dimension (2*radius)
  c  = np.array(center)
  dx = widths_vec[0] / 2.
  dy = widths_vec[1] / 2.
  dz = widths_vec[2] / 2.

  p_vec = compute_corners_positions(c[0], c[1], c[2], dx, dy, dz)
  n_vec = compute_surfaces_normals(c[0], c[1], c[2], dx, dy, dz)
  return PolygonalObstacleClass(c, dx,dy,dz,
                                n_vec, p_vec)


def compute_surfaces_normals(cx, cy, cz, dx, dy, dz):
  n1 = np.array([ 0, -1,  0])
  n2 = np.array([ 1,  0,  0])
  n3 = np.array([ 0,  1,  0])
  n4 = np.array([-1,  0,  0])
  n5 = np.array([ 0,  0,  1])
  n6 = np.array([ 0,  0, -1])
  return [n1, n2, n3, n4, n5, n6]

def get_surfaces_normals(obs):
  # obs::PolygonalObstacle
  n = obs.n_vec
  return n[0], n[1], n[2], n[3], n[4], n[5]


def compute_corners_positions(cx, cy, cz, dx, dy, dz):
  p125 = np.array([cx + dx, cy - dy, cz + dz])
  p235 = np.array([cx + dx, cy + dy, cz + dz])
  p345 = np.array([cx - dx, cy + dy, cz + dz])
  p145 = np.array([cx - dx, cy - dy, cz + dz])
  p126 = np.array([cx + dx, cy - dy, cz - dz])
  p236 = np.array([cx + dx, cy + dy, cz - dz])
  p346 = np.array([cx - dx, cy + dy, cz - dz])
  p146 = np.array([cx - dx, cy - dy, cz - dz])
  return [p125, p235, p345, p145, p126, p236, p346, p146]

def get_corners_positions(obs):
  # obs::PolygonalObstacle
  p = obs.p_vec
  p125, p235, p345, p145, p126, p236, p346, p146 = (p[0], 
        p[1], p[2], p[3], p[4], p[5], p[6], p[7])
  return p125, p235, p345, p145, p126, p236, p346, p146



def B_is_in_zone_i(x, i, obs):
  # x::Vector
  # i::Int
  # obs::PolygonalObstacle
  c = obs.c
  n = obs.n_vec[i]
  dx, dy, dz = obs.dx, obs.dy, obs.dz

  if (i == 0 or i == 2):
    return (np.dot(x-c,n) >= dy)
  elif (i == 1 or i == 3):
    return (np.dot(x-c,n) >= dx)
  elif (i == 4 or i == 5):
    return (np.dot(x-c,n) >= dz)
  else:
    error("[polygonal_obstacles.jl::is_in_zone_i] i not recognized.")


def signed_distance(x, obs):
  # x::Vector 
  # obs::PolygonalObstacle

  # extract values
  c                                              = obs.c
  cx, cy, cz                                     = obs.c[0], obs.c[1], obs.c[2]
  dx, dy, dz                                     = obs.dx,   obs.dy,   obs.dz
  p125, p235, p345, p145, p126, p236, p346, p146 = get_corners_positions(obs)
  n1, n2, n3, n4, n5, n6                         = get_surfaces_normals(obs)

  # check in which zones it is
  Bz_vec = np.zeros(6, dtype=bool) # vector of booleans: i-th value is true if x is in i-th zone
  for i in range(6):
    Bz_vec[i] = B_is_in_zone_i(x, i, obs)

  # x is outside a corner
  if (Bz_vec[0] and Bz_vec[1] and Bz_vec[4]): # in 125
    return norm(x-p125)
  elif (Bz_vec[1] and Bz_vec[2] and Bz_vec[4]): # in 235
    return norm(x-p235)
  elif (Bz_vec[2] and Bz_vec[3] and Bz_vec[4]): # in 345
    return norm(x-p345)
  elif (Bz_vec[0] and Bz_vec[3] and Bz_vec[4]): # in 145
    return norm(x-p145)
  elif (Bz_vec[0] and Bz_vec[1] and Bz_vec[5]): # in 126
    return norm(x-p126)
  elif (Bz_vec[1] and Bz_vec[2] and Bz_vec[5]): # in 236
    return norm(x-p236)
  elif (Bz_vec[2] and Bz_vec[3] and Bz_vec[5]): # in 346
    return norm(x-p346)
  elif (Bz_vec[0] and Bz_vec[3] and Bz_vec[5]): # in 146
    return norm(x-p146)

  # x is outside an edge
  elif (Bz_vec[0] and Bz_vec[1]): # in 12
    xhat = [x[0], x[1], cz+dz] 
    return norm(xhat-p125)
  elif (Bz_vec[1] and Bz_vec[2]): # in 23
    xhat = [x[0], x[1], cz+dz] 
    return norm(xhat-p235)
  elif (Bz_vec[2] and Bz_vec[3]): # in 34
    xhat = [x[0], x[1], cz+dz] 
    return norm(xhat-p345)
  elif (Bz_vec[0] and Bz_vec[3]): # in 14
    xhat = [x[0], x[1], cz+dz] 
    return norm(xhat-p145)
  elif (Bz_vec[0] and Bz_vec[4]): # in 15
    xhat = [x[0]+dx, x[2], x[2]] 
    return norm(xhat-p125)
  elif (Bz_vec[1] and Bz_vec[4]): # in 25
    xhat = [x[0], x[1]-dy, x[2]] 
    return norm(xhat-p125)
  elif (Bz_vec[2] and Bz_vec[4]): # in 35
    xhat = [x[0]+dx, x[2], x[2]] 
    return norm(xhat-p235)
  elif (Bz_vec[3] and Bz_vec[4]): # in 45
    xhat = [x[0], x[1]+dy, x[2]] 
    return norm(xhat-p345)
  elif (Bz_vec[0] and Bz_vec[5]): # in 16
    xhat = [x[0]+dx, x[2], x[2]] 
    return norm(xhat-p126)
  elif (Bz_vec[1] and Bz_vec[5]): # in 26
    xhat = [x[0], x[1]-dy, x[2]] 
    return norm(xhat-p126)
  elif (Bz_vec[2] and Bz_vec[5]): # in 36
    xhat = [x[0]+dx, x[2], x[2]] 
    return norm(xhat-p236)
  elif (Bz_vec[3] and Bz_vec[5]): # in 46
    xhat = [x[0], x[1]+dy, x[2]] 
    return norm(xhat-p346)

  # x is outside a flat edge (not on border)
  elif Bz_vec[0]: # in 1
    return (np.dot(x-c,n1) - dy)
  elif Bz_vec[1]: # in 2
    return (np.dot(x-c,n2) - dx)
  elif Bz_vec[2]: # in 3
    return (np.dot(x-c,n3) - dy)
  elif Bz_vec[3]: # in 4
    return (np.dot(x-c,n4) - dx)
  elif Bz_vec[4]: # in 5
    return (np.dot(x-c,n5) - dz)
  elif Bz_vec[5]: # in 6
    return (np.dot(x-c,n6) - dz)

  # is inside
  else:
    return (norm(x-c) - sqrt(dx**2+dy**2+dz**2)) # not quite accurate, but good enough for obstacle avoidance


def signed_distance_with_closest_point_on_surface(x, obs):
  # x::Vector 
  # obs::PolygonalObstacle

  # extract values
  c                                              = obs.c
  cx, cy, cz                                     = obs.c[0], obs.c[1], obs.c[2]
  dx, dy, dz                                     = obs.dx,   obs.dy,   obs.dz
  p125, p235, p345, p145, p126, p236, p346, p146 = get_corners_positions(obs)
  n1, n2, n3, n4, n5, n6                         = get_surfaces_normals(obs)


  # check in which zones it is
  Bz_vec = np.zeros(6, dtype=bool) # vector of booleans: i-th value is true if x is in i-th zone
  for i in range(6):
    Bz_vec[i] = B_is_in_zone_i(x, i, obs)

  # x is outside a corner
  if (Bz_vec[0] and Bz_vec[1] and Bz_vec[4]): # in 125
    return norm(x-p125), p125
  elif (Bz_vec[1] and Bz_vec[2] and Bz_vec[4]): # in 235
    return norm(x-p235), p235
  elif (Bz_vec[2] and Bz_vec[3] and Bz_vec[4]): # in 345
    return norm(x-p345), p345
  elif (Bz_vec[0] and Bz_vec[3] and Bz_vec[4]): # in 145
    return norm(x-p145), p145
  elif (Bz_vec[0] and Bz_vec[1] and Bz_vec[5]): # in 126
    return norm(x-p126), p126
  elif (Bz_vec[1] and Bz_vec[2] and Bz_vec[5]): # in 236
    return norm(x-p236), p236
  elif (Bz_vec[2] and Bz_vec[3] and Bz_vec[5]): # in 346
    return norm(x-p346), p346
  elif (Bz_vec[0] and Bz_vec[3] and Bz_vec[5]): # in 146
    return norm(x-p146), p146

  # x is outside an edge
  elif (Bz_vec[0] and Bz_vec[1]): # in 12
    xhat = [x[0], x[1], cz+dz]
    ds = norm(xhat-p125)
    return ds, (x-(xhat-p125))
  elif (Bz_vec[1] and Bz_vec[2]): # in 23
    xhat = [x[0], x[1], cz+dz]
    ds = norm(xhat-p235)
    return ds, (x-(xhat-p235))
  elif (Bz_vec[2] and Bz_vec[3]): # in 34
    xhat = [x[0], x[1], cz+dz]
    ds = norm(xhat-p345)
    return ds, (x-(xhat-p345))
  elif (Bz_vec[0] and Bz_vec[3]): # in 14
    xhat = [x[0], x[1], cz+dz]
    ds = norm(xhat-p145)
    return ds, (x-(xhat-p145))
  elif (Bz_vec[0] and Bz_vec[4]): # in 15
    xhat = [c[0]+dx, x[1], x[2]]
    ds = norm(xhat-p125)
    return ds, (x-(xhat-p125))
  elif (Bz_vec[1] and Bz_vec[4]): # in 25
    xhat = [x[0], c[1]-dy, x[2]]
    ds = norm(xhat-p125)
    return ds, (x-(xhat-p125))
  elif (Bz_vec[2] and Bz_vec[4]): # in 35
    xhat = [c[0]+dx, x[1], x[2]]
    ds = norm(xhat-p235)
    return ds, (x-(xhat-p235))
  elif (Bz_vec[3] and Bz_vec[4]): # in 45
    xhat = [x[0], c[1]+dy, x[2]]
    ds = norm(xhat-p345)
    return ds, (x-(xhat-p345))
  elif (Bz_vec[0] and Bz_vec[5]): # in 16
    xhat = [c[0]+dx, x[1], x[2]]
    ds = norm(xhat-p126)
    return ds, (x-(xhat-p126))
  elif (Bz_vec[1] and Bz_vec[5]): # in 26
    xhat = [x[0], c[1]-dy, x[2]]
    ds = norm(xhat-p126)
    return ds, (x-(xhat-p126))
  elif (Bz_vec[2] and Bz_vec[5]): # in 36
    xhat = [c[0]+dx, x[1], x[2]]
    ds = norm(xhat-p236)
    return ds, (x-(xhat-p236))
  elif (Bz_vec[3] and Bz_vec[5]): # in 46
    xhat = [x[0], c[1]+dy, x[2]]
    ds = norm(xhat-p346)
    return ds, (x-(xhat-p346))

  # x is outside a flat edge (not on border)
  elif Bz_vec[0]: # in 1
    ds = (np.dot(x-c,n1)-dy)
    return ds, (x - n1*ds)
  elif Bz_vec[1]: # in 2
    ds = (np.dot(x-c,n2)-dx)
    return ds, (x - n2*ds)
  elif Bz_vec[2]: # in 3
    ds = (np.dot(x-c,n3)-dy)
    return ds, (x - n3*ds)
  elif Bz_vec[3]: # in 4
    ds = (np.dot(x-c,n4)-dx)
    return ds, (x - n4*ds)
  elif Bz_vec[4]: # in 5
    ds = (np.dot(x-c,n5)-dz)
    return ds, (x - n5*ds)
  elif Bz_vec[5]: # in 6
    ds = (np.dot(x-c,n6)-dz)
    return ds, (x - n6*ds)

  # is inside
  else:
    # not quite accurate, but good enough for obstacle avoidance
    ds = 1.01*(norm(x-c) - sqrt(dx**2+dy**2+dz**2))
    pt_outside = (x+(x-c)*(-ds))
    _, pt_on_surface = signed_distance_with_closest_point_on_surface(pt_outside, obs)
    ds = -np.linalg.norm(x-pt_on_surface)
    return ds, pt_on_surface