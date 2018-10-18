import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.use('Agg')
import matplotlib.pyplot as plot

from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.cm
import matplotlib.colorbar
import matplotlib.colors

def cuboid_data(center, size=(1,1,1) ):
  # code taken from http://stackoverflow.com/questions/30715083/python-plotting-a-wireframe-3d-cuboid?noredirect=1&lq=1
  # suppose axis direction: x: to left; y: to inside; z: to upper
  # get the (left, outside, bottom) point
  o = [a - b / 2 for a, b in zip(center, size)]
  # get the length, width, and height
  l, w, h = size
  x = [[o[0], o[0] + l, o[0] + l, o[0], o[0]],  # x coordinate of points in bottom surface
       [o[0], o[0] + l, o[0] + l, o[0], o[0]],  # x coordinate of points in upper surface
       [o[0], o[0] + l, o[0] + l, o[0], o[0]],  # x coordinate of points in outside surface
       [o[0], o[0] + l, o[0] + l, o[0], o[0]]]  # x coordinate of points in inside surface
  y = [[o[1], o[1], o[1] + w, o[1] + w, o[1]],  # y coordinate of points in bottom surface
       [o[1], o[1], o[1] + w, o[1] + w, o[1]],  # y coordinate of points in upper surface
       [o[1], o[1], o[1], o[1], o[1]],          # y coordinate of points in outside surface
       [o[1] + w, o[1] + w, o[1] + w, o[1] + w, o[1] + w]]    # y coordinate of points in inside surface
  z = [[o[2], o[2], o[2], o[2], o[2]],                        # z coordinate of points in bottom surface
       [o[2] + h, o[2] + h, o[2] + h, o[2] + h, o[2] + h],    # z coordinate of points in upper surface
       [o[2], o[2], o[2] + h, o[2] + h, o[2]],                # z coordinate of points in outside surface
       [o[2], o[2], o[2] + h, o[2] + h, o[2]]]                # z coordinate of points in inside surface
  return x, y, z

def plot_cube_at(ax, pos, c, alpha=0.1):
  X, Y, Z = cuboid_data( (pos[0],pos[1],pos[2]) )
  ax.plot_surface(X, Y, Z, color=c, rstride=1, cstride=1, alpha=0.1)

def plot_matrix(ax, x, y, z, data, cmap='jet', cax=None, alpha=0.1):
  norm = matplotlib.colors.Normalize(vmin=data.min(), vmax=data.max())
  colors = lambda i,j,k : matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap).to_rgba(data[i,j,k]) 
  for i, xi in enumerate(x):
    for j, yi in enumerate(y):
      for k, zi, in enumerate(z):
        plot_cube_at(ax, pos=(xi, yi, zi), c=colors(i,j,k), alpha=alpha)
  
  if cax !=None:
    cbar = matplotlib.colorbar.ColorbarBase(cax, cmap=cmap, norm=norm, orientation='vertical')
    cbar.set_ticks(np.unique(data) )
    # set the colorbar transparent as well
    cbar.solids.set(alpha=alpha)
