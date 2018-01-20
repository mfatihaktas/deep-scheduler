import pprint
import numpy as np

from patch import *
from rvs import *
from sim import *
from scheduling import *

from mpi4py import MPI

def test():
  comm = MPI.COMM_WORLD
  size = comm.Get_size()
  rank = comm.Get_rank()
  
  N, T = 3, 5
  flag = np.empty(1, dtype='f')
  if rank == 0:
    for d in range(1, size):
      print("rank= {}, sending to d= {}".format(rank, d) )
      flag[0] = 1
      comm.Send([flag, MPI.FLOAT], dest=d)
    for d in range(1, size):
      n_t_r_l = np.empty(N*T, dtype='f')
      comm.Recv([n_t_r_l, MPI.FLOAT], source=d)
      n_t_r_l = n_t_r_l.reshape((N, T))
      print("rank= {}, recved from d= {} n_t_r_l= \n{}".format(rank, d, n_t_r_l) )
  else:
    comm.Recv([flag, MPI.FLOAT], source=0)
    print("rank= {}, recved flag= {}".format(rank, flag) )
    # do sth
    # n_t_r_l = np.random.rand(N, T)
    n_t_r_l = np.arange(N*T, dtype='f') * rank
    comm.Send([n_t_r_l, MPI.FLOAT], dest=0)
    n_t_r_l = n_t_r_l.reshape((N, T))
    print("rank= {}, returned to master n_t_r_l= \n{}".format(rank, n_t_r_l) )
  
if __name__ == "__main__":
  test()