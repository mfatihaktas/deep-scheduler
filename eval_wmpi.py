import sys, time
import numpy as np
from mpi4py import MPI

from scheduler import *
from modeling import *

def eval_wmpi(rank):
  log(INFO, "starting;", rank=rank)
  if rank == 0:
    blog(sinfo_m=sinfo_m)
    
    ro_scherid_X_l_m = {}
    for ro in ro_l:
      scherid_X_l_m = {}
      for scher in scher_l:
        scherid_X_l_m[scher._id] = {'ESl_l': None, 'StdSl_l': None, 'ET_l': None, 'StdT_l': None, 'Eload_l': None}
      ro_scherid_X_l_m[ro] = scherid_X_l_m
  
  sys.stdout.flush()
  comm.barrier()
  
  for ro in ro_l:
    sinfo_m['ar'] = ar_for_ro(ro, N, Cap, k, R, L, Sl)
    if rank == 0:
      scheri__ET_l_l, scheri__StdT_l_l = [], []
      scheri__ESl_l_l, scheri__StdSl_l_l = [], []
      scheri__Eload_l_l = []
      for i, scher in enumerate(scher_l):
        for p in range(1, num_mpiprocs):
          scher_i = np.array([i], dtype='i')
          log(INFO, "Master sending;", scher_i=i, p=p)
          sys.stdout.flush()
          comm.Send([scher_i, MPI.INT], dest=p)
        
        ESl_l, StdSl_l = [], []
        ET_l, StdT_l = [], []
        Eload_l = []
        for p in range(1, num_mpiprocs):
          ESl_StdSl_ET_StdT_Eload = np.empty(5, dtype=np.float64)
          comm.Recv(ESl_StdSl_ET_StdT_Eload, source=p)
          
          ESl_l.append(ESl_StdSl_ET_StdT_Eload[0] )
          StdSl_l.append(ESl_StdSl_ET_StdT_Eload[1] )
          ET_l.append(ESl_StdSl_ET_StdT_Eload[2] )
          StdT_l.append(ESl_StdSl_ET_StdT_Eload[3] )
          Eload_l.append(ESl_StdSl_ET_StdT_Eload[4] )
        log(INFO, "Master; ro= {}".format(ro), scher=scher, \
          ESl_l=ESl_l, StdSl_l=StdSl_l, ET_l=ET_l, StdT_l=StdT_l, Eload_l=Eload_l)
        sys.stdout.flush()
        
        ro_scherid_X_l_m[ro][scher._id].update({
          'ESl_l': ESl_l, 'StdSl_l': StdSl_l, 'ET_l': ET_l, 'StdT_l': StdT_l, 'Eload_l': Eload_l})
        log(INFO, "Master;", ro_scherid_X_l_m=ro_scherid_X_l_m)
        time.sleep(4)
      
      for p in range(1, num_mpiprocs):
        scher_i = np.array([-1], dtype='i')
        comm.Send([scher_i, MPI.INT], dest=p)
        print("Sent req scher_i= {} to p= {}".format(scher_i, p) )
    else:
      log(INFO, "rank= {} waiting for Master".format(rank) )
      sys.stdout.flush()
      while True:
        scher_i = np.empty(1, dtype='i')
        comm.Recv([scher_i, MPI.INT], source=0)
        scher_i = scher_i[0]
        if scher_i == -1:
          break
        
        scher = scher_l[scher_i]
        if sim_learners and scher_i == 0:
          if not scher.restore(ro__learning_count_m[ro], save_suffix='ro{}'.format(ro) ):
            log(ERROR, "scher.restore(ro__learning_count_m[{}] ) failed!".format(ro), schersche=scher, ro__learning_count_m=ro__learning_count_m)
            return
        log(INFO, "rank= {} will sim".format(rank), scher=scher, ro=ro)
        sys.stdout.flush()
        sim_m = sample_sim(sinfo_m, scher, lessreal_sim)
        log(INFO, "rank= {}".format(rank), sim_m=sim_m, scher=scher, ro=ro)
        
        l = np.array([sim_m['ESl'], sim_m['StdSl'], sim_m['ET'], sim_m['StdT'], sim_m['Eload'] ], dtype=np.float64)
        comm.Send([l, MPI.FLOAT], dest=0)
        sys.stdout.flush()
  if rank == 0:
    blog(scher_l=scher_l, ro_scherid_X_l_m=ro_scherid_X_l_m)

if __name__ == "__main__":
  comm = MPI.COMM_WORLD
  num_mpiprocs = comm.Get_size()
  rank = comm.Get_rank()
  
  sim_learners = True # False
  ro_l = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
  
  if sim_learners:
    scher_l = [
      RLScher(sinfo_m, mapping_m, sching_m, save_dir='save_expreplay_persist'),
      Scher(mapping_m, {'type': 'plain', 'a': 0} ),
      Scher(mapping_m, {'type': 'plain', 'a': sching_m['a'] } ) ]
  else: # Model checking
    k = BZipf(1, 5)
    b, beta = 10, 4
    L = Pareto(b, beta)
    sinfo_m.update({
      'k_rv': k,
      'lifetime_rv': L,
      'njob': 5000*N}) # 1*N
    r = 2
    l, u = k.l_l*L.l_l, 50*k.mean()*L.mean()
    d_l = [0, *np.logspace(math.log10(l), math.log10(u), 20) ] # ,6
    scher_l = [Scher_wMultiplicativeExpansion(mapping_m, {'type': 'expand_if_totaldemand_leq', 'r': r, 'threshold': d} ) for d in d_l]
  
  eval_wmpi(rank)
