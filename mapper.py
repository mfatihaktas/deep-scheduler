from operator import itemgetter

# ########################################  Mapper  ########################################### #
class Mapper(object):
  def __init__(self, mapping_m):
    self.mapping_m = mapping_m
    
    if self.mapping_m['type'] == 'packing':
      self.worker_l = lambda j, w_l: self.worker_l_w_packing(j, w_l)
    elif self.mapping_m['type'] == 'spreading':
      self.worker_l = lambda j, w_l: self.worker_l_w_spreading(j, w_l)
  
  def __repr__(self):
    return 'Mapper[mapping_m= {}]'.format(self.mapping_m)
  
  def worker_l_w_packing(self, job, w_l):
    w_l_ = []
    for w in w_l:
      if job.reqed <= w.nonsched_cap():
        w_l_.append((w, w.sched_load() ) )
    return w_l_
  
  def worker_l_w_spreading(self, job, w_l):
    w_load_l = []
    for w in w_l:
      if job.reqed <= w.nonsched_cap():
        w_load_l.append((w, w.sched_load() ) )
    w_load_l.sort(key=itemgetter(1) )
    return [w for w, _ in w_load_l]