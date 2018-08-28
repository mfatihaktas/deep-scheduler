import math, random, simpy, pprint

from log_utils import *

# #######################################  Task  ######################################## #
class Task(object):
  def __init__(self, _id, jid, demandperslot_rv, totaldemand, k, type_=None):
    self._id = _id
    self.jid = jid
    self.demandperslot_rv = demandperslot_rv
    self.totaldemand = totaldemand
    self.k = k
    self.type_ = type_ # 's': systematic, 'r': redundant
    
    self.prev_hop_id = None
    self.binding_time = None
    self.runtime = None
  
  def __repr__(self):
    return "Task[id= {}, jid= {}, type= {}]".format(self._id, self.jid, self.type_)

class Job(object):
  def __init__(self, _id, k, n, demandperslot_rv, totaldemand):
    self._id = _id
    self.k = k
    self.n = n
    self.demandperslot_rv = demandperslot_rv
    self.totaldemand = totaldemand
  
  def __repr__(self):
    return "Job[id= {}]".format(self._id)

class JobGen(object):
  def __init__(self, env, ar, demandperslot_mean_rv, totaldemand_rv, k_rv, njobs):
    self.env = env
    self.ar = ar
    self.k_rv = k_rv
    self.size_dist = size_dist
    self.njobs = njobs
    self.type_ = type_
    
    self.nsent = 0
    self.out = None
    
  def init(self):
    self.action = self.env.process(self.run_poisson() )
  
  def run_poisson(self):
    while 1:
      yield self.env.timeout(random.expovariate(self.ar) )
      self.nsent += 1
      k = self.k_rv.gen_sample()
      demandmean = self.demandperslot_mean_rv.sample()
      coeff_var = 0.7
      self.out.put(Job(
        _id = self.nsent,
        k = k, n = k,
        demandperslot_rv = TNormal(demandmean, demandmean*coeff_var),
        totaldemand = self.totaldemand_rv.sample() ) )
      
      if self.nsent >= self.njobs:
        return
