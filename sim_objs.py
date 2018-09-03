import math, random, simpy, pprint
from operator import itemgetter

from scheduler import *
from rvs import *
from log_utils import *

# #######################################  Task  ######################################## #
class Task(object):
  def __init__(self, _id, jid, reqed, demandperslot_rv, totaldemand, k, type_=None):
    self._id = _id
    self.jid = jid
    self.reqed = reqed
    self.demandperslot_rv = demandperslot_rv
    self.totaldemand = totaldemand
    self.k = k
    self.type_ = type_ # 's': systematic, 'r': redundant
    
    self.prev_hop_id = None
    self.binding_time = None
    self.runtime = None
    
    self.cum_supply = 0
    self.cum_demand = 0
  
  def __repr__(self):
    return "Task[id= {}, jid= {}, type= {}]".format(self._id, self.jid, self.type_)
  
  def __repr__(self):
    return "Pod[id= {}]".format(self._id)
  
  def gen_demand(self):
    d = min(self.demandperslot_rv.sample(), self.totaldemand - self.cum_demand)
    self.cum_demand += d
    return d
  
  def take_supply(self, s):
    s_ = min(self.cum_demand - self.cum_supply, s)
    self.cum_supply += s_
    return s_

class Job(object):
  def __init__(self, _id, k, n, demandperslot_rv, totaldemand):
    self._id = _id
    self.k = k
    self.n = n
    self.demandperslot_rv = demandperslot_rv
    self.totaldemand = totaldemand
    
    self.reqed = self.demandperslot_rv.mean()
  
  def __repr__(self):
    return "Job[id= {}]".format(self._id)

class JobGen(object):
  def __init__(self, env, ar, demandperslot_mean_rv, totaldemand_rv, k_rv, njob, out, **kwargs):
    self.env = env
    self.ar = ar
    self.demandperslot_mean_rv = demandperslot_mean_rv
    self.totaldemand_rv = totaldemand_rv
    self.k_rv = k_rv
    self.njob = njob
    self.out = out
    
    self.nsent = 0
    
    self.action = self.env.process(self.run_poisson() )
  
  def run_poisson(self):
    while 1:
      yield self.env.timeout(random.expovariate(self.ar) )
      self.nsent += 1
      k = self.k_rv.sample()
      demandmean = self.demandperslot_mean_rv.sample()
      coeff_var = 0.7
      self.out.put(Job(
        _id = self.nsent,
        k = k, n = k,
        demandperslot_rv = TNormal(demandmean, demandmean*coeff_var),
        totaldemand = self.totaldemand_rv.sample() ) )
      
      # if self.nsent >= self.njob:
      #   return

# #########################################  Worker  ############################################# #
class Worker(object):
  def __init__(self, env, _id, cap, out_c, straggle_m):
    self.env = env
    self._id = _id
    self.cap = cap
    self.out_c = out_c
    self.straggle_m = straggle_m
    
    self.cap_ = self.cap
    
    self.timeslot = 1
    self.t_l = []
    env.process(self.run() )
    env.process(self.straggle() )
    
    self.sched_load_l = []
  
  def straggle(self):
    while True:
      self.cap_ = self.cap*self.straggle_m['slowdown_rv'].sample()
      yield (self.env.timeout(self.straggle_m['straggle_dur_rv'].sample() ) )
      
      self.cap_ = self.cap
      yield (self.env.timeout(self.straggle_m['normal_dur_rv'].sample() ) )
  
  def __repr__(self):
    return "Worker[id= {}]".format(self._id)
  
  def sched_cap(self):
    if len(self.t_l) == 0:
      return 0
    return sum([t.reqed for t in self.t_l] )
  
  def nonsched_cap(self):
    return self.cap - self.sched_cap()
  
  def sched_load(self):
    return self.sched_cap()/self.cap
  
  def run(self):
    while True:
      yield (self.env.timeout(self.timeslot) )
      if len(self.t_l) == 0:
        self.sched_load_l.append(0)
        continue
      
      for p in self.t_l:
        p.gen_demand()
      
      # CPU scheduling
      cap_ = self.cap_
      sched_cap = self.sched_cap()
      total_supplytaken = 0
      for t in self.t_l:
        total_supplytaken += t.take_supply(min(t.reqed, t.reqed/sched_cap*cap_) )
      
      t_l_ = self.t_l
      while cap_ - total_supplytaken > 0.01:
        t_l_ = [t for t in t_l_ if t.cum_demand - t.cum_supply > 0.01]
        if len(t_l_) == 0:
          break
        
        supply_foreach = (cap_ - total_supplytaken)/len(t_l_)
        for t in t_l_:
          total_supplytaken += t.take_supply(supply_foreach)
      
      self.sched_load_l.append(self.sched_load() )
      
      # Check if a task is finished
      t_l_ = []
      for t in self.t_l:
        if t.cum_supply - t.totaldemand > -0.01:
          t.runtime = self.env.now - t.bindingt
          self.out_c.put_c(t)
          slog(DEBUG, self.env, self, "finished", t)
        else:
          t_l_.append(t)
      self.t_l = t_l_
  
  def put(self, t):
    t.bindingt = self.env.now
    self.t_l.append(t)
    slog(DEBUG, self.env, self, "binded, njob= {}".format(len(self.t_l) ), t)
  
  def put_c(self, m):
    slog(DEBUG, self.env, self, "received", m)
    
    if m['message'] == 'remove':
      jid = m['jid']
      ti = None
      for i, t in enumerate(self.t_l):
        if t.jid == jid:
          ti = i
      if ti is not None:
        slog(DEBUG, self.env, self, "removing", self.t_l[ti] )
        del self.t_l[ti]
    else:
      log(ERROR, "Unrecognized message;", m=m)

# #########################################  Cluster  ############################################ #
class Cluster(object):
  def __init__(self, env, njob, nworker, wcap, straggle_m, mapper, scher, max_exprate=1, **kwargs):
    self.env = env
    self.njob = njob
    self.mapper = mapper
    self.scher = scher
    self.max_exprate = max_exprate
    
    self.w_l = [Worker(env, i, wcap, self, straggle_m) for i in range(nworker) ]
    
    self.njob_collected = 0
    self.store_c = simpy.Store(env)
    self.wait_for_alljobs = env.process(self.run_c() )
    
    self.jid__t_l_m = {}
    self.jid_info_m = {}
    
  def __repr__(self):
    return 'Cluster'
  
  def put(self, job):
    slog(DEBUG, self.env, self, "received", job)
    w_load_l = self.mapper.worker_load_l(job, self.w_l)
    if len(w_load_l) < job.k:
      self.jid_info_m[job._id] = {'fate': 'dropped'}
      return
    
    n_max = min(int(self.max_exprate*job.k), len(w_load_l) )
    s, a = self.scher.schedule(job, [l for _, l in w_load_l[:n_max] ] )
    job.n = int(job.k*(a + 1) )
    
    wid_l = []
    for i, w in enumerate([w for w, _ in w_load_l[:job.n] ] ):
      type_ = 's' if i+1 <= job.k else 'r'
      w.put(Task(i+1, job._id, job.reqed, job.demandperslot_rv, job.totaldemand, job.k, type_) )
      wid_l.append(w._id)
    
    self.jid__t_l_m[job._id] = []
    self.jid_info_m[job._id] = {
      'expected_lifetime': job.totaldemand/job.demandperslot_rv.mean(),
      'wid_l': wid_l,
      's': s, 'a': a}
  
  def run_c(self):
    while True:
      t = yield self.store_c.get()
      try:
        self.jid__t_l_m[t.jid].append(t)
      except KeyError: # may happen due to a task completion after the corresponding job finishes
        continue
      
      t_l = self.jid__t_l_m[t.jid]
      if len(t_l) > t.k:
        log(ERROR, "len(t_l)= {} > k= {}".format(len(t_l), t.k) )
      elif len(t_l) < t.k:
        continue
      else:
        t_l = self.jid__t_l_m[t.jid]
        wrecvedfrom_id_l = [t.prev_hop_id for t in t_l]
        wsentto_id_l = self.jid_info_m[t.jid]['wid_l']
        for w in self.w_l:
          if w._id in wsentto_id_l and w._id not in wrecvedfrom_id_l:
            w.put_c({'message': 'remove', 'jid': t.jid} )
        
        self.jid_info_m[t.jid].update({
          'fate': 'finished',
          'runtime': max([t.runtime for t in self.jid__t_l_m[t.jid] ] ) } )
        self.jid__t_l_m.pop(t.jid, None)
        slog(DEBUG, self.env, self, "finished jid= {}".format(t.jid), t)
        
        self.njob_collected += 1 # for now counting only the finished jobs, ignoring the dropped ones
        if self.njob_collected >= self.njob:
          return
  
  def put_c(self, t):
    slog(DEBUG, self.env, self, "received", t)
    return self.store_c.put(t)
    
# ########################################  Mapper  ########################################### #
class Mapper(object):
  def __init__(self, mapping_m):
    self.mapping_m = mapping_m
    
    if self.mapping_m['type'] == 'packing':
      self.worker_load_l = lambda p, w_l: self.worker_load_l_w_packing(p, w_l)
    elif self.mapping_m['type'] == 'spreading':
      self.worker_load_l = lambda p, w_l: self.worker_load_l_w_spreading(p, w_l)
  
  def __repr__(self):
    return "Mapper[mapping_m=\n {}]".format(self.mapping_m)
  
  def worker_load_l_w_packing(self, job, w_l):
    w_l_ = []
    for w in w_l:
      if job.reqed <= w.nonsched_cap():
        w_l_.append(w)
    if len(w_load_l) < job.n:
      return None
    return w_l_[:job.n]
  
  def worker_load_l_w_spreading(self, job, w_l):
    w_load_l = []
    for w in w_l:
      if job.reqed <= w.nonsched_cap():
        w_load_l.append((w, w.sched_load() ) )
    # for now assuming all n need to be dispatched to separate workers
    w_load_l.sort(key=itemgetter(1) )
    return w_load_l
