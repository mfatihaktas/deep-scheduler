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
    
    self.demandperslot_rv_mean = demandperslot_rv.mean()
    
    self.prev_hop_id = None
    self.binding_time = None
    self.run_time = None
    
    self.cum_supply = 0
    self.cum_demand = 0
  
  def __repr__(self):
    return "Task[id= {}, jid= {}, type= {}]".format(self._id, self.jid, self.type_)
  
  def gen_demand(self):
    d = min(self.demandperslot_rv_mean, self.totaldemand - self.cum_demand)
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
    
    self.ntimeslots = 0
    self.avg_load = 0
  
  def straggle(self):
    while True:
      self.cap_ = self.cap*self.straggle_m['slowdown'](self.sched_load() )
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
  
  def update_avg_load(self, load):
    self.avg_load = (self.avg_load*(self.ntimeslots-1) + load)/self.ntimeslots
  
  def update_n(self):
    total_reqed, n = 0, 0
    while n < len(self.t_l) and total_reqed < self.cap:
      total_reqed += self.t_l[n]
      n += 1
    
    self.n = n
    for t in self.t_l[n]:
      if t.binding_time is None:
        t.binding_time = self.env.now
  
  def run(self):
    while True:
      yield (self.env.timeout(self.timeslot) )
      self.ntimeslots += 1
      
      t_l = self.t_l[:self.n]
      if len(t_l) == 0:
        self.update_avg_load(0)
        continue
      
      for p in t_l:
        p.gen_demand()
      
      # CPU scheduling
      cap_ = self.cap_
      sched_cap = self.sched_cap()
      total_supplytaken = 0
      for t in t_l:
        total_supplytaken += t.take_supply(min(t.reqed, t.reqed/sched_cap*cap_) )
      
      t_l_ = t_l
      while cap_ - total_supplytaken > 0.01:
        t_l_ = [t for t in t_l_ if t.cum_demand - t.cum_supply > 0.01]
        if len(t_l_) == 0:
          break
        
        supply_foreach = (cap_ - total_supplytaken)/len(t_l_)
        for t in t_l_:
          total_supplytaken += t.take_supply(supply_foreach)
      
      self.sched_load_l.append(self.sched_load() )
      
      # Check if a task is finished
      t_toremove_l = []
      for t in t_l:
        if t.cum_supply - t.totaldemand > -0.01:
          t.run_time = self.env.now - t.binding_time
          t.prev_hop_id = self._id
          self.out_c.put_c(t)
          slog(DEBUG, self.env, self, "finished", t)
        else:
          t_toremove_l.append(t)
      for t in t_toremove_l:
        self.t_l.remove(t)
        self.update_n()
  
  def put(self, t):
    slog(DEBUG, self.env, self, "received", t)
    t.ent_time = self.env.now
    self.t_l.append(t)
    self.update_n()
  
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
  def __init__(self, env, njob, nworker, wcap, straggle_m, scher, **kwargs):
    self.env = env
    self.njob = njob
    self.scher = scher
    
    self.w_l = [Worker(env, i, wcap, self, straggle_m) for i in range(nworker) ]
    
    self.store = simpy.Store(env)
    env.process(self.run() )
    
    self.njob_finished = 0
    self.store_c = simpy.Store(env)
    self.wait_for_alljobs = env.process(self.run_c() )
    
    self.jid__t_l_m = {}
    self.jid_info_m = {}
    
  def __repr__(self):
    return 'Cluster'
  
  def run(self):
    while True:
      j = yield self.store.get()
      
      s, a, w_l = self.scher.schedule(j, self.w_l)
      if a == -1:
        yield self.env.timeout(1)
        self.store.put(j)
        continue
      
      self.jid_info_m[j._id] = {'wait_time': self.env.now - j.arrival_time}
      wid_l = []
      for i, w in enumerate(w_l):
        type_ = 's' if i+1 <= j.k else 'r'
        w.put(Task(i+1, j._id, j.reqed, j.demandperslot_rv, j.totaldemand, j.k, type_) )
        wid_l.append(w._id)
      
      self.jid__t_l_m[j._id] = []
      self.jid_info_m[j._id].update({
        'expected_run_time': j.totaldemand/j.demandperslot_rv.mean(),
        'wid_l': wid_l,
        's': s, 'a': a} )
  
  def put(self, j):
    slog(DEBUG, self.env, self, "received", j)
    j.arrival_time = self.env.now
    return self.store.put(j)
  
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
          'run_time': max([t.run_time for t in self.jid__t_l_m[t.jid] ] ) } )
        self.jid__t_l_m.pop(t.jid, None)
        slog(DEBUG, self.env, self, "finished jid= {}".format(t.jid), t)
        
        self.njob_finished += 1 # for now counting only the finished jobs, ignoring the dropped ones
        if self.njob_finished >= self.njob:
          return
  
  def put_c(self, t):
    slog(DEBUG, self.env, self, "received", t)
    return self.store_c.put(t)
