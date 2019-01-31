import collections

from sim_objs import *
from mapper import *

class Task_wrelaunch():
  def __init__(self, _id, jid, reqed, lifetime, k):
    self._id = _id
    self.jid = jid
    self.reqed = reqed
    self.lifetime = lifetime
    self.k = k
    
    self.rem_lifetime = None
    self.prev_hop_id = None
    self.binding_time = None
    self.run_time = None
  
  def __repr__(self):
    return "Task_wrelaunch[id= {}, jid= {}, rem_lifetime= {}]".format(self._id, self.jid, self.rem_lifetime)

class Job_wrelaunch(object):
  def __init__(self, _id, k, n, reqed, lifetime):
    self._id = _id
    self.k = k
    self.n = n
    self.reqed = reqed
    self.lifetime = lifetime
    
    self.wait_time = None
    
  def __repr__(self):
    return "Job_wrelaunch[id= {}, k= {}, reqed= {}, lifetime= {}]".format(self._id, self.k, self.reqed, self.lifetime)

class JobGen_wrelaunch(object):
  def __init__(self, env, ar, reqed_rv, lifetime_rv, k_rv, njob, out, **kwargs):
    self.env = env
    self.ar = ar
    self.reqed_rv = reqed_rv
    self.lifetime_rv = lifetime_rv
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
      self.out.put(Job_wrelaunch(
        _id = self.nsent,
        k = k, n = k,
        reqed = self.reqed_rv.sample(),
        lifetime = self.lifetime_rv.sample() ) )

class Worker_wrelaunch():
  def __init__(self, env, _id, cap, out_c, straggle_m):
    self.env = env
    self._id = _id
    self.cap = cap
    self.out_c = out_c
    self.straggle_m = straggle_m
    
    self.t_l = []
    self.got_busy = None
    self.serv_interrupt = None
    self.add_to_serv = False
    self.cancel = False
    self.cancel_jid = None
    env.process(self.run() )
    
    self.sl = self.straggle_m['slowdown']
    self.t_load_m = {}
    
  def __repr__(self):
    return "Worker_wrelaunch[id= {}]".format(self._id)
  
  def sched_cap(self):
    if len(self.t_l) == 0:
      return 0
    return sum([t.reqed for t in self.t_l] )
  
  def nonsched_cap(self):
    return self.cap - self.sched_cap()
  
  def sched_load(self):
    return self.sched_cap()/self.cap
  
  def avg_load(self):
    t_load_m = collections.OrderedDict(sorted(self.t_load_m.items() ) )
    load_weighted_sum = 0
    _t, t, _load = 0, 1, 0
    for t, load in t_load_m.items():
      load_weighted_sum += (t - _t)*_load
      _t, _load = t, load
    return load_weighted_sum/t
  
  def run(self):
    while True:
      if len(self.t_l) == 0:
        time_gotidle = self.env.now
        self.got_busy = self.env.event()
        yield (self.got_busy)
        self.got_busy = None
        slog(DEBUG, self.env, self, "got busy!", None)
        
        self.t_load_m[time_gotidle] = 0
        self.t_load_m[self.env.now] = 0
      
      rem_lifetime_l = [t.rem_lifetime for t in self.t_l]
      serv_time = min(rem_lifetime_l)
      i_min = rem_lifetime_l.index(serv_time)
      slog(DEBUG, self.env, self, "back to serv; serv_time= {}".format(serv_time), self.t_l[i_min] )
      start_t = self.env.now
      
      self.t_load_m[self.env.now] = self.sched_load()
      self.serv_interrupt = self.env.event()
      yield (self.serv_interrupt | self.env.timeout(serv_time) )
      serv_time_ = self.env.now - start_t
      if self.add_to_serv:
        for t in self.t_l[:-1]:
          t.rem_lifetime -= serv_time_
      else:
        for t in self.t_l:
          t.rem_lifetime -= serv_time_
      # 
      if self.add_to_serv:
        slog(DEBUG, self.env, self, "new task added to serv", None)
        self.serv_interrupt = None
        self.add_to_serv = False
      elif self.cancel:
        for t in self.t_l:
          if t.jid == self.cancel_jid:
            slog(DEBUG, self.env, self, "cancelled task in serv", t)
            self.t_l.remove(t)
            break
        self.serv_interrupt = None
        self.cancel = False
      elif self.relaunch:
        for t in self.t_l:
          if t.jid == self.relaunch_jid:
            slog(DEBUG, self.env, self, "relaunched task in serv", t)
            t.rem_lifetime = t.lifetime*self.sl(self.sched_load() )
            break
        self.serv_interrupt = None
        self.relaunch = False
      else:
        t = self.t_l.pop(i_min)
        slog(DEBUG, self.env, self, "serv done", t)
        
        t.run_time = self.env.now - t.binding_time
        t.prev_hop_id = self._id
        self.out_c.put_c(t)
        slog(DEBUG, self.env, self, "finished", t)
  
  def put(self, t):
    slog(DEBUG, self.env, self, "put:: starting;", t)
    avail_cap = self.nonsched_cap()
    
    _l = len(self.t_l)
    t.binding_time = self.env.now
    t.rem_lifetime = t.lifetime*self.sl(self.sched_load() )
    self.t_l.append(t)
    if _l == 0:
      self.got_busy.succeed()
    else:
      self.add_to_serv = True
      self.serv_interrupt.succeed()
    slog(DEBUG, self.env, self, "binded, njob= {}".format(len(self.t_l) ), t)
  
  def put_c(self, m):
    slog(DEBUG, self.env, self, "received", m)
    if m['message'] == 'remove':
      self.cancel = True
      self.cancel_jid = m['jid']
      self.serv_interrupt.succeed()
    elif m['message'] == 'relaunch':
      self.relaunch = True
      self.relaunch_jid = m['jid']
      self.serv_interrupt.succeed()
    else:
      log(ERROR, "Unrecognized message;", m=m)

class Cluster_wrelaunch():
  def __init__(self, env, njob, nworker, wcap, straggle_m, scher, **kwargs):
    self.env = env
    self.njob = njob
    self.nworker = nworker
    self.wcap = wcap
    self.straggle_m = straggle_m
    self.scher = scher
    
    self.w_l = [Worker_wrelaunch(env, i, wcap, self, straggle_m) for i in range(nworker) ]
    
    self.store = simpy.Store(env)
    env.process(self.run() )
    
    self.njob_finished = 0
    self.store_c = simpy.Store(env)
    self.wait_for_alljobs = env.process(self.run_c() )
    
    self.got_ajob_torelaunch = None
    env.process(self.run_relaunch() )
    
    self.jid__t_l_m = {}
    self.jid_info_m = {}
    
  def __repr__(self):
    return 'Cluster_wrelaunch'
  
  def run(self):
    while True:
      j = yield self.store.get()
      
      while True:
        j.wait_time = self.env.now - j.arrival_time
        
        # a would be relaunch time
        s, a, w_l = self.scher.schedule(j, self.w_l, self)
        if a == -1:
          slog(DEBUG, self.env, self, "a= -1", j)
          yield self.env.timeout(0.01)
        else:
          break
      
      self.jid_info_m[j._id] = {'wait_time': self.env.now - j.arrival_time}
      wid_l = []
      for i, w in enumerate(w_l):
        w.put(Task_wrelaunch(i+1, j._id, j.reqed, j.lifetime, j.k) )
        yield self.env.timeout(0.0001)
        wid_l.append(w._id)
      
      self.jid__t_l_m[j._id] = []
      self.jid_info_m[j._id].update({
        'expected_run_time': j.lifetime,
        'wid_l': wid_l,
        's': s, 'a': a,
        'relaunch_time': self.env.now + a if a is not None else None} )
      if self.got_ajob_torelaunch is not None and a is not None:
        self.got_ajob_torelaunch.succeed()
      self.run_relaunch_interrupt_flag = True
      self.run_relaunch_interrupt.succeed()
  
  def put(self, j):
    slog(DEBUG, self.env, self, "received", j)
    j.arrival_time = self.env.now
    return self.store.put(j)
  
  def run_relaunch(self):
    while True:
      dur_jid_l = []
      for jid in self.jid__t_l_m:
        t = self.jid_info_m['relaunch_time']
        if t is not None:
          dur_jid_l.append((t - self.env.now, jid) )
      
      if len(dur_jid_l) == 0:
        self.got_ajob_torelaunch = self.env.event()
        yield (self.got_ajob_torelaunch)
        self.got_ajob_torelaunch = None
        slog(DEBUG, self.env, self, "got a job to relaunch!", None)
      
      dur, jid = min(dur_jid_l)
      self.run_relaunch_interrupt = self.env.event()
      yield (self.run_relaunch_interrupt | self.env.timeout(dur) )
      if self.run_relaunch_interrupt_flag:
        self.run_relaunch_interrupt = None
        self.run_relaunch_interrupt_flag = False
      else:
        t_l = self.jid__t_l_m[jid]
        wrecvedfrom_id_l = [t.prev_hop_id for t in t_l]
        wsentto_id_l = self.jid_info_m[jid]['wid_l']
        for w in self.w_l:
          if w._id in wsentto_id_l and w._id not in wrecvedfrom_id_l:
            w.put_c({'message': 'relaunch', 'jid': jid} )
  
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
        
        if t.jid <= self.njob:
          self.njob_finished += 1
          # log(WARNING, "job completion;", jid=t.jid, njob=self.njob, njob_finished=self.njob_finished)
          if self.njob_finished >= self.njob:
            return
  
  def put_c(self, t):
    slog(DEBUG, self.env, self, "received", t)
    return self.store_c.put(t)

# ############################################  Scher  ########################################### #
class Scher_wrelaunch(object):
  def __init__(self, mapping_m, sching_m):
    self.sching_m = sching_m
    self.mapper = Mapper(mapping_m)
    
    # self._id = 'Scher_wrelaunch'
  
  def __repr__(self):
    return 'Scher[sching_m={}, mapper= {}]'.format(self.sching_m, self.mapper)
  
  def schedule(self, j, w_l, cluster):
    w_l = self.mapper.worker_l(j, w_l)
    if len(w_l) < j.k:
      return None, -1, None
    
    a = self.sching_m['relaunch_time'](j)
    return None, a, w_l[:j.k]
