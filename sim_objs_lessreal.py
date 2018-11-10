from sim_objs import *

class Task_LessReal():
  def __init__(self, _id, jid, reqed, lifetime, k, type_=None):
    self._id = _id
    self.jid = jid
    self.reqed = reqed
    self.lifetime = lifetime
    self.k = k
    self.type_ = type_
    
    self.rem_lifetime = None
    self.prev_hop_id = None
    self.binding_time = None
    self.run_time = None
  
  def __repr__(self):
    return "Task_LessReal[id= {}, jid= {}, rem_lifetime= {}]".format(self._id, self.jid, self.rem_lifetime)

class Worker_LessReal():
  def __init__(self, env, _id, cap, out_c, straggle_m):
    self.env = env
    self._id = _id
    self.cap = cap
    self.out_c = out_c
    self.straggle_m = straggle_m
    
    self.t_l = []
    self.got_busy = None
    self.sinterrupt = None
    self.add_to_serv = False
    self.cancel = False
    self.cancel_jid = None
    env.process(self.run() )
    
    self.sl = self.straggle_m['slowdown']
    
  def __repr__(self):
    return "Worker_LessReal[id= {}]".format(self._id)
  
  def sched_cap(self):
    if len(self.t_l) == 0:
      return 0
    return sum([t.reqed for t in self.t_l] )
  
  def nonsched_cap(self):
    return self.cap - self.sched_cap()
  
  def sched_load(self):
    return self.sched_cap()/self.cap
  
  def avg_load(self):
    return self.sched_load()
  
  def run(self):
    while True:
      if len(self.t_l) == 0:
        self.got_busy = self.env.event()
        yield (self.got_busy)
        self.got_busy = None
        slog(DEBUG, self.env, self, "got busy!", None)
      
      rem_lifetime_l = [t.rem_lifetime for t in self.t_l]
      serv_time = min(rem_lifetime_l)
      i_min = rem_lifetime_l.index(serv_time)
      # slog(DEBUG, self.env, self, "back to serv; serv_time= ".format(serv_time), None)
      start_t = self.env.now
      
      self.sinterrupt = self.env.event()
      yield (self.sinterrupt | self.env.timeout(serv_time) )
      serv_time_ = self.env.now - start_t
      for t in self.t_l:
        t.rem_lifetime -= serv_time_
      
      if self.add_to_serv:
        # slog(DEBUG, self.env, self, "new task added to serv", None)
        self.sinterrupt = None
        self.add_to_serv = False
      elif self.cancel:
        for t in self.t_l:
          if t.jid == self.cancel_jid:
            # slog(DEBUG, self.env, self, "cancelled task in serv", t)
            self.t_l.remove(t)
            break
        self.sinterrupt = None
        self.cancel = False
      else:
        t = self.t_l.pop(i_min)
        # slog(DEBUG, self.env, self, "serv done", t)
        
        t.run_time = self.env.now - t.binding_time
        t.prev_hop_id = self._id
        self.out_c.put_c(t)
        slog(DEBUG, self.env, self, "finished", t)
  
  def put(self, t):
    avail_cap = self.nonsched_cap()
    if t.type_ == 's' and t.reqed > avail_cap:
      tred_l = [t for t in self.t_l if t.type_ == 'r']
      i = 0
      while i < len(tred_l) and avail_cap < t.reqed:
        tred = tred_l[i]
        avail_cap += tred.reqed
        self.t_l.remove(tred)
        i += 1
      if avail_cap < t.reqed:
        slog(ERROR, self.env, self, "could not bind", t)
        return
    elif t.type_ == 'r' and t.reqed > avail_cap:
      return
    
    _l = len(self.t_l)
    t.binding_time = self.env.now
    t.rem_lifetime = t.lifetime/self.sl(self.sched_load() )
    self.t_l.append(t)
    if _l == 0:
      self.got_busy.succeed()
    else:
      self.add_to_serv = True
      self.sinterrupt.succeed()
    slog(DEBUG, self.env, self, "binded, njob= {}".format(len(self.t_l) ), t)
  
  def put_c(self, m):
    slog(DEBUG, self.env, self, "received", m)
    if m['message'] == 'remove':
      self.cancel = True
      self.cancel_jid = m['jid']
      self.sinterrupt.succeed()
    else:
      log(ERROR, "Unrecognized message;", m=m)

class Cluster_LessReal(Cluster):
  def __init__(self, env, njob, nworker, wcap, straggle_m, scher, **kwargs):
    super().__init__()(env, njob, nworker, wcap, straggle_m, scher, **kwargs)
  
  def __repr__(self):
    return 'Cluster_LessReal'
  
  def run(self):
    while True:
      j = yield self.store.get()
      
      while True:
        s, a, w_l = self.scher.schedule(j, self.w_l, self)
        if a == -1:
          slog(DEBUG, self.env, self, "a = -1", j)
          yield self.env.timeout(0.1)
        else:
          break
      
      self.jid_info_m[j._id] = {'wait_time': self.env.now - j.arrival_time}
      wid_l = []
      lifetime = j.totaldemand/j.demandperslot_rv.mean()
      for i, w in enumerate(w_l):
        type_ = 's' if i < j.k else 'r'
        w.put(Task_LessReal(i+1, j._id, j.reqed, lifetime, j.k, type_) )
        wid_l.append(w._id)
      
      self.jid__t_l_m[j._id] = []
      self.jid_info_m[j._id].update({
        'expected_run_time': j.totaldemand/j.demandperslot_rv.mean(),
        'wid_l': wid_l,
        's': s, 'a': a} )
    