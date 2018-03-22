import math, random, simpy, pprint
from patch import *
from rvs import Pareto

# ************************  Essentials for Jobs with multiple Tasks  ***************************** # 
class Task(object):
  def __init__(self, jid, k, size, type_=None, L=None, remaining=None):
    self.jid = jid
    self.k = k
    self.size = size
    self.type_ = type_
    self.L = L
    self.remaining = remaining
    
    self.prev_hop_id = None
    self.ent_time = None
  
  def __repr__(self):
    return "Task[jid= {}, size= {}, type= {}]".format(self.jid, self.size, self.type_)
  
  def deep_copy(self):
    t = Task(self.jid, self.k, self.size, self.remaining)
    t.prev_hop_id = self.prev_hop_id
    t.ent_time = self.ent_time
    return t

class Job(object):
  def __init__(self, _id, k, size, n=0):
    self._id = _id
    self.k = k
    self.size = size
    self.n = n
    self.type_ = None
    
    self.prev_hop_id = None
  
  def __repr__(self):
    return "Job[id= {}]".format(self._id)
  
  def deep_copy(self):
    j = Job(self._id, self.k, self.size, self.n)
    j.prev_hop_id = self.prev_hop_id
    return j

class JG(object): # Job Generator
  def __init__(self, env, ar, k_dist, size_dist, ntosend, type_='poisson'):
    self.env = env
    self.ar = ar
    self.k_dist = k_dist
    self.size_dist = size_dist
    self.ntosend = ntosend
    self.type_ = type_
    
    self.nsent = 0
    self.out = None
    
  def init(self):
    if self.type_ == 'poisson':
      self.action = self.env.process(self.run_poisson() )
    elif self.type_ == 'selfsimilar':
      self.action = self.env.process(self.run_selfsimilar() )
  
  def run_poisson(self):
    while 1:
      # yield self.env.timeout(random.expovariate(self.ar) )
      yield self.env.timeout(1/self.ar)
      self.out.put(Job(self.nsent, self.k_dist.gen_sample(), self.size_dist.gen_sample() ) )
      
      self.nsent += 1
      if self.nsent >= self.ntosend:
        return
  
  def run_selfsimilar(self):
    stime = Pareto(1, 1.2)
    arepoch_l = [0]
    for i in range(self.ntosend - 1):
      arepoch_l.append(arepoch_l[-1] + random.expovariate(self.ar) )
    arepoch_l = sorted([e + stime.gen_sample() for e in arepoch_l] )
    
    for i in range(1, self.ntosend):
      yield self.env.timeout(arepoch_l[i] - arepoch_l[i-1] )
      self.out.put(Job(self.nsent, self.k_dist.gen_sample(), self.size_dist.gen_sample() ) )
      
      self.nsent += 1
      if self.nsent >= self.ntosend:
        return

class FCFS(object):
  def __init__(self, _id, env, sl_dist, out=None, out_c=None, L=None):
    self._id = _id
    self.env = env
    self.sl_dist = sl_dist
    self.out, self.out_c = out, out_c
    self.L = L
    
    self.t_l = []
    self.t_inserv = None
    self.got_busy = None
    self.cancel_flag = False
    self.cancel = None
    
    self.lt_l = []
    # self.sl_l = []
    
    self.action = env.process(self.run() )
    # self.idle_t = 0
    self.busy_t = 0
  
  def __repr__(self):
    return "FCFS[id= {}]".format(self._id)
  
  def length(self):
    # return len(self.t_l) + (self.t_inserv is not None)
    return sum([t.type_ != 'r' for t in self.t_l] ) + (self.t_inserv is not None and self.t_inserv.type_ != 'r')
    # return sum([t.size for t in self.t_l if t.type_ != 'r'] ) + (self.t_inserv.size if (self.t_inserv is not None and self.t_inserv.type_ != 'r') else 0)
  
  def run(self):
    while True:
      if len(self.t_l) == 0:
        # idle_start_t = self.env.now
        self.got_busy = self.env.event()
        yield (self.got_busy)
        self.got_busy = None
        sim_log(DEBUG, self.env, self, "got busy!", None)
        # self.idle_t += self.env.now - idle_start_t
      self.t_inserv = self.t_l.pop(0)
      if self.t_inserv.type_ == 'r':
        # if self.L is not None:
        #   if (self.length() - 1) > self.L:
        #     self.t_inserv = None
        #     continue
        #   else:
        #     if self.out_c is not None:
        #       self.out_c.put_c({'m': 'r', 'jid': self.t_inserv.jid} )
        # elif self.L is None:
        #   if (self.length() - 1) > self.t_inserv.L:
        #     self.t_inserv = None
        #     continue
        if self.length() > 0:
          self.t_inserv = None
          continue
        elif self.out_c is not None:
          self.out_c.put_c({'m': 'r', 'jid': self.t_inserv.jid} )
      
      self.cancel = self.env.event()
      clk_start_time = self.env.now
      st = self.t_inserv.size * self.sl_dist.gen_sample()
      sim_log(DEBUG, self.env, self, "starting {}s-clock on ".format(st), self.t_inserv)
      busy_start_t = self.env.now
      yield (self.cancel | self.env.timeout(st) )
      
      if self.cancel_flag:
        sim_log(DEBUG, self.env, self, "cancelled clock on ", self.t_inserv)
        self.cancel_flag = False
        # yield self.env.timeout(1)
      else:
        sim_log(DEBUG, self.env, self, "serv done in {}s on ".format(self.env.now-clk_start_time), self.t_inserv)
        if self.t_inserv.type_ != 'r':
          lt = self.env.now - self.t_inserv.ent_time
          self.lt_l.append(lt)
          # self.sl_l.append(lt/self.t_inserv.size)
          self.busy_t += self.env.now - busy_start_t
      
        self.t_inserv.prev_hop_id = self._id
        if self.out is not None:
          self.out.put(self.t_inserv)
        elif self.out_c is not None:
          self.out_c.put_c({'jid': self.t_inserv.jid} )
      self.t_inserv = None
  
  def put(self, t):
    sim_log(DEBUG, self.env, self, "recved", t)
    _l = len(self.t_l)
    t.ent_time = self.env.now
    self.t_l.append(t)
    if self.got_busy is not None and _l == 0:
      self.got_busy.succeed()
    elif _l == 1 and self.t_inserv.type_ == 'r' and t.type_ != 'r':
      self.cancel_flag = True
      self.cancel.succeed()
  
  def put_c(self, m):
    sim_log(DEBUG, self.env, self, "recved", m)
    
    # if m['m'] == 'cancel':
    jid = m['jid']
    for t in self.t_l:
      if t.jid == jid:
        self.t_l.remove(t)
    if self.t_inserv is not None and jid == self.t_inserv.jid:
      self.t_inserv = None
      self.cancel_flag = True
      self.cancel.succeed()

class JQ(object):
  def __init__(self, env, in_qid_l, out_c):
    self.env = env
    self.in_qid_l = in_qid_l
    self.out_c = out_c
    
    self.jid__t_l_map = {}
    self.deped_jid_l = []
    
    self.store = simpy.Store(env)
    self.action = env.process(self.run() )
    
    self.jid_r_m = {}
    
  def __repr__(self):
    return "JQ[in_qid_l= {}]".format(self.in_qid_l)
  
  def run(self):
    while True:
      t = (yield self.store.get() )
      if t.jid in self.deped_jid_l: # Redundant tasks of a job may be received
        sim_log(DEBUG, self.env, self, "already recved", t)
        continue
      
      if t.jid not in self.jid__t_l_map:
        self.jid__t_l_map[t.jid] = []
      self.jid__t_l_map[t.jid].append(t.deep_copy() )
      
      t_l = self.jid__t_l_map[t.jid]
      if len(t_l) > t.k:
        log(ERROR, "len(t_l)= {} > k= {}".format(len(t_l), t.k) )
      elif len(t_l) < t.k:
        continue
      else:
        sim_log(DEBUG, self.env, self, "completed jid= {}".format(t.jid), t)
        self.jid__t_l_map.pop(t.jid, None)
        self.deped_jid_l.append(t.jid)
        self.out_c.put_c({'jid': t.jid, 'm': 'jdone', 'deped_from': [t.prev_hop_id for t in t_l] } )
  
  def put(self, t):
    sim_log(DEBUG, self.env, self, "recved", t)
    return self.store.put(t)
  
  def put_c(self, m):
    sim_log(DEBUG, self.env, self, "recved", m)
    if m['m'] == 'r':
      jid = m['jid']
      if jid not in self.jid_r_m:
        self.jid_r_m[jid] = 0
      self.jid_r_m[jid] += 1
