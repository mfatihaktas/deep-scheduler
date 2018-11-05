import random

from sim_objs import *
from scheduler import *

# ###################################  Cluster_wExpReplay  ####################################### #
class Queue(object):
  def __init__(self, size):
    self.size = size
    
    self.l = []
  
  def put(self, e):
    if len(self.l) == size:
      self.l.pop(0)
    self.l.append(e)
  
def ExpQueue(Queue):
  def __init__(self, buffer_size, batch_size):
    super().__init__(buffer_size)
    self.batch_size = batch_size
  
  def sample_batch():
    return random.sample(list(range(len(mylist) ) ), self.batch_size)
  
class Cluster_wExpReplay(object):
  def __init__(self, env, nworker, wcap, straggle_m, scher, M, **kwargs):
    super().__init__(env, float('Inf'), nworker, wcap, straggle_m, scher, **kwargs)
    self.M = M # number of (s, a, r, snext, anext) to collect per training
    
    # self.waitfor_jid_l = []
    self.waitforjid_begin = 1
    self.waitforjid_end = M
    self.waitfor_njob = M
    self.last_sched_jid = None
    
    self.exp_q = ExpQueue(20*M, 2*M)
  
  def __repr__(self):
    return 'wExpReplay!_' + super()
  
  def run(self):
    while True:
      j = yield self.store.get()
      
      while True:
        s, a, w_l = self.scher.schedule(j, self.w_l)
        if a == -1:
          yield self.env.timeout(0.1)
        else:
          break
      
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
      self.last_sched_jid = j._idzh
  
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
        
        ## Learning
        if self.waitforjid_begin <= t.jid <= self.waitforjid_end:
          self.waitfor_njob -= 1
          if self.waitfor_njob == 0:
            for jid in range(self.waitforjid_begin, self.waitforjid_end):
              jinfo_m = self.jid_info_m[t.jid]
              sl = jinfo_m['runtime']/jinfo_m['expected_run_time']
              r = reward(sl)
              
              jnextinfo_m = self.jid_info_m[t.jid + 1]
              self.exp_q.put((jinfo_m['s'], jinfo_m['a'], r, jnextinfo_m['s'], jnextinfo_m['a'] ) )
            # Train
            scher.learner.train_w_sarsa_l(self.exp_q.sample_batch() )
            
            self.waitforjid_begin = self.last_sched_jid + 1
            self.waitforjid_end = self.waitforjid_begin + M-1
            self.waitfor_njob = M

def reward(slowdown):
  return -slowdown

def learn_w_experience_replay():
  scher = RLScher(sinfo_m, mapping_m, sching_m)
  N, T, s_len = scher.N, scher.T, scher.s_len
  log(INFO, "starting;", rank=rank, scher=scher)
  
  env = simpy.Environment()
  cl = Cluster(env, scher=scher, **sinfo_m)
  jg = JobGen(env, out=cl, **sinfo_m)
  env.run(until=cl.wait_for_alljobs)
  
  T = sinfo_m['njob']
  t_s_l, t_a_l, t_r_l, t_sl_l = np.zeros((T, scher.s_len)), np.zeros((T, 1)), np.zeros((T, 1)), np.zeros((T, 1))
  
  # t = 0
  # for jid, jinfo_m in sorted(cl.jid_info_m.items(), key=itemgetter(0) ):
  #   # blog(t=t, jid=jid, jinfo_m=jinfo_m)
  #   if 'fate' in jinfo_m and jinfo_m['fate'] == 'finished':
  for t in range(T):
    jinfo_m = cl.jid_info_m[t+1]
    t_s_l[t, :] = jinfo_m['s']
    t_a_l[t, :] = jinfo_m['a']
    sl = (jinfo_m['wait_time'] + jinfo_m['run_time'] )/jinfo_m['expected_run_time']
    t_r_l[t, :] = reward(sl)
    t_sl_l[t, :] = sl
  
  return t_s_l, t_a_l, t_r_l, t_sl_l, \
         np.mean([w.avg_load for w in cl.w_l] )