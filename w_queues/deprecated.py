# *************************  Learning shortest-q scheduling on the fly  ************************** #
# Learning from single trajectory at a time ended up having too much variance
class LearningShortestQ_wSingleTraj(object):
  def __init__(self, env, n):
    self.env = env
    self.n = n
    
    slowdown_dist = Dolly() # DUniform(1, 1)
    self.q_l = [FCFS(i, env, slowdown_dist, out=self) for i in range(self.n) ]
    self.jid_info_m = {}
    
    self.store = simpy.Store(env)
    self.action = env.process(self.run() )
    # 
    self.scher = DeepScher(n, n)
    
    self.training_len = 1000
    self.training_on = False
    self.jid_to_wait_l = []
    
    self.action_correctness_l = []
  
  def reset_training(self, jid):
    log(WARNING, "started with jid= {}".format(jid) )
    self.training_on = True
    self.jid_head_of_training = jid
    self.jid_tail_of_training = jid + self.training_len-1
    self.jid_to_wait_l = list(range(self.jid_head_of_training, self.jid_tail_of_training+1) )
    
    self.action_correctness_l.clear()
  
  def __repr__(self):
    return "LearningShortestQ_wSingleTraj[n= {}]".format(self.n)
  
  def state(self):
    return np.mean(np.array([q.length() for q in self.q_l] ) )
  
  def run(self):
    while True:
      j = (yield self.store.get() )
      
      s = self.state()
      # print("s= {}".format(s) )
      a = self.scher.get_random_action(s)
      
      a_ = self.scher.get_max_action(s)
      c = 1 if (s[a_] - min(s) ) < 0.1 else 0
      self.action_correctness_l.append(c)
      
      # print("qid= {}".format(a) )
      if not self.training_on:
        self.reset_training(j._id)
      
      self.jid_info_m[j._id] = {'ent': self.env.now, 'ts': j.tsize, 's': s, 'a': a}    
      self.q_l[a].put(Task(j._id, j.k, j.tsize) )
  
  def put(self, j):
    sim_log(DEBUG, self.env, self, "recved", j)
    return self.store.put(j)
  
  def put_c(self, m):
    sim_log(DEBUG, self.env, self, "recved", m)
    jid = m['jid']
    if jid not in self.jid_info_m: # from jobs that were not included in training set
      return
    
    jinfo = self.jid_info_m[jid]
    self.jid_info_m[jid]['sl'] = (self.env.now - jinfo['ent'] )/jinfo['ts']
    
    try:
      self.jid_to_wait_l.remove(jid)
      # log(WARNING, "removed jid= {}, len(jid_to_wait_l)= {}".format(jid, len(self.jid_to_wait_l) ) )
    except:
      pass
      # log(WARNING, "could not remove from jid_to_wait_l; jid= {}".format(jid) )
    if len(self.jid_to_wait_l) == 0: # Ready for training
      print("Training:: with jobs from {} to {}".format(self.jid_head_of_training, self.jid_tail_of_training) )
      self.training_on = False
      s_l, a_l, r_l = [], [], []
      for jid in range(self.jid_head_of_training, self.jid_tail_of_training+1):
        jinfo_m = self.jid_info_m[jid]
        s_l.append(jinfo_m['s'] )
        a_l.append(jinfo_m['a'] )
        r_l.append(1/jinfo_m['sl'] )
      print("Training:: sum(r_l)= {}".format(sum(r_l) ) )
      print("Training:: freq of correct= {}".format(sum(self.action_correctness_l)/len(self.action_correctness_l) ) )
      self.scher.train(s_l, a_l, r_l)
      self.jid_info_m.clear()

def learning_shortestq_wsingletraj():
  env = simpy.Environment()
  jg = JG(env, ar=2.5, k_dist=DUniform(1, 1), tsize_dist=DUniform(1, 1) )
  mq = LearningShortestQ_wSingleTraj(env, n=3)
  jg.out = mq
  jg.init()
  env.run(until=1000*100) # 50000*1

def ET_reptotwo_wcancel_woS(ns, ar, J):
  # B ~ J if Wp < Ws + J, Ws + J < X else 0
  # i.e. B ~ ro*(1-ro)*J
  ar = ar/ns
  EJ, EJ2 = moment_ith(1, J), moment_ith(2, J)
  
  eq = lambda ro: ro - ar*(1 - ro*(1-ro))*EJ
  ro = scipy.optimize.brentq(eq, 0.0001, 1)
  alog("ar= {}, ro= {}".format(ar, ro) )
  EB = EJ # (1 - ro*(1-ro))*EJ
  EB2 = EJ**2 # (1 - ro*(1-ro))*EJ**2
  
  # ar = ar*(1 - ro*(1-ro))
  ET = EB + ar*EB2/2/(1 - ar*EB)
  return ET if ET < ET_MAX else None
  
  # mu = 1/EJ
  # ar = ar*(1 - ro*(1-ro))
  # ET = ro/(mu - ar) + (1 - ro)/(2*mu - ar)
  # return ET
  
def _ar_ub_reptotwo_wcancel(ns, V):
  return ns/moment_ith(1, V)

def _ET_reptotwo_wcancel(ns, ar, V):
  ar = ar/ns
  V21 = X_n_k(V, 2, 1)
  EV, EV2 = moment_ith(1, V), moment_ith(2, V)
  EV21, EV21_2 = moment_ith(1, V21), moment_ith(2, V21)
  
  # eq = lambda ro: ro**2*ar*EV21/2 + ro*(ar*EV - 3/2*ar*EV21 - 1) + ar*EV21
  # ro = scipy.optimize.brentq(eq, 0.0001, 1)
  # EB = ro*EV + (1 - ro)*(1 - ro/2)*EV21
  # EB2 = ro*EV2 + (1 - ro)*(1 - ro/2)**2*EV21_2
  
  eq = lambda ro: ro**2*ar*EV21 + ro*(ar*EV - 2*ar*EV21 - 1) + ar*EV21
  ro = scipy.optimize.brentq(eq, 0.0001, 1)
  EB = ro*EV + (1 - ro)**2*EV21
  EB2 = ro*EV2 + (1 - ro)*(1 - ro)**2*EV21_2
  
  ET = EB + ar*EB2/2/(1 - ar*EB)
  return ET if ET < ET_MAX else None

def ETub_reptotwo_wcancel(ns, ar, V):
  ar = ar/ns
  Y = X_n_k(V, 2, 1)
  EV, EV2 = moment_ith(1, V), moment_ith(2, V)
  EY, EY2 = moment_ith(1, Y), moment_ith(2, Y)
  ro = ar*EY/(1 - ar*EV + ar*EY)
  
  EB = ro*EV + (1 - ro)*EY
  EB2 = ro*EV2 + (1 - ro)*EY2
  ET = EB + ar*EB2/2/(1 - ar*EB)
  return ET if ET < ET_MAX else None

def _ETapprox_reptotwo_wcancel(ns, ar, V):
  ar = ar/ns
  X = Exp(ar)
  
  EV, EV2 = moment_ith(1, V), moment_ith(2, V)
  Y = X_n_k(V, 2, 1)
  EY, EY2 = moment_ith(1, Y), moment_ith(2, Y)
  
  def laplace(X, r):
    return mpmath.quad(lambda x: math.exp(-r*x) * X.pdf(x), [0, X.u_l] )
  a = laplace(Y, ar)
  b = laplace(V, ar)
  eq = lambda W: (a-b)*W**2 + (b + ar*(EY - EV) )*W + ar*EV-1
  p_fast = scipy.optimize.brentq(eq, 0.0001, 2)
  p_fast = 1 if p_fast > 1 else p_fast
  print("p_fast= {}".format(p_fast) )
  
  EB = p_fast*EY + (1 - p_fast)*EV
  EB2 = p_fast*EY2 + (1 - p_fast)*EV2
  ET = EB + ar*EB2/2/(1 - ar*EB)
  return ET if ET < ET_MAX else None
  
"""
plot_reptod_wcancel:: ns= 5, T= 50000, J= TPareto(\min= 1, \max= 100, \alpha= 1.1), S= DUniform[1, 1], ar_ub= 1.2239166932608347
d= 2

> ar= 0.01
r_freqj_l= [0.00656, 0.99344]
d_freqj_l= [0.0, 1.0]
sching_m= {'reptod-wcancel': 0, 'd': 2}, 
ro_sim= 0.007972226491907434, 
ET_sim= 4.09160538832408
EV= 3.9857986211437093, EV2= 72.68889302162749
EV21= 3.9857986211437093, EV21_2= 72.68889302162749
ET_reptotwo_wcancel:: ar= 0.002, 
ro= 0.007955868140387089
ET= 4.051061328453414

> ar= 0.3134791733152087
r_freqj_l= [0.21872, 0.78128]
d_freqj_l= [0.0, 1.0]
sching_m= {'reptod-wcancel': 0, 'd': 2}, 
ro_sim= 0.20839304880722626, 
ET_sim= 5.903064209255924
EV= 3.9857986211437093, EV2= 72.68889302162749
EV21= 3.9857986211437093, EV21_2= 72.68889302162749
ET_reptotwo_wcancel:: ar= 0.06269583466304174, 
ro= 0.23854524341878833
ET= 6.661401654385554

> ar= 0.6169583466304174
r_freqj_l= [0.43278, 0.56722]
d_freqj_l= [0.0, 1.0]
sching_m= {'reptod-wcancel': 0, 'd': 2}, 
ro_sim= 0.4041399055142322, 
ET_sim= 10.648021402441367
EV= 3.9857986211437093, EV2= 72.68889302162749
EV21= 3.9857986211437093, EV21_2= 72.68889302162749
ET_reptotwo_wcancel:: ar= 0.12339166932608348, 
ro= 0.461260471594443
ET= 11.545288128525751

> ar= 0.9204375199456261
r_freqj_l= [0.66482, 0.33518]
d_freqj_l= [0.0, 1.0]
sching_m= {'reptod-wcancel': 0, 'd': 2}, 
ro_sim= 0.6333399826640576, 
ET_sim= 21.85920639110326
EV= 3.9857986211437093, EV2= 72.68889302162749
EV21= 3.9857986211437093, EV21_2= 72.68889302162749
ET_reptotwo_wcancel:: ar= 0.1840875039891252, 
ro= 0.6948409377289061
ET= 24.5371189900176

> ar= 1.2239166932608347
r_freqj_l= [0.97144, 0.02856]
d_freqj_l= [0.0, 1.0]
sching_m= {'reptod-wcancel': 0, 'd': 2}, 
ro_sim= 0.9626459789623473, 
ET_sim= 316.48111639863293
EV= 3.9857986211437093, EV2= 72.68889302162749
EV21= 3.9857986211437093, EV21_2= 72.68889302162749
ET_reptotwo_wcancel:: ar= 0.24478333865216695, 
ro= 0.9681316701683959
ET= 280.9665756567154
"""