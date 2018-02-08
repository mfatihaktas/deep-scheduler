import numpy as np

def prob_jinsys_mmn(ar, mu, n, j):
  ro = ar/mu/n
  
  P0 = 1/sum([ro**i/fact(i) + ro**n/fact(n)*(n*mu/(n*mu - ar) ) for i in range(n) ] )
  if j <= n:
    return ro**j/fact(j) * P0
  else:
    return ro**j/fact(n)/(n)**(j-n) * P0


# ###########################################  Sim  ############################################## #
class Server(object):
  def __init__(self, _id, env, slowdown_dist, out_c):
    self._id = _id
    self.env = env
    self.slowdown_dist = slowdown_dist
    self.out_c = out_c
    
    self.t_inserv = None
    self.cancel_flag = False
    self.cancel = None
    
  def __repr__(self):
    return "Server[_id= {}]".format(self._id)
  
  def busy(self):
    return self.t_inserv == None
  
  def serv_run(self):
    self.cancel = self.env.event()
    # clk_start_time = self.env.now
    st = self.t_inserv.size * self.slowdown_dist.gen_sample()
    # sim_log(DEBUG, self.env, self, "starting {}s-clock on ".format(st), self.t_inserv)
    yield (self.cancel | self.env.timeout(st) )
    if self.cancel_flag:
      # sim_log(DEBUG, self.env, self, "cancelled clock on ", self.t_inserv)
      self.cancel_flag = False
    else:
      # sim_log(DEBUG, self.env, self, "serv done in {}s on ".format(self.env.now-clk_start_time), self.t_inserv)
      self.out_c.put_c({'jid': self.t_inserv.jid} )
    self.t_inserv = None
  
  def put(self, t):
    sim_log(DEBUG, self.env, self, "recved", t)
    self.t_inserv = t
  
  def put_c(self, m):
    sim_log(DEBUG, self.env, self, "recved", m)
    # if m['m'] == 'cancel':
    jid = m['jid']
    if self.t_inserv is not None and jid == self.t_inserv.jid:
      self.cancel_flag = True
      self.cancel.succeed()

class MGn(object):
  def __init__(self, env, n, sl_dist):
    self.n = n
    
    self.q_l = [FCFS(i, env, sl_dist, out=self.jq, out_c=self.jq, L=L) for i in range(self.n) ]
  
  