from policygrad_learning import *
from q_learning import *

# #############################################  Test  ########################################### #
def test():
  s_len, a_len, nn_len = 3, 3, 10
  scher = PolicyGradLearner(s_len, a_len, nn_len)
  # scher = QLearner(s_len, a_len, nn_len)
  
  def state():
    s = np.random.randint(10, size=s_len)
    sum_s = sum(s)
    return s/sum_s if sum_s != 0 else s
  
  def reward(s, a):
    # s_min = min(s)
    # r = 10 if s[a] == s_min else 0
    # return min(100, 1/(0.001 + s[a] - min(s) ) )
    # return 100*math.exp(-(s[a] - min(s) ) )
    return 1/(0.1 + s[a] - min(s) )
  
  def evaluate():
    num_shortest_found = 0
    for e in range(100):
      s = state()
      a = scher.get_max_action(s)
      if s[a] - min(s) < 0.01:
        num_shortest_found += 1
    print("freq shortest found= {}".format(num_shortest_found/100) )
  
  def train_w_mult_trajs():
    N, T = 10, 100
    def gen_N_traj():
      n_t_s_l, n_t_a_l, n_t_r_l = np.zeros((N, T, s_len)), np.zeros((N, T, 1)), np.zeros((N, T, 1))
      for n in range(N):
        for t in range(T):
          s = state()
          a = scher.get_random_action(s)
          n_t_s_l[n, t, :] = s
          n_t_a_l[n, t, :] = a
          n_t_r_l[n, t, :] = reward(s, a)
      return n_t_s_l, n_t_a_l, n_t_r_l
    
    for i in range(100*20):
      n_t_s_l, n_t_a_l, n_t_r_l = gen_N_traj()
      scher.train_w_mult_trajs(n_t_s_l, n_t_a_l, n_t_r_l)
      if i % 10 == 0:
        evaluate()
  train_w_mult_trajs()

def vsimple_regress():
  s_len = 3
  T = 100
  def state():
    s = np.random.randint(10, size=s_len)
    sum_s = sum(s)
    return s/sum_s if sum_s != 0 else s
  
  def reward(s):
    return 10*max(s)
  
  def sample_traj():
    t_s_l, t_r_l = np.zeros((T, s_len)), np.zeros((T, 1))
    for t in range(T):
      s = state()
      t_s_l[t, :] = s
      t_r_l[t, :] = reward(s)
    return t_s_l, t_r_l
  
  value_ester = VEster(s_len, nn_len=10, straj_training=False)
  for i in range(100*40):
    t_s_l, t_r_l = sample_traj()
    value_ester.train_w_single_traj(t_s_l, t_r_l)

class A(object):
  def __init__(self):
    self.a = 'A'
  
  def __repr__(self):
    return self.a
  
  def test(self):
    print("self.a= {}".format(self.a) )
  
class B(A):
  def __init__(self):
    super().__init__()
    
    self.init()
  
  def init(self):
    self.a = 'B'

if __name__ == "__main__":
  # test()
  # vsimple_regress()
  
  # b = B()
  # # print("b= {}".format(b) )
  # b.test()
  
  # learner = PolicyGradLearner(s_len=1, a_len=1)
  learner = QLearner(s_len=1, a_len=1)
  learner.save(0)
  restore_result = learner.restore(0)
  print("restore_result= {}".format(restore_result) )
