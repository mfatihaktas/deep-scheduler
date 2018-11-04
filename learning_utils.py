from sim_objs import *

STATE_LEN = 3
def state(j, wload_l=None):
  if STATE_LEN == 1:
    return [j.totaldemand] # j.k
  elif STATE_LEN == 3:
    # return [j.totaldemand, min(wload_l), max(wload_l) ]
    return [j.totaldemand, np.mean(wload_l), np.std(wload_l) ]
  elif STATE_LEN == 5:
    return [j.totaldemand, min(wload_l), max(wload_l), np.mean(wload_l), np.std(wload_l) ]

def sample_traj(sinfo_m, scher):
  def reward(slowdown):
    # return 1/slowdown
    # return 10 if slowdown < 1.5 else -10
    
    ## The following allows Q-learning to converge
    # if slowdown < 1.1:
    #   return 10
    # elif slowdown < 1.5:
    #   return 10/slowdown
    # else:
    #   return -slowdown
    
    return -slowdown
    
    # if slowdown < 2:
    #   return 10/slowdown
    # else:
    #   return -10*slowdown
    
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
         np.mean([w.avg_load for w in cl.w_l] ), \
         0
         # sum([1 for _, jinfo_m in cl.jid_info_m.items() if 'fate' in jinfo_m and jinfo_m['fate'] == 'dropped'] )/len(cl.jid_info_m)

def evaluate(sinfo_m, scher):
  alog("scher= {}".format(scher) )
  for _ in range(3):
    t_s_l, t_a_l, t_r_l, t_sl_l = sample_traj(sinfo_m, scher)
    print("avg_s= {}, avg_a= {}, avg_r= {}".format(np.mean(t_s_l), np.mean(t_a_l), np.mean(t_r_l) ) )
