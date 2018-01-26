import inspect, math, mpmath, scipy, itertools
from scipy import special

# dark_color = itertools.cycle(('green', 'red', 'blue', 'turquoise', 'goldenrod', 'purple', 'gray', 'brown', 'magenta', 'gold', 'olive', 'orangered', 'silver', 'rosybrown', 'plum', 'lightsteelblue', 'lightpink', 'orange', 'darkgray'))
dark_color = itertools.cycle(('green', 'red', 'blue', 'goldenrod', 'magenta', 'purple', 'gray', 'brown', 'turquoise', 'gold', 'olive', 'silver', 'rosybrown', 'plum', 'lightsteelblue', 'lightpink', 'orange', 'darkgray', 'orangered'))
light_color = itertools.cycle(('silver', 'rosybrown', 'plum', 'lightsteelblue', 'lightpink', 'orange', 'turquoise'))
linestyle = itertools.cycle(('-', '--', '-.', ':') )
marker = itertools.cycle(('^', 'p', 'd', '+', 'v', '<', '>', '1' , '2', '3', '4', 'x') )
skinny_marker_l = ['x', '+', '1', '2', '3', '4']

mew, ms = 3, 5

INFO = 0
DEBUG = 1
WARNING = 2
ERROR = 3

# DEBUG_LEVEL = INFO
# DEBUG_LEVEL = WARNING
DEBUG_LEVEL = ERROR

debug_level__string_map = {INFO: "INFO", DEBUG: "DEBUG", WARNING: "WARNING", ERROR: "ERROR"}

"""
*log: To have a unified logging which can be refactored easily
"""
def sim_log(dlevel, env, caller, action, affected):
  """
  Parameters
  ----------
  dlevel= int -- debug level
  env= simpy.Environment
  caller= string -- name of the sim component acting
  action= string
  affected= any -- whatever component being acted on/with e.g., packet
  """
  if DEBUG_LEVEL <= dlevel:
    print("{} t: {:.2f}] {} {}\n\t{}".format(debug_level__string_map[dlevel], env.now, caller, action, affected) )

def log(dlevel, log):
  """
  Parameters
  ----------
  dlevel= int -- debug level
  log= string to be logged
  """
  if DEBUG_LEVEL <= dlevel:
    print("{}] {}:: {}".format(debug_level__string_map[dlevel], inspect.stack()[1][3], log) )

def alog(log):
  print("{}:: {}".format(inspect.stack()[1][3], log) )

def save_name(folder_name, prob_name, ns, d, ar):
  return "{}/{}_ns{}_d{}_ar{}".format(folder_name, prob_name, ns, d, ar)

