import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.use('Agg')
import matplotlib.pyplot as plot
import itertools

mew, ms = 3, 5

NICE_BLUE = '#66b3ff'
NICE_RED = '#ff9999'
NICE_GREEN = '#99ff99'
NICE_ORANGE = '#ffcc99'

nicecolor_c = itertools.cycle((NICE_BLUE, NICE_RED, NICE_GREEN, NICE_ORANGE))
darkcolor_c = itertools.cycle(('green', 'red', 'blue', 'gray', 'magenta', 'goldenrod', 'purple', 'brown', 'turquoise', 'gold', 'olive', 'silver', 'rosybrown', 'plum', 'lightsteelblue', 'lightpink', 'orange', 'darkgray', 'orangered'))
lightcolor_c = itertools.cycle(('silver', 'rosybrown', 'plum', 'lightsteelblue', 'lightpink', 'orange', 'turquoise'))
linestyle_c = itertools.cycle(('-', '--', '-.', ':') )
marker_c = itertools.cycle(('^', 'p', 'd', 'v', '<', '>', '1', '2', '3', '4', 'x', '+') )
skinnym_l = ['x', '+', '1', '2', '3', '4']

def prettify(ax):
  plot.tick_params(top='off', right='off', which='both')
  ax.patch.set_alpha(0.2)
  ax.spines['right'].set_visible(False)
  ax.spines['top'].set_visible(False)