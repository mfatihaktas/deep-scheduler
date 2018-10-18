import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.use('Agg')
import matplotlib.pyplot as plot
import itertools

mew, ms = 3, 5

dark_color = itertools.cycle(('green', 'red', 'blue', 'goldenrod', 'magenta', 'purple', 'gray', 'brown', 'turquoise', 'gold', 'olive', 'silver', 'rosybrown', 'plum', 'lightsteelblue', 'lightpink', 'orange', 'darkgray', 'orangered'))
light_color = itertools.cycle(('silver', 'rosybrown', 'plum', 'lightsteelblue', 'lightpink', 'orange', 'turquoise'))
linestyle = itertools.cycle(('-', '--', '-.', ':') )
marker = itertools.cycle(('^', 'p', 'd', '+', 'v', '<', '>', '1' , '2', '3', '4', 'x') )
skinny_marker_l = ['x', '+', '1', '2', '3', '4']