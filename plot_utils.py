import inspect, math, mpmath, scipy, itertools
from scipy import special

mew, ms = 3, 5

dark_color = itertools.cycle(('green', 'red', 'blue', 'goldenrod', 'magenta', 'purple', 'gray', 'brown', 'turquoise', 'gold', 'olive', 'silver', 'rosybrown', 'plum', 'lightsteelblue', 'lightpink', 'orange', 'darkgray', 'orangered'))
light_color = itertools.cycle(('silver', 'rosybrown', 'plum', 'lightsteelblue', 'lightpink', 'orange', 'turquoise'))
linestyle = itertools.cycle(('-', '--', '-.', ':') )
marker = itertools.cycle(('^', 'p', 'd', '+', 'v', '<', '>', '1' , '2', '3', '4', 'x') )
skinny_marker_l = ['x', '+', '1', '2', '3', '4']