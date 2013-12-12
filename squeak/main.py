from __future__ import division
import os
import pandas as pd
import numpy as np
from scipy import interpolate, interp, stats
from math import sqrt
import matplotlib.pyplot as plt
import math

# Normalizaton
def even_time_steps(x, y, t, length = 101):
	"""Interpolates x/y coordinates and timestamps to 101 even time steps
	
	Input can be lists, or numpy arrays.
	Returns normalized x coordinates, y coordinates, and corresponding normalized timestamps.
	"""
	nt = np.arange(min(t), max(t), (float(max(t)-min(t))/length))
	nx = interp(nt, t, x)
	ny = interp(nt, t, y)
	return nx, ny, nt

def normalize_space(array, start=0, end=1):
	"""Interpolates array of 1-d coordinates to given start and end value.
	
	TODO: Might not work on decreasing arrays. Test ti"""
	old_delta = array[-1] - array[0] # Distance between old start and end values.
	new_delta = end - start # Distance between new ones.
	# Convert coordinates to float values
	array = array.astype('float')
	# Finally, interpolate. We interpolate from (start minus delta) to (end plus delta)
	# to handle cases where values go below the start, or above the end values.
	#~ normal = interp(array, [array[0] - old_delta, array[-1] + old_delta], [start - new_delta, end + new_delta])
	old_range = np.array([array[0] - old_delta, array[-1]+old_delta])
	new_range = np.array([start - new_delta, end + new_delta])
	#if max(array) > old_range[-1]:
		#print 'Array: %s , old_range: %s' % (array, old_range)
	normal = np.interp(array, old_range, new_range)
	return normal
	
def remap_right(array):
	"""Flips decreasing coordinates horizontally on their origin
	
	>>> remap_right([10,11,12])
	array([10, 11, 12])

	>>> remap_right([10, 9, 8])
	array([10, 11, 12])
	"""
	array = np.array(array)
	if array[-1] - array[0] < 0:
		array_start = array[0]
		return ((array-array_start)*-1)+array_start
	else:
		return array
	
def list_from_string(string_list):
	"""Converts string represation of list '[1,2,3]' to an actual pythonic list [1,2,3]
	
	A rough and ready function"""
	try:
		first = string_list.strip('[]')
		then = first.split(',')
		for i in range(len(then)):
			then[i] = int(then[i])
		return(np.array(then))
	except:
		return None
	
	
# # # Functions to apply to a set of trajectories at a time # # #

def average_path(x, y, full_output=False):#, length=101):
	"""Averages Pandas data columns of x and y trajectories into a single mean xy path.
	
	Finds length of first row, and then averages the i-th entry of the x and y columns
	for i in range(0,length).
	
	Parameters
	----------
	x, y : Pandas DataFrame columns
	full_output : bool, optional
		Return all values, not just the average.
		Used by compare_means_1d()
	TODO: Allow other datatypes as input.
	TODO: Create the option of returning variance as well as mean?
		See http://stanford.edu/~mwaskom/software/seaborn/timeseries_plots.html
	"""
	# Can this be done more efficiently with .apply()?
	mx, my = [], []
	fullx, fully = [], []
	length = len(x.iloc[0])
	for i in range(length):
		this_x, this_y = [], []
		for p in range(len(x)):
			this_x.append(x.iloc[p][i])
			this_y.append(y.iloc[p][i])
		if full_output:
			fullx.append(this_x)
			fully.append(this_y)
		mx.append(np.mean(this_x))
		my.append(np.mean(this_y))
	if full_output:
		return fullx, fully
	return np.array(mx), np.array(my)
	
def compare_means_1d(dataset, groupby, condition_a, condition_b, y = 'x', test = 't', length=101):
	"""Possibly depreciated: Compares average coordinates from two conditions using a series of t or Mann-Whitney tests.
	
	Parameters
	----------
	dataset: Pandas DataFrame
	groupby: string
		The column in which the groups are defined
	condition_a, condition_b: string
		The labels of each group (in column groupby)
	y: string, optional
		The column name of the coordinates to be compared.
		Default 'x'
	test: string, optional
		Statistical test to use.
		Default: 't' (independant samples t test)
		Alternate: 'u' (Non-parametric Mann-Whitney test)
		
	Returns
	-----------
	t101 : t (or U) values for each point in the trajectory
	p101 : Associated p values"""
	a_x, a_y = average_path(dataset[y][dataset[groupby] == condition_a], dataset[y] [dataset[groupby] == condition_a], full_output=True)
	b_x, b_y = average_path(dataset[y][dataset[groupby] == condition_b], dataset[y][dataset[groupby] == condition_b], full_output=True)
	t101, p101 = [], []
	for i in range(length):
		if test == 't':# t-test
			t, p = stats.ttest_ind(a_y[i], b_y[i])
		elif test == 'u':# Mann-Whitney
			t, p = stats.mannwhitneyu(a_y[i], b_y[i])
		t101.append(t)
		p101.append(p)
	return t101, p101
	
# Depreciated Plotting functions
def plot_means_1d(dataset, groupby, condition_a, condition_b, y = 'x', legend=True, title=None):
	"""Depreciated: Convenience function for plotting two 1D lines, representing changes on x axis
	
		Parameters
	----------
	dataset: Pandas DataFrame
	groupby: string
		The column in which the groups are defined
	condition_a, condition_b: string
		The labels of each group (in column groupby)
	y: string, optional
		The column name of the coordinates to be compared.
		Default 'x'
	legend: bool, optional
		Include legend on plot
	title: string, optional

	Takes a Pandas DataFrame, divides it by the grouping variable 'groupby' 
	(a string), and plots the average of all paths in 'condition_a' in blue,
	and the average from 'condition_b' in red.
	Includes a legend by default, and a title if given."""
	a_x, a_y = average_path(dataset[y][dataset[groupby] == condition_a], dataset[y][dataset[groupby] == condition_a])
	b_x, b_y = average_path(dataset[y][dataset[groupby] == condition_b], dataset[y][dataset[groupby] == condition_b])
	l1 = plt.plot(a_y, color = 'r', label = condition_a)
	l2 = plt.plot(b_y, 'b', label=condition_b)
	if legend:
		plt.legend()
	plt.title(y)
	return None

def plot_means_2d(dataset, groupby, condition_a, condition_b, x='x', y='y', length=101, legend=True, title='Average paths'):
    	"""Plots x and y coordinates.
	Assumes that dataset contains values 'x' and 'y', comprising mouse
	paths standarised into 101 time steps.
	Takes a Pandas DataFrame, divides it by the grouping variable 'groupby' 
	(a string), and plots the average of all paths in 'condition_a' in blue,
	and the average from 'condition_b' in red.
	Includes a legend by default, and a title if given."""
	a_x, a_y = average_path(dataset[x][dataset[groupby] == condition_a], dataset[y][dataset[groupby] == condition_a], length=length)
	b_x, b_y = average_path(dataset[x][dataset[groupby] == condition_b], dataset[y][dataset[groupby] == condition_b], length=length)
	l1 = plt.plot(a_x, a_y, color = 'b', label = condition_a)
	l2 = plt.plot(b_x, b_y, 'r', label=condition_b)
	if legend:
		plt.legend()
	if title:
		plt.title(title)
	return a_x, a_y, b_x, b_y

def plot_all(dataset, groupby, condition_a, condition_b, x='x', y='y', legend=True, title=None):
	"""Plots all trajectories in condition_a and _b"""
	# Don't use this, use DataFrame.apply(lambda trial: plt.plot(trial['x'], trial['y'], color_map[trial['conditon']])
	for i in range(len(dataset)):
		y_path = dataset[y].iloc[i]
		if type(x) == list:
			x_path = x
		elif x == 'time':
			x_path = range(len(y_path))
		else:
			x_path = dataset[x].iloc[i]
		if dataset[groupby].iloc[i] == condition_a:
			plt.plot(x_path, y_path, 'b')
		elif dataset[groupby].iloc[i] == condition_b:
			plt.plot(x_path, y_path, 'r')
    #return?


# # # Functions to apply to a single trajectory at a time # # #
def rel_distance(x_path, y_path, full_output = False):
	"""Takes a path's x and y co-ordinates, and returns
	a list showing relative distance from each response at
	each point along path, with values closer to 0 when close to
	response 1, and close to 1 for response 2"""
	# TODO make these reference targets flexible as input
	rx1, ry1, rx2, ry2 = -1, 0, 1, 0
    	r_d, d_1, d_2 = [], [], []
    	for i in range(len(x_path)):
			x = x_path[i]
			y = y_path[i]
			# Distance from each
			d1 = sqrt( (x-rx1)**2 + (y-ry1)**2 )
			d2 = sqrt( (x-rx2)**2 + (y-ry2)**2 )
			# Relative distance
			rd = (d1 / (d1 + d2) )
			r_d.append(rd)
			if full_output:
				d_1.append(d1)
				d_2.append(d2)
	if full_output:
		return r_d, d_1, d_2
	else:
		return np.array(r_d)

def avg_incr(series):
    d = []
    for i in range(len(series)-1):
        d.append(series[i+1] - series[i])
    return float(sum(d)) / len(d)
    
def extend_raw_path(path, target_duration=3000, t=None, rate=10):
    if type(t) == list:
        smart_t = avg_incr(t)
    path = list(path)
    for i in range( int((target_duration / rate) - len(path)) ):
        path.append(path[-1])
    return np.array(path)

# Use this instead
def uniform_time(coordinates, timepoints, desired_interval=10, max_duration=3000):
    # Interpolte to desired_interval
    regular_timepoints = np.arange(0, timepoints[-1]+1, desired_interval)
    regular_coordinates = interp(regular_timepoints, timepoints, coordinates)
    # How long should this be so that all trials are the same duration?
    required_length = int(max_duration / desired_interval)
    # Generate enough of the last value to make up the difference
    extra_values = np.array([regular_coordinates[-1]] * (required_length - len(regular_coordinates)))
    return np.concatenate([regular_coordinates, extra_values])


def get_init_time(t, y, y_limit, ascending = True):
	"""Returns the time taken for the path to go above
	y_limit (or below, if ascending is set to False)"""
	j = 0
	this_y = y[j]
	if ascending:
		while this_y < y_limit:
			# Loop until y is above the limit
			j += 1
			this_y = y[j]
	else:
		while this_y > y_limit:
			# Loop until y is above the lim
			j += 1
			this_y = y[j]
	# Return time corresponding to this y
	return(t[j])
	
def get_init_step(y, y_threshold = .01, ascending = True):
	"""Returns the index of where where the path goes above
	y_limit (or below, if ascending is set to False)"""
	# Get array that is True when y is beyond the threshold
	if ascending:
		started = np.array(y) > y_threshold
	else:
		started = np.array(y) < y_threshold
	# Get the first True value's index.
	step = np.argmax(started)
	return step

#~ def max_dev(x,y):
	#~ global n, p
	#~ # # This is treating positive and negative deviations as the same.
	#~ # # Change this!
	#~ startx, starty, endx, endy = 0,1,1,0
	#~ ideal_x = np.arange(startx, endx, (endx-startx)*.1)
	#~ ideal_y = np.arange(starty, endy, (endy-starty)*.1)
	#~ ideal_x = np.append(ideal_x, endx)
	#~ ideal_y = np.append(ideal_y, endy)
	#~ deviations, md_signs, dev_locations = [], [], []
	#~ for i in range(len(x)):
		#~ distances = []
		#~ dist_locations = []
		#~ signs = []
		#~ this_x = x[i]
		#~ this_y = y[i]
		#~ for j in range(11):
			#~ refx = ideal_x[j]
			#~ refy = ideal_y[j]
			#~ dist = sqrt( (refx-this_x)**2 + (refy-this_y)**2)
			#~ distances.append(dist)
			#~ signs.append(np.sign(this_x-refx))
			#~ dist_locations.append([refx, this_x])
			#~ #print dist, dist_locations[distances.index(min(distances))
		#~ #print len(distances), len(dist_locations)
		#~ deviations.append(min(distances))
		#~ md_signs.append(signs[distances.index(min(distances))])
		#~ min_dist_loc = dist_locations[distances.index(min(distances))]
		#~ dev_locations.append(min_dist_loc)
	#~ md = max(deviations)
	#~ md_sign = md_signs[deviations.index(max(deviations))]
	#~ #md_location = dev_locations[deviations.index(md)]
	#~ return md*md_sign #, md_location
	#~ 
#~ def abs_max(array):
    #~ loc = abs(array).argmax()
    #~ return array[loc]
    #~ 
#~ def abs_min(array):
    #~ loc = abs(array).argmin()
    #~ return array[loc]
#~ def min_distance(x, y, x_start=0, y_start=0, x_end=1, y_end=1):
	#~ ideal_x = np.arange(x_start, x_end)
	#~ 
#~ def md2(x, y):
	#~ local_minimums = []
	#~ 
#~ 
#~ def pythagoras(x1, y1, x2, y2):
	#~ dist = sqrt( (x1-x2)**2 + (y1-y2)**2)
	#~ return dist
	
def max_deviation(x, y):
	rx, ry = [], []
	# Turn the path on its side.
	for localx, localy in zip(x, y):
		rot = rotate(localx, localy, math.radians(45))
		rx.append(rot[0])
		ry.append(rot[1])
	max_positive = abs(min(rx))
	max_negative = abs(max(rx))
	#print max_positive, max_negative
	if max_positive > max_negative:
		# The return the positive MD
		return max_positive
	else:
		# Return the negative (rare)
		return -1*max_negative


def rotate(x, y, rad):
	"""Rotate counter-clockwise by rad radians.
	"""
	s, c = [f(rad) for f in (math.sin, math.cos)]
	x, y = (c*x - s*y, s*x + c*y)
	return x,y

        			
	
def auc(x, y):
	areas = []
	j = len(x) - 1
	for i in range(len(x)):
		x1y2 = y[i]*x[j]
		x2y1 = x[i] * y[j]
		area = x2y1 - x1y2
		areas.append(area)
		j = i
	return float(sum(areas))/2
	
def auc2(x, y):
	areas = []
	x = list(x)
	y = list(y)
	x.append(x[-1])
	y.append(y[0])
	j = len(x) - 1
	for i in range(len(x)):
		x1y2 = y[i]*x[j]
		x2y1 = x[i] * y[j]
		area = x2y1 - x1y2 
		areas.append(area)
		j = i
	triangle = .5 * abs(x[-1] - x[0]) * abs(y[-1]*y[0])
	return float(sum(areas)) - triangle

def pythag(o, a):
	return np.sqrt( o**2 + a**2)

def velocity(x, y):
	vx = np.ediff1d(x)
	vy = np.ediff1d(y)
	vel = np.sqrt( vx**2 + vy **2 ) # Pythagoras
	return vel
    
def bimodality_coef(samp):
	n = len(samp)
	m3 = stats.skew(samp)
	m4 = stats.kurtosis(samp, fisher=True)
	#b = ( g**2 + 1) / ( k + ( (3 * (n-1)**2 ) / ( (n-2)*(n-3) ) ) )
	b=(m3**2+1) / (m4 + 3 * ( (n-1)**2 / ((n-2)*(n-3)) ))
	return b

# Inference
def chisquare_boolean(array1, array2):
    observed_values = np.array([sum(array1), sum(array2)])
    total_len = np.array([len(array1), len(array2)])
    expected_ratio = sum(observed_values) / sum(total_len)
    expected_values = total_len * expected_ratio
    chisq, p = stats.chisquare(observed_values, f_exp = expected_values)
    return chisq, p


# Make a GIF 
#~ for i in range(301):
    #~ plt.clf()
    #~ for j in range(len(data)):
        #~ if data.code.iloc[j] == 'lure':
            #~ style = 'r.'
        #~ elif data.code.iloc[j] == 'control':
            #~ style = 'b.'
        #~ else:
            #~ style = None
        #~ if style:
            #~ x = data.fx.iloc[j]
            #~ y = data.fy.iloc[j]
            #~ if len(x) > i:
                #~ plt.plot(x[i], y[i], style)
            #~ else:
                #~ plt.plot(x[-1], y[-1], style)
    #~ plt.xlim((-1.2, 1.2))
    #~ plt.ylim((-.2, 1.2))
    #~ plt.title('%ims' % (i*10))
    #~ plt.savefig(os.path.join(path, 'Step_%i.png' % (1000+i)))
# Then, using imagemagick
# convert C:\Users\40027000\Desktop\Software\ImageMagick\ImageMagick-6.8.7-0\convert -delay 10 -loop 1 *.png Output.gif
