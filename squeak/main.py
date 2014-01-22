from __future__ import division
import os
import pandas as pd
import numpy as np
from scipy import interpolate, interp, stats
from math import sqrt
import matplotlib.pyplot as plt
import math
import seaborn as sns

# # # Normalizaton
def even_time_steps(x, y, t, length = 101):
	"""Interpolates x/y coordinates and t to 101 even time steps, returns x and y TimeSeries
	
	Parameters
	----------
	x, y : array-like
		Coordinates to be interpolated
	t : array-like
		Associated time stamps
	length : int, optional
		Number of time steps to interpolate to. Default 101
		
	Returns
	---------
	TimeSeries(nx, nt) : Pandas.TimeSeries object
		x coordinates intepolated to 101 even time steps
	TimeSeries(ny, nt) : Pandas.TimeSeries object
		y coordinates intepolated to 101 even time steps
	"""
	nt = np.arange(min(t), max(t), (float(max(t)-min(t))/length))
	nx = interp(nt, t, x)[:101] # Sometimes it ends up 102 steps long.
	ny = interp(nt, t, y)[:101]
	return pd.TimeSeries(nx, range(len(nx))), pd.TimeSeries(ny, range(len(ny)))

def normalize_space(array, start=0, end=1):
	"""Interpolates array of 1-d coordinates to given start and end value.
	
	TODO: Might not work on decreasing arrays. Test this"""
	old_delta = array[-1] - array[0] # Distance between old start and end values.
	if old_delta < 0:
		raise Exception('normalize_space requires the end value of the array to be higher than the start')
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

def uniform_time(coordinates, timepoints, desired_interval=20, max_duration=3000):
	"""Extend coordinates to desired duration by repeating the final value
	
	Parameters
	----------
	coordinates : array-like
		1D x or y coordinates to extend
	timepoitns : array-like
		timestamps corresponding to each coordinate
	desired_interval : int, optional
		frequency of timepoints in output, in ms
		Default 10
	max_duration : int, optional
		Length to extend to.
		Note: Currently crashes if max(timepoints) > max_duration
		Default 3000
		
	Returns
	---------
	uniform_time_coordinates : coordinates extended up to max_duration"""
	# Interpolte to desired_interval
	regular_timepoints = np.arange(0, timepoints[-1]+.1, desired_interval)
	regular_coordinates = interp(regular_timepoints, timepoints, coordinates)
	# How long should this be so that all trials are the same duration?
	required_length = int(max_duration / desired_interval)
	# Generate enough of the last value to make up the difference
	extra_values = np.array([regular_coordinates[-1]] * (required_length - len(regular_coordinates)+1))
	
	extended_coordinates = np.concatenate([regular_coordinates, extra_values])
	extended_timepoints = np.arange(0, max_duration+.1, desired_interval)
	#print len(extended_coordinates), len(extended_timepoints)
	# Return as a time series
	return pd.TimeSeries(extended_coordinates, extended_timepoints)

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

# # # Functions to apply to a single trajectory at a time # # #
def rel_distance(x_path, y_path, full_output = False):
	"""In development - Do not use
	
	Takes a path's x and y co-ordinates, and returns
	a list showing relative distance from each response at
	each point along path, with values closer to 0 when close to
	response 1, and close to 1 for response 2"""
	# TODO make these reference targets flexible as input
	rx1, ry1 = x_path[0], x_path[-1]
	rx2, ry2 = -1* rx1, ry1
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

def get_init_time(t, y, y_threshold=.01, ascending = True):
	"""Returns time  from t of point where y exceeds y_threshold.
	
	Parameters
	----------
	y, t : array-like
		y coordinates, and associated timestamps
	y_threshold : int, optional
		Value beyond which y is said to be in motion
		Default is .01
	ascending : bool, optional
		If True (default) return first value where y > threshold.
		Otherwise, return first where y < threshold (y decreases).
	
	Returns
	-------
	init_time : Timestamp of first y value to exceed y_threshold.
	"""
	init_step = get_init_step(y, y_threshold, ascending)
	return t[init_step]
	
def get_init_step(y, y_threshold = .01, ascending = True):
	"""Return index of point where y exceeds y_threshold
	
	Parameters
	----------
	y : array-like
	y_threshold : int, optional
		Value beyond which y is said to be in motion
		Default is .01
	ascending : bool, optional
		If True (default) return first value where y > threshold.
		Otherwise, return first where y < threshold (y decreases).
	
	Returns
	-------
	step: index of y which first exceeds y_threshold
	"""
	# Get array that is True when y is beyond the threshold
	if ascending:
		started = np.array(y) > y_threshold
	else:
		started = np.array(y) < y_threshold
	# Get the first True value's index.
	step = np.argmax(started)
	return step

def max_deviation(x, y):
	"""Caluclate furthest distance between observed path and ideal straight one.
	
	Parameters
	----------
	x, y : array-like
		x and y coordinates of the path.
		
	Returns
	----------
	max_dev : Greatest distance between observed and straight line.
	
	As with the rest of Squeak, this assumes a line running from bottom center
	(0, 0) to top right (1, 1), or (1, 1.5), as it relies on rotating the
	line 45 degrees anticlockwise and comparing it to the y axis.
	
	Will return negative values in cases where the greatest distance
	is to the right (i.e. AWAY from the alternative response).
	"""
	rx, ry = [], []
	# Turn the path on its side.
	radians_to_rotate = math.atan(float(x[len(x)-1])/y[len(y)-1]) # Handling Pandas dataforms
	for localx, localy in zip(x, y):
		rot = rotate(localx, localy, radians_to_rotate)#math.radians(45))
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


def rotate(x, y, rad): # Needed for max_deviation()
	"""Rotate counter-clockwise around origin by `rad` radians.
	"""
	s, c = [f(rad) for f in (math.sin, math.cos)]
	x, y = (c*x - s*y, s*x + c*y)
	return x,y

	
def auc(x, y):
	"""Calculates area between observed path and idea straight line.
	
	An alternative to max_deviation
	
	Parameters
	----------
	x, y : array-like
		x and y coordinates of the path.
		
	Returns
	----------
	area : Total area enclosed by the curve and line together
	
	NOTE: auc() and auc2() differ in that the former calcuates area 
	enclosed by the curve and the ideal straight line only, while the 
	latter calculates the area between the curve and the x axis, and then
	subtracts the triangular area between the straight line and the 
	x axis.
	TODO: Find out which method is most reliable (suspect auc2).
	"""
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
	"""Calculates area between observed path and idea straight line.
	
	An alternative to max_deviation
	
	Parameters
	----------
	x, y : array-like
		x and y coordinates of the path.
		
	Returns
	----------
	area : Total area enclosed by the curve and line together
	
	NOTE: auc() and auc2() differ in that the former calcuates area 
	enclosed by the curve and the ideal straight line only, while the 
	latter calculates the area between the curve and the x axis, and then
	subtracts the triangular area between the straight line and the 
	x axis.
	TODO: Find out which method is most reliable (suspect auc2).
	"""
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
	"""Returns array of velocity at each time step"""
	vx = np.ediff1d(x)
	vy = np.ediff1d(y)
	vel = np.sqrt( vx**2 + vy **2 ) # Pythagoras
	return vel

# # # Inference
def bimodality_coef(samp):
	"""Checks sample for bimodality (values > .555)
	
	See `Freeman, J.B. & Dale, R. (2013). Assessing bimodality to detect 
	the presence of a dual cognitive process. Behavior Research Methods.` 
	"""
	n = len(samp)
	m3 = stats.skew(samp)
	m4 = stats.kurtosis(samp, fisher=True)
	#b = ( g**2 + 1) / ( k + ( (3 * (n-1)**2 ) / ( (n-2)*(n-3) ) ) )
	b=(m3**2+1) / (m4 + 3 * ( (n-1)**2 / ((n-2)*(n-3)) ))
	return b

def chisquare_boolean(array1, array2):
	"""Untested convenience function for chi-square test
	
	Parameters
	----------
	array1, array2 : array-like
		Containing boolean values to be tested
		
	Returns
	--------
	chisq : float
		Chi-square value testing null hypothesis that there is an 
		equal proporion of True and False values in each array.
	p : float
		Associated p-value
	"""
	observed_values = np.array([sum(array1), sum(array2)])
	total_len = np.array([len(array1), len(array2)])
	expected_ratio = sum(observed_values) / sum(total_len)
	expected_values = total_len * expected_ratio
	chisq, p = stats.chisquare(observed_values, f_exp = expected_values)
	return chisq, p

# # # Functions to apply to a set of trajectories at a time # # #
# # Most of this is best done using Pandas' built in methods.
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

def plot_means_2d(dataset, groupby, condition_a, condition_b, x='x', y='y', legend=True, title=None):
	"""Depreciated: Convenience function for plotting average 2D mouse paths
	
	Parameters
	----------
	dataset: Pandas DataFrame
	groupby: string
		The column in which the groups are defined
	condition_a, condition_b: string
		The labels of each group (in column groupby)
	x, y: string, optional
		The column names of the coordinates to be compared.
		Default 'x', 'y'
	legend: bool, optional
		Include legend on plot
	title: string, optional

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
	"""Depreciated: Convenience function plotting every trajectory in 2 conditions
	
	Parameters
	----------
	dataset: Pandas DataFrame
	groupby: string
		The column in which the groups are defined
	condition_a, condition_b: string
		The labels of each group (in column groupby)
	x, y: string, optional
		The column names of the coordinates to be compared.
		Default 'x', 'y'
	legend: bool, optional
		Include legend on plot
	title: string, optional
	
	Depreciated: Use:
		``color_map = {'condition_a': 'b', condition_b': 'r'}
		DataFrame.apply(lambda trial: plt.plot(trial['x'], trial['y'], color_map[trial['conditon']])``
		
	Takes a Pandas DataFrame, divides it by the grouping variable 'groupby' 
	(a string), and plots all paths in 'condition_a' in blue,
	and 'condition_b' in red.
	Includes a legend by default, and a title if given."""
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



# Make a GIF 
def make_gif(dataset, groupby, condition_a, condition_b, save_to, frames=101, x='x', y='y'):
	"""Very Experimental!
	
	Creates and saves series of plots, showing position of all paths
	from both conditions, with condition_a in blue, and condition_b in red.
	These can then be combined into an animated GIF file, which shows
	your data in action.
	
	Parameters
	----------
	dataset : Pandas DataFrame containing your data
	groupby, condition_a, condition_b : string
		name of column to group by, and labels for each group in it
	save_to : string
		Path of existing folder to save the images to.
	frames : int, optional
		How many timesteps to animimate.
	x, y : string, optional
		The column names for the x and y variables to visualize
	
	I find this works best with your raw time data, extended so that there's
	an even number of time steps in each trial using ``uniform_time()``,
	rather than using the 101 normalized time steps, although of course the 
	choice is yours.
	
	Convert the save images to a gif using ImageMagick
	
	From the command line (Linux, but should also work in Windows/OSX)
	``convert -delay 10 -loop 1 path/to/images/*.png path/to/save/Output.gif``
	"""
	for i in range(frames):
	  plt.clf()
	  for j in range(len(data)):
	  	  if data.code.iloc[j] == 'lure':
	  	  	  style = 'r.'
	  	  elif data.code.iloc[j] == 'control':
	  	  	  style = 'b.'
	  	  else:
	  	  	  style = None
	  	  if style:
	  	  	  x = data.fx.iloc[j]
	  	  	  y = data.fy.iloc[j]
	  	  	  if len(x) > i:
	  	  	  	  plt.plot(x[i], y[i], style)
	  	  	  else:
	  	  	  	  plt.plot(x[-1], y[-1], style)
	  plt.xlim((-1.2, 1.2))
	  plt.ylim((-.2, 1.2))
	  plt.title('%ims' % (i*10))
	  plt.savefig(os.path.join(path, 'Step_%i.png' % (1000+i)))

def angular_deviation(x, y, t=None, response_x=1, response_y=1, alt_x=-1, alt_y=1, normalized=False):
	"""
	Shows how far, in degrees, the path deviated from going straight to the response,
	at every step along the way.
	
	Parameters
	----------
	x, y : Pandas Series objects (including TimeSeries)
		The mouse coordinates
	response_x, response_y, alt_x, alt_y : int
		The locations of the responses
	normalized : Bool
		Not implemented: Normalize the result, so that straight towards
		the response returns 0, and straight towards the alternative
		returns 1.
	"""
	# Generate vectors of the actual change in position,
	# and distance from the chosen and alternative responses,
	# at each time step.
	dx, dy = x.diff(), y.diff()
	response_dx = response_x - x
	response_dy = response_y - y
	alt_dx = alt_x - x
	alt_dy = alt_y - y
	# Use those vectors to calculate the respective angles.
	actual_angle = np.arctan2(dy, dx)
	angle_to_response = np.arctan2(response_dy, response_dx)
	angle_to_alt = np.arctan2(alt_dy, alt_dx)
	# Where cursor was stationary, give angle as 0
	velocity = np.sqrt(dx**2 + dy**2)
	actual_angle *= (velocity > .05)
	angle_to_alt *= (velocity > .05)
	angle_to_response *= (velocity > .05)
	# Deviation is the difference between the actual angle, and the
	# angle going straight for the response
	deviation_angle = (actual_angle - angle_to_response) # Reverse signs?
	if t == None:
		t = range(len(dx))
	if normalized: # This doesn't work
		raise Exception("normalization isn't implemented yet for angular_deviation")
		normal = (deviation_angle - angle_to_response) / (angle_to_alt - angle_to_response)
		return normal
	else:
		return deviation_angle

def movement_angle(x, y):
	try:
		# TimeSeries
		dx, dy = x.diff(), y.diff()
	except AttributeError:
		# Array
		dx, dy = np.ediff1d(x), np.ediff1d(y)
	# Measuring from the y axis
	angle = np.arctan2(dx, dy)
	velocity = np.sqrt(dx**2 + dy**2)
	angle *= (velocity > .05) # Treat steps that move less than this as 0 degrees
	#return  1.5707963267948966 - angle # 90 degrees minus angle
	return np.nan_to_num(angle) # Maybe leave in the NAN for better averaging?

def movement_angle2(x, y):
	try:
		# TimeSeries
		dx, dy = x.diff(), y.diff()
	except AttributeError:
		# Array
		dx, dy = np.ediff1d(x), np.ediff1d(y)
	# Measuring from the y axis
	angle = np.arctan2(dx, dy)
	velocity = np.sqrt(dx**2 + dy**2)
	angle *= (velocity > .01) # Treat steps that move less than this as 0 degrees
	angle.iloc[0] = 0
	for i in np.arange(1, angle.size):
		if angle.iloc[i] == 0:
			angle.iloc[i] = angle.iloc[i-1]
	return angle # Maybe leave in the NAN for better averaging?

def tsplot(MetaSeries):
	"""Does what must be done to turn a Pandas column of Serieses into something that SeaBorn can deal with"""
	x = range(len(MetaSeries.iloc[0]))
	sns.tsplot(x, np.array( [np.array(trial) for trial in MetaSeries]))

def smooth_gaussian(array ,degree=5):
	"""
	Smoothes jagged, oversampled time series data.
	
	Parameters
	----------
	array : 
		TimeSeries to smooth
	degree : int, optional, default=5
		window over which to smooth
		
	Code from http://www.swharden.com/blog/2008-11-17-linear-data-smoothing-in-python/
	With thanks to  Scott W Harden
	"""
	window=degree*2-1  
	weight=np.array([1.0]*window)  
	weightGauss=[]  
	for i in range(window):  
		i=i-degree+1  
		frac=i/float(window)  
		gauss=1/(np.exp((4*(frac))**2))  
		weightGauss.append(gauss)  
	weight=np.array(weightGauss)*weight  
	smoothed=[0.0]*(len(array)-window)  
	for i in range(len(smoothed)):  
		smoothed[i]=sum(np.array(array[i:i+window])*weight)/sum(weight)  
	return smoothed  


# Binning data
def map_bin(x, bins):
	kwargs = {}
	if x == max(bins):
		kwargs['right'] = True
	bin = bins[np.digitize([x], bins, **kwargs)[0]]
	bin_lower = bins[np.digitize([x], bins, **kwargs)[0]-1]
	return bin_lower

def bin_series(series, bins=None):
	if bins == None:
		maximum = series.max()
		bins = np.arange(0, maximum+.1, maximum/len(series))
	binned = series.index.map(lambda s: map_bin(s, bins))
	raw = pd.DataFrame(data=zip(series, binned), index=series.index, columns=['val', 'bin'])
	grouped = raw.groupby('bin').mean()
	return pd.TimeSeries(grouped.val, bins)

## Advanced AUC functions
def auc_even_odd(x, y, resolution=.05):
	try:
		start_x, end_x, start_y, end_y = x.iloc[0], x.iloc[-1], y.iloc[0], y.iloc[-1]
	except:
		# Not a Pandas Series
		start_x, end_x, start_y, end_y = x[0], x[-1], y[0], y[-1]
	points_under_curve = []
	for px in np.arange(-end_x, end_x+.01, resolution):
		for py in np.arange(start_y, end_y+.01, resolution):
			test = even_odd_rule(px, py, x, y)
			if test:
				plt.plot(px, py, 'or')
			points_under_curve.append(test)
	auc = float(sum(points_under_curve)) / len(points_under_curve)
	return auc

def even_odd_rule(point_x, point_y, line_x, line_y):
	# Possibly use a sparse sample of the lines to speed this up?
	poly = zip(line_x, line_y)
	num = len(poly)
	i = 0
	j = num - 1
	c = False
	for i in range(num):
			if  ((poly[i][1] > point_y) != (poly[j][1] > point_y)) and \
					(point_x < (poly[j][0] - poly[i][0]) * (point_y - poly[i][1]) / (poly[j][1] - poly[i][1]) + poly[i][0]):
				c = not c
			j = i
	return c

# Make a GIF (
#~ path = '/path/to/save/images/'
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
# cd /path/to/save/images/
# convert -delay 10 -loop 1 *.png Output.gif
