#~ from __future__ import division
import os
import pandas as pd
import numpy as np
from scipy import interpolate, interp, stats
from math import sqrt
import math
import warnings


###########################
# Normalization functions #
###########################
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

def normalize_space(array, start=0, end=1, preserve_direction=False):
	"""Interpolates array of 1-d coordinates to given start and end value.
	 
	Parameters
	----------
	array : array-like
		array of coordinates to interpolate
	start : numeric
		Default: 0
		Value to interpolate first coordinate to
	end : numeric
		Default: 1
		Value to interpolate last coordinate to		
	preserve_direction : boolean
		Default: False
		If False, decrasing coordinates (i.e. where the
		last value is less than the first) will remain decreasing
		in the output. If start=0, and end=1, increasing coordinates will output [0, ..., 1],
		while decreasing coordinates will output [0, ..., -1].
		If False, all output will run [0, ..., 1].
	
	TODO: Might not work on decreasing arrays. Test this"""
	# Convert data to numpy array if it's a regular list
	if type(array) != np.ndarray:
		if type(array) == list:
			array = np.array(array)
		else:
			raise TypeError("You've used an input of type %s.\nPlease input either a Numpy array or a list.\nThe value you used was:\n%s" % (type(array), repr(array)))
	reverse_when_done = False
	# 'Delta' denotes distance between start and end values
	old_delta = array[-1] - array[0]
	# Check if decreasing 
	if old_delta < 0:
		array = array * -1
		old_delta = array[-1] - array[0]
		if preserve_direction:
			reverse_when_done = True
	new_delta = end - start
	# Convert coordinates to float values
	array = array.astype('float')
	# Finally, interpolate. We interpolate from (start - 2*delta) to (end + 2*delta)
	# to handle cases where values go below the start, or above the end values.
	
	# Note that in rare cases where the coordinates manage to go outside of
	# this range on either side, this might cause problems.
	# For now, I'm going to catch this as an error, because I
	# don't know how wide I need the function to be.
	# In future, I should probably test the range of the data, and
	# interpolate based on that.
	if max(array) > array[-1] + 2*old_delta or min(array) < array[0] - 2*old_delta:
		raise ValueError("Input included values way outside of the range\
		of your data, given the start and end coordinates.\n\
		This shouldn't happen.")
	old_range = np.array([array[0] - 2*old_delta, array[-1]+2*old_delta])
	new_range = np.array([start - 2*new_delta, end + 2*new_delta])
	#if max(array) > old_range[-1]:
		#print 'Array: %s , old_range: %s' % (array, old_range)
	normal = np.interp(array, old_range, new_range)
	if reverse_when_done:
		return normal * -1
	else:
		return normal
	
def remap_right(array):
	"""Flips decreasing coordinates horizontally on their origin
	
	>>> remap_right([10,11,12])
	array([10, 11, 12])

	>>> remap_right([10, 9, 8])
	array([10, 11, 12])
	
	Parameters
	----------
	array : array-like
		array of coordinates to remap
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
	"""Parses string represation of list '[1,2,3]' to an actual pythonic list [1,2,3]
	
	A rough and ready function"""
	try:
		first = string_list.strip('[]')
		then = np.array(first.split(','))
		lastly = then.astype(float)
		return lastly
	except:
		warnings.warn('There was an error parsing one of your strings to a list.\n\
		`None` value used instead', UserWarning)
		return None

# # # Functions to apply to a single trajectory at a time # # #
def distance_from_response(x, y, from_foil=False):
	"""Calculate distance from the ultimate response for each step of a trajectory.
	If `from_foil` is True, shows distance from the foil response,
	assuming the foil is located on the opposite side of the x axis to the
	response.
	
	Parameters
	x, y: array-like
		Coordinates of trajectory
	from_foil: boolean, optional, default False
		If `True`, return distance to the non-chosen response.
	"""
	# TODO make these reference targets flexible as input
	
	# Infer response locations
	response_x, response_y = (x[-1], y[-1])
	if from_foil:
		response_x *= -1
	# Get distance from them along paths
	distance = np.sqrt( (x - response_x)**2 + (y - response_y)**2 )
	return distance


def get_init_time(t, y, y_threshold=.01, ascending = True):
	"""Returns time  from t of point where y exceeds y_threshold.
	 TODO - Replace this with faster code using generators
	init_time = next(time for (time, location) in zip(tList, yList) if location < (h-30)) # More than 30 pixels from bottom

	
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

#~ def rotate(x, y, rad): # Redundant
	#~ """Rotate counter-clockwise around origin by `rad` radians.
	#~ """
	#~ s, c = [f(rad) for f in (math.sin, math.cos)]
	#~ x, y = (c*x - s*y, s*x + c*y)
	#~ return x,y

def get_deviation(x, y):
	"""Returns the deviation away from a straight line over the course of a path.
	Calculated by rotating the trajectory so that it starts at (0, 0),
	and ends at (0, 1), so that the x-axis coordinates represent the deviation
	
	Parameters
	----------
	x, y : array-like
		x and y coordinates of the path.
		
	Returns
	----------
	deviation : np.array
		Distance between observed and straight line at every step
	
	
	"""
	path = np.array(zip(x, y)).T
	# Turn the path on its side.
	radians_to_rotate = math.atan(float(x[len(x)-1])/y[len(y)-1]) # Handling Pandas dataforms
	rotMatrix = np.array([[np.cos(radians_to_rotate), -np.sin(radians_to_rotate)], \
						[np.sin(radians_to_rotate),  np.cos(radians_to_rotate)]])
	deviation, deviation_y = rotMatrix.dot(path)
	return -1 * deviation # Reverse the sign

def max_deviation(x, y, allow_negative=True):
	"""Caluclate furthest distance between observed path and ideal straight one.
	
	Parameters
	----------
	x, y : array-like
		x and y coordinates of the path.
	allow_negative : boolean, optional, default True
		If False, ignore deviation AWAY from foil response.
		
	Returns
	----------
	max_dev : Greatest distance between observed and straight line.
	
	As with the rest of Squeak, this assumes a line running from bottom center
	(0, 0) to top right (1, 1), or (1, 1.5), as it relies on rotating the
	line 45 degrees anticlockwise and comparing it to the y axis.
	
	Will return negative values in cases where the greatest distance
	is to the right (i.e. AWAY from the alternative response).
	"""
	# Rotate line to vertical, and get x axis
	deviation = get_deviation(x, y)
	max_positive = abs(max(deviation))
	max_negative = abs(min(deviation))
	#print max_positive, max_negative
	if allow_negative:
		if max_positive > max_negative:
			# The return the positive MD
			return max_positive
		else:
			# Return the negative (rare)
			return -1*max_negative
	else:
		return max_positive

# Area under the curve
# There are a number of ways of calculating this, the results of which 
# don't really differ in their results, except that the even-odd method
# is DEFINITELY more accurate, but very slow.
# I've wrapped all the methods in the same function, with polygon1
# as the default method.

	
def auc(x, y, method='polygon'):
	"""Calculates area between observed path and idea straight line.
	
	An alternative to max_deviation
	
	Parameters
	----------
	x, y : array-like
		x and y coordinates of the path.
	method : string, optional, default 'polygon'
		Method used to calculate area under curve.
		Options are:
			- 'polygon' - Using formula for area of irregular polygon,
			- 'even-odd' - Slower, using the even-odd rule on each pixel
					to see if if falls within the curve
					
	Returns
	----------
	area : Total area enclosed by the curve and line together
	"""
	if method == 'polygon':
		return polygon_auc(x, y)		
	elif method == 'even-odd':
		return even_odd_auc(x, y)
	else:
		raise ValueError("AUC method must be either 'polygon' or 'even-odd'.\n\
		You entered '%s'" % method)
	
def polygon_auc(x, y):
	areas = []
	j = len(x) - 1
	for i in range(len(x)):
		x1y2 = y[i]*x[j]
		x2y1 = x[i] * y[j]
		area = x2y1 - x1y2
		areas.append(area)
		j = i
	return float(sum(areas))/2
	
def even_odd_auc(x, y, resolution=.05, debug=False):
	# This method can be agonizingly slow, as it goes through every pixel,
	# and applies the even-odd rule, which itself goes through every 
	# vector in the trajectory iteratively.
	# It's been speeded up by
	# - using a relatively low resolution (.05)
	#	-> = approx 52x43 = 2236 pixels
	# - Resampling the trajectory by a factor of 5, so it calculates
	#	over closer to 20 vectors, rather than 101+
	# It might be worthwhile trying to implement these graphics functions in C?

	# Finally, the unit of measurement is different here than for the
	# other auc method, and corresponds to % of the screen under curve.
	# I could probably just divide by a scaling factor to leave them both
	# the same?
	
	# Use numpy arrays, and measure from curve to x axis
	x, y = np.array(x), np.array(y)
	start_x, end_x, start_y, end_y = x[0], x[-1], y[0], y[-1]
	#~ x = np.append(x, [end_x, start_x]) # Extending to x axis
	#~ y = np.append(y, [start_y]*2)
	if debug:
		plt.plot(x, y, '-ob')
		# Slower function, but plots a graph
		points_under_curve = []
		for px in np.arange(-end_x*1.51, end_x*1.51, resolution):
			for py in np.arange(start_y-.2, end_y+.3, resolution):
				test = even_odd_rule(px, py, x, y)
				if test:
					plt.plot(px, py, 'or')
				points_under_curve.append(test)
	else:
		points_under_curve = [even_odd_rule(px, py, x, y) \
		for px in np.arange(-end_x*1.51, end_x*1.51, resolution) \
			for py in np.arange(start_y-.2, end_y+.3, resolution)]
	auc = float(sum(points_under_curve)) / len(points_under_curve)
	return auc

def even_odd_rule(point_x, point_y, line_x, line_y, resample=5):
	# Possibly use a sparse sample of the lines to speed this up?
	#~ poly = zip(line_x[::5], line_y[::5])
	
	# Look at other optimization methods
	line_x, line_y = line_x[::resample], line_y[::resample]
	line_x = np.append(line_x, [line_x[-1], line_x[0]]) # Extending to x axis
	line_y = np.append(line_y, [line_y[0]]*2) # Extending to x axis
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


def pythag(o, a):
	return np.sqrt( o**2 + a**2)

def velocity(x, y):
	"""Returns array of velocity at each time step"""
	if isinstance(x, pd.Series):
		x = x.values
		y = y.values
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

#~ def chisquare_boolean(array1, array2):
	#~ """Untested convenience function for chi-square test
	#~ 
	#~ Parameters
	#~ ----------
	#~ array1, array2 : array-like
		#~ Containing boolean values to be tested
		#~ 
	#~ Returns
	#~ --------
	#~ chisq : float
		#~ Chi-square value testing null hypothesis that there is an 
		#~ equal proporion of True and False values in each array.
	#~ p : float
		#~ Associated p-value
	#~ """
	#~ observed_values = np.array([sum(array1), sum(array2)])
	#~ total_len = np.array([len(array1), len(array2)])
	#~ expected_ratio = sum(observed_values) / sum(total_len)
	#~ expected_values = total_len * expected_ratio
	#~ chisq, p = stats.chisquare(observed_values, f_exp = expected_values)
	#~ return chisq, p

# # # Functions to apply to a set of trajectories at a time # # #
# These are depreciated. Move to submodule.
# # Most of this is best done using Pandas' built in methods.
#~ def average_path(x, y, full_output=False):#, length=101):
	#~ """Averages Pandas data columns of x and y trajectories into a single mean xy path.
	#~ 
	#~ Finds length of first row, and then averages the i-th entry of the x and y columns
	#~ for i in range(0,length).
	#~ 
	#~ Parameters
	#~ ----------
	#~ x, y : Pandas DataFrame columns
	#~ full_output : bool, optional
		#~ Return all values, not just the average.
		#~ Used by compare_means_1d()
	#~ TODO: Allow other datatypes as input.
	#~ TODO: Create the option of returning variance as well as mean?
		#~ See http://stanford.edu/~mwaskom/software/seaborn/timeseries_plots.html
	#~ """
	#~ # Can this be done more efficiently with .apply()?
	#~ mx, my = [], []
	#~ fullx, fully = [], []
	#~ length = len(x.iloc[0])
	#~ for i in range(length):
		#~ this_x, this_y = [], []
		#~ for p in range(len(x)):
			#~ this_x.append(x.iloc[p][i])
			#~ this_y.append(y.iloc[p][i])
		#~ if full_output:
			#~ fullx.append(this_x)
			#~ fully.append(this_y)
		#~ mx.append(np.mean(this_x))
		#~ my.append(np.mean(this_y))
	#~ if full_output:
		#~ return fullx, fully
	#~ return np.array(mx), np.array(my)
	
#~ def compare_means_1d(dataset, groupby, condition_a, condition_b, y = 'x', test = 't', length=101):
	#~ """Possibly depreciated: Compares average coordinates from two conditions using a series of t or Mann-Whitney tests.
	#~ 
	#~ Parameters
	#~ ----------
	#~ dataset: Pandas DataFrame
	#~ groupby: string
		#~ The column in which the groups are defined
	#~ condition_a, condition_b: string
		#~ The labels of each group (in column groupby)
	#~ y: string, optional
		#~ The column name of the coordinates to be compared.
		#~ Default 'x'
	#~ test: string, optional
		#~ Statistical test to use.
		#~ Default: 't' (independant samples t test)
		#~ Alternate: 'u' (Non-parametric Mann-Whitney test)
		#~ 
	#~ Returns
	#~ -----------
	#~ t101 : t (or U) values for each point in the trajectory
	#~ p101 : Associated p values"""
	#~ a_x, a_y = average_path(dataset[y][dataset[groupby] == condition_a], dataset[y] [dataset[groupby] == condition_a], full_output=True)
	#~ b_x, b_y = average_path(dataset[y][dataset[groupby] == condition_b], dataset[y][dataset[groupby] == condition_b], full_output=True)
	#~ t101, p101 = [], []
	#~ for i in range(length):
		#~ if test == 't':# t-test
			#~ t, p = stats.ttest_ind(a_y[i], b_y[i])
		#~ elif test == 'u':# Mann-Whitney
			#~ t, p = stats.mannwhitneyu(a_y[i], b_y[i])
		#~ t101.append(t)
		#~ p101.append(p)
	#~ return t101, p101
	
# Depreciated Plotting functions
#~ def plot_means_1d(dataset, groupby, condition_a, condition_b, y = 'x', legend=True, title=None):
	#~ """Depreciated: Convenience function for plotting two 1D lines, representing changes on x axis
	#~ 
		#~ Parameters
	#~ ----------
	#~ dataset: Pandas DataFrame
	#~ groupby: string
		#~ The column in which the groups are defined
	#~ condition_a, condition_b: string
		#~ The labels of each group (in column groupby)
	#~ y: string, optional
		#~ The column name of the coordinates to be compared.
		#~ Default 'x'
	#~ legend: bool, optional
		#~ Include legend on plot
	#~ title: string, optional
#~ 
	#~ Takes a Pandas DataFrame, divides it by the grouping variable 'groupby' 
	#~ (a string), and plots the average of all paths in 'condition_a' in blue,
	#~ and the average from 'condition_b' in red.
	#~ Includes a legend by default, and a title if given."""
	#~ a_x, a_y = average_path(dataset[y][dataset[groupby] == condition_a], dataset[y][dataset[groupby] == condition_a])
	#~ b_x, b_y = average_path(dataset[y][dataset[groupby] == condition_b], dataset[y][dataset[groupby] == condition_b])
	#~ l1 = plt.plot(a_y, color = 'r', label = condition_a)
	#~ l2 = plt.plot(b_y, 'b', label=condition_b)
	#~ if legend:
		#~ plt.legend()
	#~ plt.title(y)
	#~ return None

def plot_means_2d(dataset, groupby, condition_a, condition_b, x='x', y='y', legend=True, title=None):
	#~ """Depreciated: Convenience function for plotting average 2D mouse paths
	#~ 
	#~ Parameters
	#~ ----------
	#~ dataset: Pandas DataFrame
	#~ groupby: string
		#~ The column in which the groups are defined
	#~ condition_a, condition_b: string
		#~ The labels of each group (in column groupby)
	#~ x, y: string, optional
		#~ The column names of the coordinates to be compared.
		#~ Default 'x', 'y'
	#~ legend: bool, optional
		#~ Include legend on plot
	#~ title: string, optional
#~ 
	#~ Takes a Pandas DataFrame, divides it by the grouping variable 'groupby' 
	#~ (a string), and plots the average of all paths in 'condition_a' in blue,
	#~ and the average from 'condition_b' in red.
	#~ Includes a legend by default, and a title if given."""
	#~ a_x, a_y = average_path(dataset[x][dataset[groupby] == condition_a], dataset[y][dataset[groupby] == condition_a], length=length)
	#~ b_x, b_y = average_path(dataset[x][dataset[groupby] == condition_b], dataset[y][dataset[groupby] == condition_b], length=length)
	#~ l1 = plt.plot(a_x, a_y, color = 'b', label = condition_a)
	#~ l2 = plt.plot(b_x, b_y, 'r', label=condition_b)
	#~ if legend:
		#~ plt.legend()
	#~ if title:
		#~ plt.title(title)
	#~ return a_x, a_y, b_x, b_y

#~ def plot_all(dataset, groupby, condition_a, condition_b, x='x', y='y', legend=True, title=None):
	#~ """Depreciated: Convenience function plotting every trajectory in 2 conditions
	#~ 
	#~ Parameters
	#~ ----------
	#~ dataset: Pandas DataFrame
	#~ groupby: string
		#~ The column in which the groups are defined
	#~ condition_a, condition_b: string
		#~ The labels of each group (in column groupby)
	#~ x, y: string, optional
		#~ The column names of the coordinates to be compared.
		#~ Default 'x', 'y'
	#~ legend: bool, optional
		#~ Include legend on plot
	#~ title: string, optional
	#~ 
	#~ Depreciated: Use:
		#~ ``color_map = {'condition_a': 'b', condition_b': 'r'}
		#~ DataFrame.apply(lambda trial: plt.plot(trial['x'], trial['y'], color_map[trial['conditon']])``
		#~ 
	#~ Takes a Pandas DataFrame, divides it by the grouping variable 'groupby' 
	#~ (a string), and plots all paths in 'condition_a' in blue,
	#~ and 'condition_b' in red.
	#~ Includes a legend by default, and a title if given."""
	#~ for i in range(len(dataset)):
		#~ y_path = dataset[y].iloc[i]
		#~ if type(x) == list:
			#~ x_path = x
		#~ elif x == 'time':
			#~ x_path = range(len(y_path))
		#~ else:
			#~ x_path = dataset[x].iloc[i]
		#~ if dataset[groupby].iloc[i] == condition_a:
			#~ plt.plot(x_path, y_path, 'b')
		#~ elif dataset[groupby].iloc[i] == condition_b:
			#~ plt.plot(x_path, y_path, 'r')



# Make a GIF 
#~ def make_gif(dataset, groupby, condition_a, condition_b, save_to, frames=101, x='x', y='y'):
	#~ """Very Experimental!
	#~ 
	#~ Creates and saves series of plots, showing position of all paths
	#~ from both conditions, with condition_a in blue, and condition_b in red.
	#~ These can then be combined into an animated GIF file, which shows
	#~ your data in action.
	#~ 
	#~ Parameters
	#~ ----------
	#~ dataset : Pandas DataFrame containing your data
	#~ groupby, condition_a, condition_b : string
		#~ name of column to group by, and labels for each group in it
	#~ save_to : string
		#~ Path of existing folder to save the images to.
	#~ frames : int, optional
		#~ How many timesteps to animimate.
	#~ x, y : string, optional
		#~ The column names for the x and y variables to visualize
	#~ 
	#~ I find this works best with your raw time data, extended so that there's
	#~ an even number of time steps in each trial using ``uniform_time()``,
	#~ rather than using the 101 normalized time steps, although of course the 
	#~ choice is yours.
	#~ 
	#~ Convert the save images to a gif using ImageMagick
	#~ 
	#~ From the command line (Linux, but should also work in Windows/OSX)
	#~ ``convert -delay 10 -loop 1 path/to/images/*.png path/to/save/Output.gif``
	#~ """
	#~ for i in range(frames):
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

def movement_angle(x, y, step_by=5):
	original_index = x.index
	x, y = x[::step_by], y[::step_by]
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
	angle = np.nan_to_num(angle) # Maybe leave in the NAN for better averaging?
	# Recreate with the original index
	return angle.reindex(index=original_index).interpolate()

#~ def simple_angle(x1, y1, x2, y2):
	#~ dx = x2 - x1
	#~ dy = y2 - y1
	#~ return np.arctan2(dx, dy)
#~ 
#~ def angle_to_point(x, y, point_x=1, point_y=1.5):
    #~ dx = point_x - x
    #~ dy = point_y - y
    #~ angle = np.arctan2(dx, dy)
    #~ return angle
#~ 
#~ def movement_angle2(x, y):
	#~ try:
		#~ # TimeSeries
		#~ dx, dy = x.diff(), y.diff()
	#~ except AttributeError:
		#~ # Array
		#~ dx, dy = np.ediff1d(x), np.ediff1d(y)
	#~ # Measuring from the y axis
	#~ angle = np.arctan2(dx, dy)
	#~ velocity = np.sqrt(dx**2 + dy**2)
	#~ angle *= (velocity > .01) # Treat steps that move less than this as 0 degrees
	#~ angle.iloc[0] = 0
	#~ for i in np.arange(1, angle.size):
		#~ if angle.iloc[i] == 0:
			#~ angle.iloc[i] = angle.iloc[i-1]
	#~ return angle # Maybe leave in the NAN for better averaging?


#~ def get_xflips(path):
	#~ if type(path) != np.ndarray:
		#~ path = np.array(path)
	#~ new_path = path[::5]
	#~ flips = 0
	#~ for i in range(len(new_path)-1):
		#~ this, next_point = new_path[i], new_path[i+1]
		#~ if np.sign(this) != np.sign(next_point):
			#~ #print '
			#~ flips += 1
	#~ return flips
		  

#~ def relative_attraction(trial, xvar='nx', yvar='ny'):
	#~ x, y = trial[xvar], trial[yvar]
	#~ try:
		#~ end_x, end_y = x[-1], y[-1]
	#~ except KeyError:
		#~ end_x, end_y = x.iloc[-1], y.iloc[-1]
	print end_x, end_y
	#~ alt_end_x, alt_end_y = end_x*-1, end_y
	#~ response_x_dist = end_x - x
	#~ alt_x_dist = alt_end_x - x
	#~ common_y_dist = end_y - y
	#~ response_dist = -np.sqrt(response_x_dist**2 + common_y_dist**2)
	#~ alt_dist = np.sqrt(alt_x_dist**2 + common_y_dist**2)
	#~ response_attraction = (-1*response_dist).diff()
	#~ alt_attraction = (-1*alt_dist).diff()
	#~ return response_attraction + alt_attraction # Add, because alt is negative



#~ def tsplot(MetaSeries):
	#~ """Does what must be done to turn a Pandas column of Serieses into something that SeaBorn can deal with"""
	#~ x = range(len(MetaSeries.iloc[0]))
	#~ sns.tsplot(x, np.array( [np.array(trial) for trial in MetaSeries]))

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
#~ def map_bin(x, bins):
	#~ kwargs = {}
	#~ if x == max(bins):
		#~ kwargs['right'] = True
	#~ bin = bins[np.digitize([x], bins, **kwargs)[0]]
	#~ bin_lower = bins[np.digitize([x], bins, **kwargs)[0]-1]
	#~ return bin_lower
#~ 
#~ def bin_series(series, bins=None):
	#~ if bins == None:
		#~ maximum = series.max()
		#~ bins = np.arange(0, maximum+.1, maximum/len(series))
	#~ binned = series.index.map(lambda s: map_bin(s, bins))
	#~ raw = pd.DataFrame(data=zip(series, binned), index=series.index, columns=['val', 'bin'])
	#~ grouped = raw.groupby('bin').mean()
	#~ return pd.TimeSeries(grouped.val, bins)



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

#### From http://wiki.scipy.org/Cookbook/SignalSmooth
def smooth(x,window_len=11,window='hanning'):
    """smooth the data using a window with requested size.
    
    http://wiki.scipy.org/Cookbook/SignalSmooth
   """

    if x.ndim != 1:
        raise ValueError, "smooth only accepts 1 dimension arrays."
    if x.size < window_len:
        raise ValueError, "Input vector needs to be bigger than window size."
    if window_len<3:
        return x
    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError, "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'"
    s=np.r_[x[window_len-1:0:-1],x,x[-1:-window_len:-1]]
    #print(len(s))
    if window == 'flat': #moving average
        w=np.ones(window_len,'d')
    else:
        w=eval('np.'+window+'(window_len)')
    y=np.convolve(w/w.sum(),s,mode='valid')
    return y
    
def smooth_timeseries(series, window_len=11, window='hanning'):
	original_index = series.index
	smoothed = smooth(series, window_len, window)
	# Return to original length via interpolation
	interpolated = np.interp(np.linspace(0, len(smoothed), len(original_index)),\
		range(len(smoothed)), smoothed)
	return pd.TimeSeries(interpolated, original_index)

	
def jitter(array, scale=.1):
    return array + np.random.normal(scale=scale*np.std(array), size=len(array))

# Dale's (2011) functions
# Not completely satisfied with these yet (21/7/14)
def acceleration_components(velocity):
	if isinstance(velocity, pd.Series):
		velocity = velocity.values
	acc = np.ediff1d(velocity)
	#~ components = [np.sign(acc[t]) != np.sign(acc[t+1]) for t in range(len(acc)-1)]
	c = acc[:-1] * acc[1:]
	components = c < -.000001 # Anything smaller counts as not moving at all.
	#~ components = np.concatenate([[False], components]) # Make length match velocity length
	return np.array(components)
	#~ return sum(components)# There will always be 2 flips, corresponding to the
	# start and end of the movement
	
def x_flips(x):
	if isinstance(x, pd.Series):
		x = x.values
	dx = np.ediff1d(x)
	#~ flips = [np.sign(dx[t]) != np.sign(dx[t+1]) for t in range(len(dx)-1)]
	changes = dx[:-1] * dx[1:]
	flips = changes < -.00001 # Anything smaller counts as not moving at all.
	#~ flips = np.concatenate([[False], flips]) # Make length match velocity length

	#~ plt.plot(dx, 'r-o', np.array([0]+flips), 'b')
	return np.array(flips)
	#~ return sum(flips) - 2 # There will always be 2 flips, corresponding to the
	# start and end of the movement

# Analyting as eye data
def ballistic_direction(x):
	#~ if isinstance(x, pd.Series):
		#~ x = x.values
	#~ direction = np.sign(np.ediff1d(x))
	dx = x.diff()
	side_of_screen = np.sign(x)
	direction = np.sign(dx)
	direction.iloc[0] = 0
	sizable = dx.abs() > .01 # Disregard movements smaller than this
	direction *= sizable
	# If the mouse was in motion, and then stops agin,
	# report the direction it was moving in before it stopped.
	# Doing this in a loop means that time spend hovering over a response
	# is coded as moving towards it, rather than not moving at all.
	#~ for i in range(1, len(direction)):
		#~ if direction.iloc[i] == 0 and direction.iloc[i-1] != 0:
			#~ direction.iloc[i] = direction.iloc[i-1]
	# Alternative - Replace late 0 values of direction with side of screen
	for i in range(1, len(direction)):
		if direction.iloc[i] == 0 and direction.iloc[i-1] != 0:
			direction.iloc[i] = side_of_screen[i-1]
	return direction

	
