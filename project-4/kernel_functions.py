import numpy as np
np.set_printoptions(suppress=True)
import pandas as pd
import sklearn
import dill as pickle
from joblib import dump, load
import multiprocessing as mp
import datetime


from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import cross_val_score

# Uncomment the following 3 lines if you're getting annoyed with warnings from sklearn
import warnings
from sklearn.exceptions import DataConversionWarning
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=DeprecationWarning)
warnings.filterwarnings(action='ignore', category=DataConversionWarning)

connections_matrix = np.empty((1,1))
kernel_data = np.empty((1,1))
distance_matrix = np.empty((1,1))

def friend_distance(x1, x2):
    # distance from each other in graph = 1/distance
	# ['Id', 'Hour1', 'Hour2', 'Hour3', 'Posts', 'Friends', 'Posts_st']
	return connections_matrix[int(x1[0]), int(x2[0])] + 1
	# if you don't have many friends, but you are friends - friend_distance multiplied by 1/log number of combined friends

def friends_in_common(x1, x2):
	# ['Id', 'Hour1', 'Hour2', 'Hour3', 'Posts', 'Friends', 'Posts_st']
	val = np.dot(connections_matrix[int(x1[0])], connections_matrix[int(x2[0])])
	print(val, "friends in common")
	return val
    # return dot product of rows from friends matrix (divide by total friends?? - make not quite +1 for each - +1 if every friend in common)

def friend_kernel(x1, x2):
	# ['Id', 'Hour1', 'Hour2', 'Hour3', 'Posts', 'Friends', 'Posts_st']
	print("--------------------------------------------------------------------")
	print(x1)
	print(x2)
	if total_friends(x1, x2) == 0:
		print("no friends")
		return 0
	val = (friend_distance(x1, x2) / np.log(total_friends(x1, x2))) + (2 * friends_in_common(x1, x2) / total_friends(x1, x2))
	return val
	# same thing above for each person / 2

def total_friends(x1, x2):
	# ['Id', 'Hour1', 'Hour2', 'Hour3', 'Posts', 'Friends', 'Posts_st']
	return x1[5] + x2[5]

def posts_kernel(x1, x2):
	# ['Id', 'Hour1', 'Hour2', 'Hour3', 'Posts', 'Friends', 'Posts_st']
	x1STPosts = x1[6]
	x2STPosts = x2[6]

	val = (8 - np.absolute(x1STPosts - x2STPosts)) / 8
	return val

def hours_kernel(x1, x2):
	#### might be best to use this one only for the longitude attribute? can't tell what's going on / if it gives us anything
	#### this method is working in theory, but there isn't much variation in the values it outputs
	# ['Id', 'Hour1', 'Hour2', 'Hour3', 'Posts', 'Friends', 'Posts_st']
	# print(x1)
	# print(x2)
	similarity = 0
	for x in range(1, 4):
		for y in range(1, 4):
			# print("~~~~~~")
			hourA = x1[x]; hourB = x2[y]
			hourA24 = hourA+24; hourB24 = hourB+24
			# print(hourA, hourB, hourA24, hourB24)
			if hourA == 25 or hourB == 25: ## accounts for the missing data issue
				# print("passing")
				continue
			diff = min(abs(hourA - hourB), abs(hourA24 - hourB), abs(hourA - hourB24)) ## accounts for the "midnight" issue
			diff = 1 - (diff / 24) ## values close to 1 mean v similar, close to 0 mean not similar
			weight = 1 - abs(x-y)/2 ## weights the difference based on which hour pair is being considered
			# print(diff, weight)
			weighted_diff = diff * weight
			similarity += weighted_diff

	return similarity
	### max similarity possible is 6.333
	# print("similarity:", similarity)
	# print("-------------------")

def kernel_fn(function):

	x1 = kernel_data.as_matrix(); x2 = kernel_data.as_matrix()

	print(function)

	print(function.__name__)
	K = np.zeros((x1.shape[0], x2.shape[0]))
	print(x1.shape, x2.shape, K.shape)

	for i in range(x1.shape[0]):
		for j in range(x2.shape[0]):
			val = function(x1[i], x2[j])
			K[i][j] = val
			if j == 100:
				return K
			if j % 10000 == 0 and i % 10000 == 0:
				print("update:", function.__name__, "is at", str((i, j)), "time is", datetime.datetime.now())

	if function == total_friends:
		print(datetime.datetime.now())
		print("saving total friends")
		dump(K, '/mnt/disks/mount/total_friends_kernel_matrix.joblib')
		print("done saving total friends")
	if function == posts_kernel:
		print(datetime.datetime.now())
		print("saving posts")
		dump(K, '/mnt/disks/mount/posts_kernel_matrix.joblib')
		print("done saving posts")
	if function == hours_kernel:
		print(datetime.datetime.now())
		print("saving hours")
		dump(K, '/mnt/disks/mount/hours_kernel_matrix.joblib')
		print("done saving hours kernel")
	if function == friend_kernel:
		print(datetime.datetime.now())
		print("saving friends")
		dump(K, '/mnt/disks/mount/friends_kernel_matrix.joblib')
		print("done saving friends kernel")
	else:
		print(datetime.datetime.now())
		print("saving soemthing")
		dump(K, '/mnt/disks/mount/unknown_function_kernel_matrix.joblib')
		print("done saving something")
	print("returning K")
	return K

def compute_kernels(df):
	df = df.as_matrix()
	# print(df)

	print("running total friends")
	fr = kernel_fn(total_friends, df, df)
	print("saving total friends")
	dump(fr, 'pickles/total_friends_kernel_matrix.joblib') 
	print("saved total friends")

	print("running posts")
	pst = kernel_fn(posts_kernel, df, df)
	print("saving posts")
	dump(pst, 'pickles/posts_kernel_matrix.joblib') 
	print("saved posts")

	print("running hours")
	hrs = kernel_fn(hours_kernel, df, df)
	print("saving hours")
	dump(hrs, 'pickles/hours_kernel_matrix.joblib') 
	print("saved hours")

def parse_graph(training, testing):

	lines = np.loadtxt("graph.txt", comments="#", delimiter="\t", unpack=False)

	m = int(np.amax(lines))
	connections = np.zeros((m+1, m+1))
	# num_friends = np.zeros((m+1, m+1))

	maps = {}
	for line in lines:
		connections[int(line[0])][int(line[1])] = 1
		# num_friends[int(line[0])] += 1
		if line[0] in maps:
			maps[line[0]].append(line[1])
		else:
			maps[line[0]] = [line[1]]

	for i, row in training.iterrows():
		if row['Id'] in maps:
			training.at[i, 'Friends'] = len(maps[row['Id']])
		else:
			training.at[i, 'Friends'] = 0

	for i, row in testing.iterrows():
		if row['Id'] in maps:
			testing.at[i, 'Friends'] = len(maps[row['Id']])
		else:
			testing.at[i, 'Friends'] = 0


	# training['Friends'] = 0; testing['Friends'] = 0
	# for i, row in training.iterrows():
	# 	training.at[i, 'Friends'] = num_friends[int(row['Id'])]
	# for i, row in testing.iterrows():
	# 	testing.at[i, 'Friends'] = num_friends[int(row['Id'])]

	# print(training)

	return connections, training, testing

def parse_posts(training, testing):

	training['Posts_st'] = np.log(training['Posts'].as_matrix().reshape(-1, 1))
	testing['Posts_st'] = np.log(testing['Posts'].as_matrix().reshape(-1, 1))

	return training, testing

def parse_class(row):
	if row['Lat'] == 0 and row['Lon'] == 0:
		return 0
	else:
		return 1

def poolJobs(jobs):
    # takes in a list of jobs and creates a pool. Hands off jobs to workers and
    # stores the returned values in "resMap" which it returns.
    
    # creates a job sharing pool with 30 workers
    # the number of workers should be slightly less than the number of cores you have 
    pool = mp.Pool(4)

    # running pool.map(mp_worker, jobs) takes the elements out of jobs list and hands them
    # off to the workers. The code for each worker is below in mp_worker
    resMap = pool.map(mp_worker, jobs)
    return resMap

def mp_worker(n):
    # defines the functionality of a worker
    # it just takes the value its passed and calls calcClosestLesserPrime   
    print("loading in job: " + str(n))
    return kernel_fn(n)

if __name__ == "__main__":

	print("starting", datetime.datetime.now())
	# data_analysis(None, None)

	training_orig = pd.read_csv("posts_train.txt")
	testing_orig = pd.read_csv("posts_test.txt")
	connections_matrix, training, testing = parse_graph(training_orig, testing_orig)
	training, testing = parse_posts(training, testing)
	print("done loading", datetime.datetime.now())
	print(training)
	print(training.max())
	# print(testing)

	all_data = pd.concat([training, testing])
	kernel_data = all_data[['Id', 'Hour1', 'Hour2', 'Hour3', 'Posts', 'Friends', 'Posts_st']]
	kern = kernel_fn(friend_kernel)

	print("done calculating", datetime.datetime.now())
	listOfJobs = [friends_kernel, total_friends, posts_kernel, hours_kernel]

	listOfKernels = poolJobs(listOfJobs)
	print(listOfKernels)
