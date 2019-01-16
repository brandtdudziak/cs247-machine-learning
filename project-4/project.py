import numpy as np
np.set_printoptions(suppress=True)
import pandas as pd
import sklearn
import dill as pickle
import datetime
from scipy import stats

from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import cross_val_score
from sklearn import metrics
from sklearn.neural_network import MLPRegressor

# Uncomment the following 3 lines if you're getting annoyed with warnings from sklearn
import warnings
from sklearn.exceptions import DataConversionWarning
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=DeprecationWarning)
warnings.filterwarnings(action='ignore', category=DataConversionWarning)

connections_matrix = np.empty((1,1))

def both_locations(training, testing):


	#### if using modes:
	# X_tr = np.delete(training, [0,4,5,7,8,9], axis=1)
	# y_tr = training[:, 4:6]
	# X_te = np.delete(testing, [0], axis=1)

	### if using averages:
	X_tr = np.delete(training, [0,4,5], axis=1)
	y_tr = training[:, 4:6]
	X_te = np.delete(testing, [0], axis=1)

	## preprocess data!
	scaler = sklearn.preprocessing.MinMaxScaler()
	X_tr = scaler.fit(X_tr).transform(X_tr)
	X_te = scaler.transform(X_te)

	print("Fitting the regression model to the training set")
	mse_scorer = metrics.make_scorer(metrics.mean_squared_error)

	score_dict = {}
	# params = ['logistic', 'relu']
	params = [200,300,400]
	for p in params:
		reg = MLPRegressor(max_iter=p)
		scores = cross_val_score(reg, X_tr, y_tr, cv=3, n_jobs=23, scoring=mse_scorer)
		score_dict[np.amin(scores)] = reg
		print("done with param=",p)
	print("cross val scores:")
	print(score_dict)
	min_score = min(score_dict)
	clf = score_dict[min_score]
	print(min_score, clf)
	clf.fit(X_tr, y_tr)
	preds = clf.predict(X_te)

	print("saving pickle")
	filename = "pickles/raw_preds_mlp.pkl"
	outfile = open(filename, 'wb')
	pickle.dump(preds, outfile)
	outfile.close()
	print("saved pickle")
	# filename = "pickles/raw_preds_mlp.pkl"
	# outfile = open(filename, 'rb')
	# preds = pickle.load(outfile)
	# outfile.close()

	print(preds)

	return(preds)

def single_location(training, testing, direction):

	if direction == "LATITUDE":

		X_tr = np.delete(training, [0,4,5], axis=1)
		y_tr_lat = training[:,4]
		X_te = np.delete(testing, [0], axis=1)

		## preprocess data!
		
		scaler = sklearn.preprocessing.MinMaxScaler()
		X_tr = scaler.fit(X_tr).transform(X_tr)
		X_te = scaler.transform(X_te)

		print("Fitting the regression model to the training set for latitude")

		mse_scorer = metrics.make_scorer(metrics.mean_squared_error)

		score_dict = {}
		neighbors = [1, 3, 5, 7, 9, 11]
		depth = [3,5,8,10]
		for d in depth:
			# svm = KNeighborsRegressor(n_neighbors = ne, weights='distance')
			svm = DecisionTreeRegressor(max_depth=d)
			scores = cross_val_score(svm, X_tr, y_tr_lat, cv=3, n_jobs=23, scoring=mse_scorer)
			score_dict[np.amin(scores)] = svm
			print("done with neighbors=",d)
		print(score_dict)
		min_score = min(score_dict)
		clf = score_dict[min_score]
		print(min_score, clf)
		clf.fit(X_tr, y_tr_lat)
		y_preds_lat = clf.predict(X_te)

		filename = "pickles/raw_lat.pkl"
		outfile = open(filename, 'wb')
		pickle.dump(y_preds_lat, outfile)
		outfile.close()

		testing_with_predictions = np.concatenate((testing, y_preds_lat.reshape(-1,1)), axis=1)
		return(testing_with_predictions)

	else:

		X_tr = np.delete(training, [0,4,5], axis=1)
		X_tr = np.concatenate((X_tr, training[:,4].reshape(-1,1)), axis=1)
		y_tr_lon = training[:,5]
		X_te = np.delete(testing, [0], axis=1)
		
		## preprocess data!

		scaler = sklearn.preprocessing.MinMaxScaler()
		X_tr = scaler.fit(X_tr).transform(X_tr)
		X_te = scaler.transform(X_te)

		print("Fitting the regression model to the training set for longitude")

		mse_scorer = metrics.make_scorer(metrics.mean_squared_error)

		score_dict = {}
		neighbors = [1, 3, 5, 7, 9, 11]
		depth = [3,5,8,9,10]
		for d in depth:
			# svm = KNeighborsRegressor(n_neighbors = ne, weights='distance')
			svm = DecisionTreeRegressor(max_depth=d)
			scores = cross_val_score(svm, X_tr, y_tr_lon, cv=3, n_jobs=23, scoring=mse_scorer)
			score_dict[np.amin(scores)] = svm
			print("done with depth=",d)
		print(score_dict)
		min_score = min(score_dict)
		clf = score_dict[min_score]
		print(min_score, clf)
		clf.fit(X_tr, y_tr_lon)
		y_preds_lon = clf.predict(X_te)

		filename = "pickles/raw_lon.pkl"
		outfile = open(filename, 'wb')
		pickle.dump(y_preds_lon, outfile)
		outfile.close()

		testing_with_predictions = np.concatenate((testing, y_preds_lon.reshape(-1,1)), axis=1)
		return(testing_with_predictions)

def parse_graph_modes(training, testing):

	average_latitude = 35.81251739505751
	average_longitude = -43.7429278501596
	defaulted = 0

	lines = np.loadtxt("graph.txt", comments="#", delimiter="\t", unpack=False)

	m = int(np.amax(lines))
	connections = np.zeros((m+1, m+1))

	maps = {}
	for line in lines:
		connections[int(line[0])][int(line[1])] = 1
		if line[0] in maps:
			maps[line[0]].append(line[1])
		else:
			maps[line[0]] = [line[1]]

	print("compiling friends info")

	te_friends_info = np.zeros((len(testing),3))
	for i in range(testing.shape[0]):
		if i % 5000 == 0:
			print("update:", i, datetime.datetime.now())
		if testing[i,0] in maps:
			rel_list = maps[testing[i,0]]
			rel_users = np.array([row.tolist() for row in training if row[0] in rel_list])
			
			if len(rel_list) == 0 or len(rel_users) == 0:
				lat = average_latitude
				lon = average_longitude
				defaulted += 1
			elif len(rel_list) == 1 or len(rel_users) == 1:
				lat = rel_users[0][7]
				lon = rel_users[0][8]
			else:
				lat = stats.mode(rel_users[:,7])[0][0]
				lon = stats.mode(rel_users[:,8])[0][0]

			te_friends_info[i,0] = len(rel_list)
			te_friends_info[i,1] = lat
			te_friends_info[i,2] = lon
		else:
			te_friends_info[i,0] = 0
			te_friends_info[i,1] = 0
			te_friends_info[i,2] = 0
	print("done with testing friends info")
	print(defaulted, "defaulted")
	testing = np.concatenate((testing, te_friends_info), axis=1)
	print(testing.shape)
	print(testing[0])
	# print("saving testing")
	# filename = "pickles/processed_testing_data_1.pkl"
	# outfile = open(filename, 'wb')
	# pickle.dump(testing, outfile)
	# outfile.close()
	# print("saved testing pickle")

	tr_friends_info = np.zeros((len(training),3))
	for i in range(training.shape[0]):
		if i % 5000 == 0:
			print("update:", i, datetime.datetime.now())
		if training[i,0] in maps:
			rel_list = maps[training[i,0]]
			rel_users = np.array([row.tolist() for row in training if row[0] in rel_list])
			
			if len(rel_list) == 0 or len(rel_users) == 0:
				lat = training[i,7]
				lon = training[i,8]
			elif len(rel_list) == 1 or len(rel_users) == 1:
				lat = rel_users[0][7]
				lon = rel_users[0][8]
			else:
				lat = stats.mode(rel_users[:,7])[0][0]
				lon = stats.mode(rel_users[:,8])[0][0]

			tr_friends_info[i,0] = len(rel_list)
			tr_friends_info[i,1] = lat
			tr_friends_info[i,2] = lon
		else:
			tr_friends_info[i,0] = 0
			tr_friends_info[i,1] = 0
			tr_friends_info[i,2] = 0
	print("done with training friends info")
	training = np.concatenate((training, tr_friends_info), axis=1)
	print(training.shape)
	print(training[0])

	# print("saving training")
	# filename = "pickles/processed_training_data_1.pkl"
	# outfile = open(filename, 'wb')
	# pickle.dump(training, outfile)
	# outfile.close()
	# print("saved training pickle")
	
	return connections, training, testing

def parse_graph_averages(training, testing):

	average_latitude = 35.81251739505751
	average_longitude = -43.7429278501596
	defaulted = 0

	lines = np.loadtxt("graph.txt", comments="#", delimiter="\t", unpack=False)

	m = int(np.amax(lines))
	connections = np.zeros((m+1, m+1))

	maps = {}
	for line in lines:
		connections[int(line[0])][int(line[1])] = 1
		if line[0] in maps:
			maps[line[0]].append(line[1])
		else:
			maps[line[0]] = [line[1]]

	# print("compiling friends info")

	te_friends_info = np.zeros((len(testing),3))
	for i in range(testing.shape[0]):
		if i == 10:
			break
		if i % 5000 == 0:
			print("update:", i, datetime.datetime.now())
		if testing[i,0] in maps:
			rel_list = maps[testing[i,0]]
			rel_users = np.array([row.tolist() for row in training if row[0] in rel_list])
			
			# print("------------")
			# print(rel_list)
			# print(rel_users)

			if len(rel_list) == 0 or len(rel_users) == 0:
				lat = average_latitude
				lon = average_longitude
				defaulted += 1
			elif len(rel_list) == 1 or len(rel_users) == 1:
				# print("one")
				lat = rel_users[0][4]
				lon = rel_users[0][5]
			else:
				# print("multi")
				lat = np.average(rel_users[:,4])
				lon = np.average(rel_users[:,5])

			# print(lat, lon)
			te_friends_info[i,0] = len(rel_list)
			te_friends_info[i,1] = lat
			te_friends_info[i,2] = lon
		else:
			te_friends_info[i,0] = 0
			te_friends_info[i,1] = 0
			te_friends_info[i,2] = 0
	print("done with testing friends info")
	print(defaulted, "defaulted")
	testing = np.concatenate((testing, te_friends_info), axis=1)
	print(testing.shape)
	# print("saving testing")
	# filename = "pickles/processed_testing_data_1.pkl"
	# outfile = open(filename, 'wb')
	# pickle.dump(testing, outfile)
	# outfile.close()
	# print("saved testing pickle")

	tr_friends_info = np.zeros((len(training),3))
	for i in range(training.shape[0]):
		if i == 10:
			break
		if i % 5000 == 0:
			print("update:", i, datetime.datetime.now())
		if training[i,0] in maps:
			rel_list = maps[training[i,0]]
			rel_users = np.array([row.tolist() for row in training if row[0] in rel_list])
			
			if len(rel_list) == 0 or len(rel_users) == 0:
				lat = training[i,4]
				lon = training[i,5]
			elif len(rel_list) == 1 or len(rel_users) == 1:
				lat = rel_users[0][4]
				lon = rel_users[0][5]
			else:
				lat = np.average(rel_users[:,4])
				lon = np.average(rel_users[:,5])

			tr_friends_info[i,0] = len(rel_list)
			tr_friends_info[i,1] = lat
			tr_friends_info[i,2] = lon
		else:
			tr_friends_info[i,0] = 0
			tr_friends_info[i,1] = 0
			tr_friends_info[i,2] = 0
	print("done with training friends info")
	training = np.concatenate((training, tr_friends_info), axis=1)
	print(training.shape)

	# print("saving training")
	# filename = "pickles/processed_training_data_1.pkl"
	# outfile = open(filename, 'wb')
	# pickle.dump(training, outfile)
	# outfile.close()
	# print("saved training pickle")

	print(training)
	print(training[0])
	
	return connections, training, testing

def parse_and_save_preds(predictions):

	### to save predictions to pkl file:
	print("saving pickle of predictions")
	filename = "pickles/preds_ave.pkl"
	outfile = open(filename, 'wb')
	pickle.dump(predictions, outfile)
	outfile.close()
	print("done saving pickle of predictions")

	### to open predictions from saved pkl file:
	# print("opening pickle of predictions")
	# filename = "pickles/preds_mlp.pkl"
	# outfile = open(filename, 'rb')
	# predictions = pickle.load(outfile)
	# outfile.close()
	# print("done opening pickle of predictions")

	predictions = predictions[:, [0,-2,-1]]

	### format and save predictions:
	print("saving predictions")
	np.savetxt("preds/preds_ave.txt", predictions,
		delimiter=',', newline='\n', header='Id,Lat,Lon', fmt=['%i', '%f', '%f'])
	print("done saving predictions")

def discretize_locs(training):

	lats = training[:,4]
	lons = training[:,5]

	dlats = np.array([val - val%5 for val in lats])
	dlons = np.array([val - val%10 for val in lons])
	dlons2 = np.array([val-val%25 for val in lons])
	training = np.concatenate((training, dlats.reshape(-1,1), dlons.reshape(-1,1), dlons2.reshape(-1,1)), axis=1)
	return training

def parse_hours(training, testing):
	hours_map = {0: 0, 1: 0, 2: 0, 3: 0, 4: 1, 5: 1, 6: 1, 7: 1, 8: 2, 9: 2, 10: 2, 11: 2, 
		12: 3, 13: 3, 14: 3, 15: 3, 16: 2, 17: 2, 18: 2, 19: 2, 20: 1, 21: 1, 22: 1, 23: 1, 25: 0}

	# print("TRAINING BEFORE")
	# print(training[0])

	tr_hr1 = np.array(training[:,1]).reshape(-1,1)
	tr_hr2 = np.array([tr_hr1[i,0] if training[i,2] == 25 else training[i,2] for i in range(len(training[:,2]))]).reshape(-1,1)
	tr_hr3 = np.array([tr_hr1[i,0] if training[i,3] == 25 else training[i,3] for i in range(len(training[:,3]))]).reshape(-1,1)
	te_hr1 = np.array(testing[:,1]).reshape(-1,1)
	te_hr2 = np.array([te_hr1[i,0] if testing[i,2] == 25 else testing[i,2] for i in range(len(testing[:,2]))]).reshape(-1,1)
	te_hr3 = np.array([te_hr1[i,0] if testing[i,3] == 25 else testing[i,3] for i in range(len(testing[:,3]))]).reshape(-1,1)

	tr_hours = np.concatenate((tr_hr1, tr_hr2, tr_hr3), axis=1)
	te_hours = np.concatenate((te_hr1, te_hr2, te_hr3), axis=1)
	training[:,1:4] = tr_hours
	testing[:,1:4] = te_hours

	tr_hours_disc = np.array([hours_map[val] for val in training[:,1]])
	te_hours_disc = np.array([hours_map[val] for val in testing[:,1]])

	training = np.concatenate((training, tr_hours_disc.reshape(-1,1)), axis=1)
	testing = np.concatenate((testing, te_hours_disc.reshape(-1,1)), axis=1)

	# print("TRAINING AFTER")
	# print(training[0])
	return training, testing

if __name__ == "__main__":

	print("starting", datetime.datetime.now())

	### generate and save data

	# print("parsing and generating data")
	# training_orig = np.genfromtxt("posts_train.txt", delimiter=',', skip_header=1)
	# testing_orig = np.genfromtxt("posts_test.txt", delimiter=',', skip_header=1)

	# training = discretize_locs(training_orig)
	# connections_matrix, training, testing = parse_graph_averages(training, testing_orig)
	# training, testing = parse_hours(training, testing)
	# print(training)
	# print(testing)

	# print("saving")
	# filename = "pickles/averages_training.pkl"
	# outfile = open(filename, 'wb')
	# pickle.dump(training, outfile)
	# outfile.close()

	# filename = "pickles/averages_testing.pkl"
	# outfile = open(filename, 'wb')
	# pickle.dump(testing, outfile)
	# outfile.close()
	# print("saved pickle")

	###### done generating and saving data


	### open saved data

	print("opening")
	filename = "pickles/averages_training.pkl"
	outfile = open(filename, 'rb')
	training = pickle.load(outfile)
	outfile.close()

	filename = "pickles/averages_testing.pkl"
	outfile = open(filename, 'rb')
	testing = pickle.load(outfile)
	outfile.close()
	print("opened pickles")

	### done opening saved data


	### time to learn:

	print(training[0])
	print(testing[0])

	### remove null points from the training set!
	training = training[(training[:,4]) != 0]

	### to predict latitude first and then then use it to predict longitude:
	# print("running latitude")
	# testing = single_location(training, testing, "LATITUDE")
	# print(testing)
	# print("testing is now:")
	# print(testing)
	# print("running longitude")
	# testing = single_location(training, testing, "LONGITUDE")
	# print("testing is now:")
	# print(testing)

	### to predict latitude and longitude together using MLP Regressor:
	preds = both_locations(training, testing)
	testing = np.concatenate((testing, preds), axis=1)
	print(testing)

	### to save predictions:
	parse_and_save_preds(testing)





