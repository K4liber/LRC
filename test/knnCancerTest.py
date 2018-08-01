#!/usr/bin/env python3.6

import os
import numpy as np
import sys
import operator
import scipy.stats as stats
from sklearn.neighbors import NearestNeighbors
from pandas import DataFrame as df
from knn import KNN

def getResponse(neighbors):
	classVotes = {}
	for x in range(len(neighbors)):
		response = neighbors[x]
		if response in classVotes:
			classVotes[response] += 1
		else:
			classVotes[response] = 1

	sortedVotes = sorted(classVotes.items(), key=operator.itemgetter(1), reverse=True)
	return sortedVotes[0][0]

# Params
nodes = 31
learnRecords = 450
FILENAME = os.path.dirname(__file__) + '/wdbc.dat'

# Load data from file
with open(FILENAME) as file:
    data = file.readlines()  

dataSize = len(data)
cleanData = [[0 for x in range(nodes)] for y in range(dataSize)]

for index, line in enumerate(data):
	raw = line.split(',')

	if raw[1] == 'B':
		raw[1] = 0
	else:
		raw[1] = 1

	cleanData[index] = raw[1:nodes+1]

cleanData = np.array(cleanData, dtype='f')

def normalize(data):
	min_data = min(data)
	norm = max(data) - min_data

	for i in range(len(data)):
		data[i] = (data[i] - min_data)/norm

	return data

def normalize2(data, targets):
	r = stats.pearsonr(targets, data)[0]
	min_data = min(data)
	norm = max(data) - min_data

	for i in range(len(data)):
		data[i] = r*(data[i] - min_data)/norm

	return data

for i in range(len(cleanData[0])-1):
	cleanData[:,i+1] = normalize2(cleanData[:,i+1], cleanData[:,0])

allValues = 0.0
good_predictions = 0.0

#TEST
series = 1000

for i in range(series):
	np.random.shuffle(cleanData)

	learn_data = np.array(cleanData, dtype='f')[:learnRecords,1:]
	targets = np.array(cleanData, dtype='f')[:learnRecords,0]
	predict_data = np.array(cleanData, dtype='f')[learnRecords:,1:]
	true_values = np.array(cleanData, dtype='f')[learnRecords:,0]

	predictor = KNN(
		k = 15, 
		algoritm = 'ball_tree', 
		learn_data = learn_data,
		targets = targets
	)
	predictions = predictor.predict(predict_data=predict_data)

	for j in range(len(true_values)):
		allValues += 1.0

		if predictions[j] == true_values[j]:
			good_predictions += 1.0

	sys.stdout.write("\r\x1b[K" + "Prediction progress: " 
		+ str(int((i+1)/series*100)) + "%")
	sys.stdout.flush()


error = (allValues-good_predictions)/allValues

print("\n")
print("KNN test error: " + str(error))