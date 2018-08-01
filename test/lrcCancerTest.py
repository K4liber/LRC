#!/usr/bin/env python3.6

import sys
sys.path.insert(0, '../')

import os
from LRC import LRC 
from sklearn.decomposition import PCA
import numpy as np
import sys
from sklearn.neighbors import NearestNeighbors
import scipy.stats as stats

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

allValues = 0.0
good_predictions = 0.0

#TEST
series = 200

for i in range(series):
	np.random.shuffle(cleanData)

	learn_data = np.array(cleanData, dtype='f')[:learnRecords,1:]
	targets = np.array(cleanData, dtype='f')[:learnRecords,0]
	predict_data = np.array(cleanData, dtype='f')[learnRecords:,1:]
	true_values = np.array(cleanData, dtype='f')[learnRecords:,0]

	predictor = LRC(	
		learn_data = learn_data, 
		targets = targets
	)

	predictions = predictor.predict(predict_data)

	for j in range(len(true_values)):
		allValues += 1.0
		if true_values[j] == predictions[j]:
			good_predictions += 1.0

	sys.stdout.write("\r\x1b[K" + "Prediction progress: " 
		+ str(int((i+1)/series*100)) + "%")
	sys.stdout.flush()

print("\n")
print("LRC test error: " + str((allValues-good_predictions)/allValues))