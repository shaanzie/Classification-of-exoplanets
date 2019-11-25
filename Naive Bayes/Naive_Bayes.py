import pandas as pd
import random
import math


# This loads the csv files
def load_csv_file(filename):

    lines = pd.read_csv('new_data_numeric.csv')
	# lines.drop([''])
    dataset = list(lines)
    for i in range(len(dataset)):

        dataset[i] = [float(x) for x in dataset] 

    return dataset


# Mimics the functions of train_test_split from sklearn

def data_split(dataset, splitRatio):

	trainSize = int(len(dataset) * splitRatio)
	trainSet = []
	copy = list(dataset)

	while len(trainSet) < trainSize:

		index = random.randrange(len(copy))
		trainSet.append(copy.pop(index))
		
	return [trainSet, copy]

# Computing seperations

def sep_class(dataset):

	separated = {}

	for i in range(len(dataset)):

		vector = dataset[i]
		if (vector[-1] not in separated):

			separated[vector[-1]] = []

		separated[vector[-1]].append(vector)

	return separated

def mean(numbers):

	return sum(numbers)/float(len(numbers))

def stdev(numbers):

	avg = mean(numbers)
	variance = sum([pow(x-avg,2) for x in numbers])/float(len(numbers)-1)

	return math.sqrt(variance)

# Compute summaries for the dataset

def summarize(dataset):
	summaries = [(mean(attribute), stdev(attribute)) for attribute in zip(*dataset)]
	del summaries[-1] 	# Causes some extra summary at the end
	return summaries


# Class summaries

def sum_class(dataset):

	separated = sep_class(dataset)
	summaries = {}

	for classValue, instances in separated.items():

		summaries[classValue] = summarize(instances)

	return summaries

def calculateProbability(x, mean, stdev):

    if(stdev != 0):

	    exponent = math.exp(-(math.pow(x-mean,2)/(2*math.pow(stdev,2))))
    else:

        exponent = 1

        stdev = 0.00000001

    return (1 / (math.sqrt(2*math.pi) * stdev)) * exponent

# Calculate per class probabilities

def calculateClassProbabilities(summaries, inputVector):

	probabilities = {}
	
	for classValue, classSummaries in summaries.items():

		probabilities[classValue] = 1
		for i in range(len(classSummaries)):

			mean, stdev = classSummaries[i]
			x = inputVector[i]
			probabilities[classValue] *= calculateProbability(x, mean, stdev)

	return probabilities
			
# Sample predict method for ground truth

def predict(summaries, inputVector):

	probabilities = calculateClassProbabilities(summaries, inputVector)

	label_best, prob_best = None, -1

	for classValue, probability in probabilities.items():

		if label_best is None or probability > prob_best:

			prob_best = probability
			label_best = classValue

	return label_best

# Perform validations here

def getPredictions(summaries, testSet):

	predictions = []
	for i in range(len(testSet)):

		result = predict(summaries, testSet[i])
		predictions.append(result)

	return predictions

# Calculate accuracy metric

def getAccuracy(testSet, predictions):

	correct = 0

	for i in range(len(testSet)):

		if testSet[i][-1] == predictions[i]:

			correct += 1

	return (correct/float(len(testSet))) * 100.0

filename = 'new_data_numeric.csv'

splitRatio = 0.80

dataset = load_csv_file(filename)

trainingSet, testSet = data_split(dataset, splitRatio)

summaries = sum_class(trainingSet)

predictions = getPredictions(summaries, testSet)

accuracy = getAccuracy(testSet, predictions)

print('Accuracy: ', accuracy)