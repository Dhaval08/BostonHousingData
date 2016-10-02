import numpy as np
import matplotlib.pyplot as plt
import pandas
import math
import sys
from sklearn import linear_model
import random
def calculateMSE(parameters, data, target):

    total = 0

    for i in range (0, len(target)):
        total = total + math.pow((np.dot(parameters[:,0], data[i,:])) - target[i], 2)

    return total/len(target)

def closedFormRidge(data, target, lamda):
    target_array = np.asarray(target)
    first_term = np.dot(np.transpose(data),data) + np.dot(lamda, np.identity(14))
    parameters[:,0] = np.dot(np.linalg.pinv(first_term), np.dot(np.transpose(data), target_array))

    return parameters

def getResidueVector(data, target, parameters):

    residue_vector = np.empty(shape=(data.shape[0],1))

    for i in range (0, data.shape[0]):
        residue_vector[i,0] = target[i] - np.dot(parameters[:,0], data[i,:])
    return residue_vector

def getPearsonValue(data, residue_vector, considered_features):
    residue_data2 = np.asarray(residue_vector)
    mean2 = residue_vector[:,0].mean()
    std2 = residue_data2.std()

    max_pearson = -sys.maxsize

    for i in range (1,data.shape[1]):
        if i in considered_features:
            continue
        else:
            residue_data1 = data[:,i]
            mean1 = residue_data1.mean()
            std1 = residue_data1.std()
            pearson = abs(((residue_data1-mean1)*(residue_data2[:,0]-mean2)).mean()/(std1*std2))

            if(pearson > max_pearson):
                max_pearson = pearson
                answer = i

    return answer

def splitFeatures(data, featurelist):

    split_data = np.empty(shape=(data.shape[0], 0))

    for i in range(1,14):
        if i in featurelist:

            temp_column = np.empty(shape=(data.shape[0], 1))

            temp_column[:,0] = data[:,i]

            split_data = np.concatenate((split_data, temp_column), axis= 1)

    return split_data

from sklearn.datasets import load_boston
boston = load_boston()

train_data = np.empty(shape=(433,13))
train_target = []

test_data = np.empty(shape=(73,13))
test_target = []

standardized_train_data = np.empty(shape=(433,13))
standardized_test_data = np.empty(shape=(73,13))

index = 0
train_count = 0
test_count = 0

# Splitting the data into training and testing data. Every 7ith (i=0,1,2,...) belongs to test data

for each_value in boston.data:
    if index%7 == 0:
        test_data[test_count] = boston.data[index].tolist()
        test_target.append(boston.target[index])
        test_count = test_count + 1
    else:
        train_target.append(boston.target[index])
        train_data[train_count] = boston.data[index].tolist()
        train_count = train_count + 1
    index = index+1

# Plotting the histogram for each attribute in the training data

'''
for i in range (0,12):
    df = pandas.DataFrame(train_data[:,i])
    #plt.figure()
    df.plot.hist(bins=10)
    plt.show()
'''
# Now we calculate pearson correlation of each attribute with respect to the target value

data2 = np.asarray(train_target)
mean2 = data2.mean()
std2 = data2.std()

# Inserting a column of 1's at the beginning
standardized_train_data = np.insert(standardized_train_data, 0, 1, axis=1)
standardized_test_data = np.insert(standardized_test_data, 0, 1, axis=1)
pearson_correlation = []

for i in range (0,13):
    data1 = train_data[:,i]
    mean1 = data1.mean()
    std1 = data1.std()

    pearson = ((data1-mean1)*(data2-mean2)).mean()/(std1*std2)
    pearson_correlation.append(pearson)
    standardized_train_data[:,i+1] = (train_data[:,i] - mean1)/std1
    standardized_test_data[:,i+1] = (test_data[:,i] - mean1)/std1

print(pearson_correlation)

# one for bias and rest for each attribute = 14

parameters = np.zeros(shape=(14,1))

# Calculating the cost function
costValue = []
iterations = []
check = 0
linear_cost = 0
linear_parameters = parameters

print "---------------Linear Regression----------------"

linear_parameters = np.zeros(shape=(standardized_train_data.shape[1], 1))
first_term = np.dot(np.transpose(standardized_train_data), standardized_train_data)
linear_parameters[:,0] = np.dot(np.linalg.pinv(first_term), np.dot(np.transpose(standardized_train_data), train_target))

#print(test_target)
train_linear_MSE = calculateMSE(linear_parameters, standardized_train_data, train_target)
test_linear_MSE = calculateMSE(linear_parameters, standardized_test_data, test_target)

print "Train MSE for linear regression: ", train_linear_MSE
print "Test MSE for linear regression: ", test_linear_MSE

# Ridge regrssion

print "\n---------------Ridge Regression----------------"
for lamda in (0.01, 0.1, 1.0):
    ridge_parameters = closedFormRidge(standardized_train_data, train_target, lamda)

    print "Train MSE for Ridge Regression for lamda =",lamda,": ",calculateMSE(ridge_parameters, standardized_train_data, train_target)
    print "Test MSE for Ridge Regression for lamda =",lamda,": ",calculateMSE(ridge_parameters, standardized_test_data, test_target)

# Split the train data into 10 sets for 10 fold cross validation

target_array = np.empty(shape=(1,433))
target_array[0,:] = train_target

split_data = np.array_split(standardized_train_data, 10)
split_target = np.array_split(np.asarray(target_array), 10, axis=1)

lamda = 0.0001
optimal_mse = float("inf")

print "\n--------------Cross Validation----------------"

while lamda<=10:
    cross_validation_MSE = 0
    cross_test_MSE = 0
    min_cost = "inf"

    for i in range(0,10):

        train_cross_data = np.empty(shape = (0,14))
        train_cross_target = np.empty(shape = (1,0))

        validation_cross_data = split_data[i]
        validation_cross_target = split_target[i]

        for j in range(0,10):
            if j != i:
                train_cross_data = np.concatenate((train_cross_data, split_data[j]), axis = 0)
                train_cross_target = np.concatenate((train_cross_target, split_target[j]), axis = 1)

        cross_ridge_parameters = closedFormRidge(train_cross_data, train_cross_target.tolist()[0], lamda)

        test = calculateMSE(cross_ridge_parameters, validation_cross_data, validation_cross_target.tolist()[0])

        cross_validation_MSE = cross_validation_MSE + test

    print "CV Result for lambda =", lamda, ": ", cross_validation_MSE/10

    if (cross_validation_MSE/10 < optimal_mse):
        optimal_mse = cross_validation_MSE/10
        optimal_lamda = lamda
    lamda = lamda + 1.3512

print "The best choice for lambda is ", optimal_lamda

# Now, taking the best lamda, we calculate MSE for test data

cross_validation_parameters = closedFormRidge(standardized_train_data, train_target, optimal_lamda)
print 'MSE for test data using lamda = ',optimal_lamda,':',calculateMSE(cross_ridge_parameters, standardized_test_data, test_target)

absolute_pearson = []

for i in range(0, len(pearson_correlation)):
    absolute_pearson.append(abs(pearson_correlation[i]))

indices = sorted(range(len(absolute_pearson)), key=lambda k:absolute_pearson[k], reverse=True)

print "\n-------------Selection of 4 features based on pearson correlation--------------"

print "The 4 features selected based on pearson correlation (Using zero-indexing) are: ", indices[0:4]

feature_selection_data = np.empty(shape=(433,0))
feature_selection_test = np.empty(shape=(73,0))

for i in range(0,4):
    new_column = np.empty(shape=(433,1))
    new_column[:,0] = standardized_train_data[:,indices[i]+1]

    new_test_column = np.empty(shape=(73,1))
    new_test_column[:,0] = standardized_test_data[:,indices[i]+1]

    feature_selection_test = np.concatenate((feature_selection_test, new_test_column), axis = 1)
    feature_selection_data = np.concatenate((feature_selection_data, new_column), axis = 1)

feature_selection_data = np.insert(feature_selection_data, 0, 1, axis=1)
feature_selection_test = np.insert(feature_selection_test, 0, 1, axis=1)

linear_parameters = np.zeros(shape=(feature_selection_data.shape[1], 1))
first_term = np.dot(np.transpose(feature_selection_data), feature_selection_data)
linear_parameters[:,0] = np.dot(np.linalg.pinv(first_term), np.dot(np.transpose(feature_selection_data), train_target))

print "Train MSE for selected 4 features:", calculateMSE(linear_parameters, feature_selection_data, train_target)
print "Test MSE for selected 4 features:", calculateMSE(linear_parameters,feature_selection_test, test_target)

print "\n------------Selection of 4 features based on residue-------------------"

residue_train_data = np.empty(shape=(433,0))
feature_selection_test = np.empty(shape=(73,0))

new_column = np.empty(shape=(433,1))
new_column[:,0] = standardized_train_data[:,indices[0]+1]

residue_train_data = np.concatenate((residue_train_data, new_column), axis=1)

new_test_column = np.empty(shape=(73,1))
new_test_column[:,0] = standardized_test_data[:,indices[0]+1]

feature_selection_test = np.concatenate((feature_selection_test, new_test_column), axis=1)

residue_train_data = np.insert(residue_train_data, 0, 1, axis=1)

considered_features = []

considered_features.append(indices[0]+1)

index = 1

while(index<4):

    residue_parameters = np.zeros(shape=(index+1,1))

    first_term = np.dot(np.transpose(residue_train_data), residue_train_data)

    residue_parameters[:,0] = np.dot(np.linalg.pinv(first_term), np.dot(np.transpose(residue_train_data), train_target))

    residue_vector = getResidueVector(residue_train_data, train_target, residue_parameters)

    next_feature = getPearsonValue(standardized_train_data, residue_vector, considered_features)

    considered_features.append(next_feature)

    new_column = np.empty(shape=(433,1))
    new_column[:,0] = standardized_train_data[:,next_feature]

    new_test_column = np.empty(shape=(73,1))
    new_test_column[:,0] = standardized_test_data[:,next_feature]

    feature_selection_test = np.concatenate((feature_selection_test, new_test_column), axis = 1)

    residue_train_data = np.concatenate((residue_train_data, new_column), axis=1)

    index = index+1

linear_parameters = np.zeros(shape=(residue_train_data.shape[1], 1))
first_term = np.dot(np.transpose(residue_train_data), residue_train_data)
linear_parameters[:,0] = np.dot(np.linalg.pinv(first_term), np.dot(np.transpose(residue_train_data), train_target))

feature_selection_test = np.insert(feature_selection_test, 0, 1, axis=1)

considered_features = [x-1 for x in considered_features]

print "The 4 features selected based on residue (using zero indexing) are:", considered_features

print "Train MSE for selected 4 features: ", calculateMSE(linear_parameters, residue_train_data, train_target)

print "Test MSE for selected 4 features: ", calculateMSE(linear_parameters, feature_selection_test, test_target)

print "\n-----------------Brute Force selection of features----------------"

final_brute_parameters = np.zeros(shape=(5,1))

featurelist = []
min_indices = []
min_mse = float("inf")
count = 0
for i in range(1, 11):
    for j in range (i+1, 12):
        for k in range(j+1, 13):
            for l in range(k+1,14):
                featurelist = [i,j,k,l]
                count = count + 1
                brute_cost = 0

                brute_data = splitFeatures(standardized_train_data, featurelist)
                brute_data = np.insert(brute_data, 0, 1, axis=1)
                brute_parameters = np.zeros(shape=(brute_data.shape[1],1))


                first_term = np.dot(np.transpose(brute_data), brute_data)

                brute_parameters[:,0] = np.dot(np.linalg.pinv(first_term), np.dot(np.transpose(brute_data), train_target))

                temp = (calculateMSE(brute_parameters, brute_data, train_target))

                if(temp<min_mse):
                    min_mse = temp
                    min_indices = featurelist
                    final_brute_parameters = brute_parameters

brute_test_data = splitFeatures(standardized_test_data, min_indices)
brute_test_data = np.insert(brute_test_data, 0 , 1, axis=1)

min_indices = [x-1 for x in min_indices]

print "The 4 features selected using brute-force (using zero-indexing) are: ",min_indices

print "Train MSE for selected 4 features: ", min_mse
print "Test MSE for selected 4 features: ", calculateMSE(final_brute_parameters, brute_test_data, test_target)


print "\n-----------------Polynomial Feature Expansion-----------------"

expanded_train_data = np.empty(shape=(433,0))
expanded_test_data = np.empty(shape=(73,0))

expanded_train_data = np.concatenate((expanded_train_data, standardized_train_data[:,1:14]), axis=1)

expanded_test_data = np.concatenate((expanded_test_data, standardized_test_data[:,1:14]), axis=1)

for i in range(0,13):
    for j in range(i,13):

        new_train = np.empty(shape=(433,1))
        new_test = np.empty(shape=(73,1))

        new_train[:,0] = (standardized_train_data[:,i+1]*standardized_train_data[:,j+1])
        new_test[:,0] = (standardized_test_data[:,i+1]*standardized_test_data[:,j+1])

        expanded_train_data = np.concatenate((expanded_train_data, new_train), axis=1)
        expanded_test_data = np.concatenate((expanded_test_data, new_test), axis=1)

expanded_train_data = np.insert(expanded_train_data, 0, 1, axis=1)
expanded_test_data = np.insert(expanded_test_data, 0, 1, axis=1)

for i in range (14,104):
    data1 = expanded_train_data[:,i]
    mean1 = data1.mean()
    std1 = data1.std()

    expanded_train_data[:,i] = (expanded_train_data[:,i] - mean1)/std1
    expanded_test_data[:,i] = (expanded_test_data[:,i] - mean1)/std1

expanded_parameters = np.zeros(shape=(expanded_train_data.shape[1], 1))

first_term = np.dot(np.transpose(expanded_train_data), expanded_train_data)

expanded_parameters[:,0] = np.dot(np.linalg.pinv(first_term), np.dot(np.transpose(expanded_train_data), train_target))

print "Train MSE after polynomial feature expansion:",calculateMSE(expanded_parameters, expanded_train_data, train_target)

print "Test MSE after polynomial feature expansion:",calculateMSE(expanded_parameters, expanded_test_data, test_target)