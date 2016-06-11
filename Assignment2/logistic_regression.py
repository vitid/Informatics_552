# -*- coding: utf-8 -*-
"""
Created on Sun Mar  6 18:36:28 2016

@author: vitidn
"""
import sys
import numpy as np
import pandas as pd

#prepare vectorized cost function
def computeCost(label,prob):
    return -((label * np.log(prob)) + (1.0-label)*np.log(1.0-prob))

computeCostV = np.vectorize(computeCost)

def fit_logistic(data,labels,learning_rate=0.1,max_iteration=1000,target_error=0.1):
    """
    \n data - a list of data vector
    \n format: np.array([np.array([x1,y1,z1]),np.array([x2,y2,z2]),...])
    \n labels - a list of data labels, consists only 0/1
    \n format: np.array([0,1,0,0,1,...])
    \n return: weights - the optimized weights
    \n format: np.array([w0,w1,w2,...])
    """
    dimension = len(data[0])
    #weight vector - np.array([w1,w2,w3,w4])
    weights = np.random.uniform(low=-0.01,high=0.01,size=(dimension+1))
    iteration = 0
    
    while iteration < max_iteration:
        iteration = iteration + 1
        
        predicted_prob = np.apply_along_axis(predict, 1,data,weights)
        errors = predicted_prob - labels
        
        current_error = np.sum(computeCostV(labels,predicted_prob)) / len(data)
        print("Iteration {0}, error:{1}".format(iteration,current_error))
        #stop the algorithm if target error rate is reached
        if(current_error < target_error):
            break
        
        for j in range(len(weights)):
            sum_term = np.sum([errors[i]*data[i][j-1] if j!=0 else errors[i] for i in range(len(data))])
            weights[j] = weights[j] - learning_rate * sum_term
        
    return weights

def sigmoidFunction(x):
    return 1.0 / (1.0 + np.e ** -x)
    
def predict(data_point,weights):
    """
    \n return the probability of belonging to class 1 for the data points
    \n return : predicted_prob    
    """   
    return sigmoidFunction(np.dot(np.append([1.0],data_point),weights))
    
def displayAccuracy(true_labels,predicted_labels):
    score = zip(true_labels,predicted_labels)
    score = [ 1 if x[0] == x[1] else 0 for x in score]
    print("Accuracy:{0}".format( (np.sum(score)+0.0)/len(score)))
    
def rescaleData(data,column_names):    
    """
    \n re-scale the data in each column_names to [0,1] range
    """
    for column_name in column_names:
        min_value = np.min(data[column_name])
        max_value = np.max(data[column_name])
        data[column_name] = (data[column_name] - min_value) / (max_value - min_value)
        
if __name__ == "__main__":
    filename = sys.argv[1]
    
    my_data = pd.read_csv(filename,skipinitialspace=True,header=None)
    my_data.columns = ["x","y","z","unused","label"]
    #re-scale the data can make the algorithm converges faster
    rescaleData(my_data,["x","y","z"])
    vector_data = [np.array([row["x"],row["y"],row["z"]]) for index,row in my_data.iterrows() ]
    vector_data = np.array(vector_data)
    labels = [ row["label"] for index,row in my_data.iterrows() ]
    #change labels to 1/0
    labels = np.array([ 0 if x==-1.0 else x for x in labels])
    weights = fit_logistic(vector_data,labels)
    
    #let's predict the original data
    predicted_prob = np.apply_along_axis(predict, 1, vector_data,weights)
    #coerce the predicted probability into label, use threshold probability = 0.5
    predicted_labels = [1 if x >= 0.5 else 0 for x in predicted_prob]
    displayAccuracy(labels,predicted_labels)
    
    #output the result
    prob_data = pd.DataFrame(predicted_prob)
    prob_data.columns = ["prob label = 1"]
    prob_data.to_csv("/tmp/result_7.csv",index=False)  
