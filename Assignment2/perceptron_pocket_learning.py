'''
Created on Mar 7, 2016

@author: vitidn
'''
import sys
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

#prepare vectorized cost function
def computeCost(label,predicted_label):
    return np.abs(label-predicted_label)

computeCostV = np.vectorize(computeCost)

def fit_perceptron_pocket(data,labels,learning_rate=1.0,max_iteration=10000):
    """
    \n return the best weights and also number of mis-classified data in each iteration
    \n format: ( np.array([w0,w1,...]) , [5,2,1,...] )
    """
    dimension = len(data[0])
    #weight vector - np.array([w1,w2,w3,w4])
    weights = np.random.uniform(low=-0.01,high=0.01,size=(dimension+1))
    iteration = 0
    miss_points = []
    #store weight that yields lowes error
    #format: (np.array([w1,w2,...]) , error)
    pocket = (None,-1)
    while iteration < max_iteration:
        iteration = iteration + 1
        
        predicted_labels = np.apply_along_axis(predict, 1,data,weights)
        errors = labels - predicted_labels
        
        miss_point = np.sum(computeCostV(labels,predicted_labels))
        miss_points.append(miss_point)
        
        current_error = miss_point / len(data)
        print("Iteration {0}, error:{1}".format(iteration,current_error))
        if(pocket[1]==-1 or current_error < pocket[1]):
            pocket = (weights.copy(),current_error)
        if(current_error == 0):
            break
        
        for j in range(len(weights)):
            sum_term = np.sum([errors[i]*data[i][j-1] if j!=0 else errors[i] for i in range(len(data))])
            weights[j] = weights[j] + learning_rate * sum_term
        
    return (pocket[0],miss_points)

def classifierFunction(x):
    if x > 0:
        return 1
    return 0

def predict(data_point,weights): 
    return classifierFunction(np.dot(np.append([1.0],data_point),weights))

def displayAccuracy(true_labels,predicted_labels):
    score = zip(true_labels,predicted_labels)
    score = [ 1 if x[0] == x[1] else 0 for x in score]
    print("Accuracy:{0}".format( (np.sum(score)+0.0)/len(score)))
    
if __name__ == "__main__":
    #filename = sys.argv[1]
    filename = "/home/vitidn/Dropbox/MS/INF552/Assignment2/linear.txt"
    
    my_data = pd.read_csv(filename,skipinitialspace=True,header=None,usecols=[0,1,2,4])
    my_data.columns = ["x","y","z","label"]
    #re-scale the data can make the algorithm converges faster
    vector_data = [np.array([row["x"],row["y"],row["z"]]) for index,row in my_data.iterrows() ]
    vector_data = np.array(vector_data)
    labels = [ row["label"] for index,row in my_data.iterrows() ]
    #change labels to 1/0
    labels = np.array([ 0 if x==-1.0 else x for x in labels])
    weights,miss_points = fit_perceptron_pocket(vector_data, labels,max_iteration=500)
    #display the number of miss-classified points for each iteration
    plt.plot(miss_points)
    plt.xlabel("# number of iteration")
    plt.ylabel("# number of miss-classified")
    print("")

    print("Weights: {0}".format(weights))
    print("")
    
    #let's predict the original data
    predicted_labels = np.apply_along_axis(predict, 1, vector_data,weights)
    displayAccuracy(labels,predicted_labels)