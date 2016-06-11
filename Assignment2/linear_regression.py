import sys
import numpy as np
import pandas as pd
from numpy.linalg import inv

def predict(data_point,weights):
    """
    \n return the predicted value for the data point
    """   
    return np.dot(np.append([1.0],data_point),weights)

def displayError(target,predicted):
    score = zip(target,predicted)
    score = [ (x[0] - x[1])**2.0 for x in score]
    print("Root-Mean Square Error: {0}".format( ((np.sum(score)+0.0)/len(score))**0.5) )
    
def fit_linearRegression(data_matrix,target_vectors):
    """
    \n return weights of fitted linear model
    """
    data_matrix = np.insert(data_matrix,0,[1] * data_matrix.shape[1],axis=0)
    weights = inv(data_matrix * data_matrix.transpose()) * data_matrix * target_vectors
    return np.array(weights)

if __name__ == "__main__":
    #filename = sys.argv[1]
    filename = "/home/vitidn/Dropbox/MS/INF552/Assignment2/linear.txt"
    my_data = pd.read_csv(filename,skipinitialspace=True,header=None,usecols=[0,1,2])
    my_data.columns = ["x","y","z"]
    data_matrix = np.matrix(my_data.iloc[:,0:2])
    data_matrix = data_matrix.transpose()
    target_vectors = np.matrix(my_data.iloc[:,2:3])
    weights = fit_linearRegression(data_matrix, target_vectors)
    weights = np.array(weights.transpose()[0])
    print("Weights for the fitted model:{0}".format(weights))
    
    print("")
    
    #let's predict the original data
    vector_data = [np.array([row["x"],row["y"]]) for index,row in my_data.iterrows() ]
    vector_data = np.array(vector_data)
    target_vectors = np.array(my_data["z"])
    predicted_values = np.apply_along_axis(predict, 1,vector_data,weights)
    displayError(target_vectors, predicted_values)
        
    predict_data = pd.DataFrame(predicted_values)
    predict_data.columns = ["predicted_value"]
    predict_data.to_csv("result_6.txt",index=False)


