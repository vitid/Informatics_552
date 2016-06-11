'''
Created on Mar 9, 2016

@author: vitidn
'''
import sys
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def pca(vector_data,k):
    """
    \n vector_data - np.array of data points, each element is np.array of 1 data point
    \n k - number of dimensions to transform to
    \n return - matrix, each row is a transformation of each original data point into k dimension
    """
    normalized_vector_data = generateNormalizedMatrix(vector_data)
    #calculate the covariance-matrix
    cov_matrix = generateCovarianceMatrix(np.matrix(normalized_vector_data))
    eigen_values,eigen_vectors = np.linalg.eig(cov_matrix)
    #arrange the index of eigen_values, from highest to lowest
    eigen_value_indexs = np.argsort(eigen_values)[::-1]
    #pick the first corresponding k eigen-vectors, and put them into matrix u_truncate
    u_truncate = eigen_vectors[:,eigen_value_indexs[0:k]]
    
    transformed_data = []
    for data_point in normalized_vector_data:
        z = u_truncate.transpose() * np.matrix(data_point).transpose()
        transformed_data.append(z.transpose())
    
    return np.vstack(transformed_data)

def generateNormalizedMatrix(data_matrix):
    
    normalized_matrix = data_matrix.copy()
    means = np.apply_along_axis(np.mean, 0, data_matrix)
    
    for i in range(data_matrix.shape[1]):
        normalized_matrix[:,i] = data_matrix[:,i] - means[i]
        
    return normalized_matrix

#verify with the professor is this a correct formula?
def generateCovarianceMatrix(normalized_matrix):
    """
    \n normalized_matrix - matrix of size num_observation x num_dimension
    \n format: supposed num_observation=3, num_dimension = 5
    \n np.matrix([[A1,A2,A3,A4,A5],[B1,B2,B3,B4,B5],[C1,C2,C3,C4,C5]])
    """
    num_observation = normalized_matrix.shape[0]
    covariance_matrix = normalized_matrix.transpose() * normalized_matrix
    return covariance_matrix / (num_observation - 1.0)

if __name__ == "__main__":
    #filename = sys.argv[1]
    filename = "/home/vitidn/Dropbox/MS/INF552/Assignment2/dims.txt"
    k = 2
    my_data = pd.read_csv(filename,skipinitialspace=True,header=None)
    my_data.columns = ["w","x","y","z"]
    vector_data = [np.array([row["w"],row["x"],row["y"],row["z"]]) for index,row in my_data.iterrows() ]
    vector_data = np.array(vector_data) 
    transformed_data = pca(vector_data, k)
    transformed_data = pd.DataFrame(transformed_data)
    transformed_data.columns = ["x","y"]
    plt.scatter(x=transformed_data["x"],y=transformed_data["y"])
    transformed_data.to_csv("/tmp/result_2.csv",index=False) 