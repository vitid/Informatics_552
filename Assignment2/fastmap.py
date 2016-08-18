import sys
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def fastmap(data_matrix,k,current_dimension=0,mapped_matrix=None,pivot_matrix=None):
    """
    \n data_matrix - n x d, n:number of data and d:original dimension
    """
    if(current_dimension == k):
        return (mapped_matrix,pivot_matrix)
    
    #n - number of data
    n = data_matrix.shape[0]
    
    #the first time method is called
    if(mapped_matrix==None):
        mapped_matrix = np.zeros((n,k))
        pivot_matrix = np.zeros((2,k))
        #diagonal matrix to store distances
        #contains distance of object i and object j where i > j
    
    #pick objectA
    index_a = np.random.choice(range(n))
    object_a = data_matrix[index_a,:]
    #store (index_point,distance_point) that has the largest distance from object_a
    dis_bucket = [-1,-1]
    #in this loop, we will calculate distance from any points to object_a
    for i in range(data_matrix.shape[0]):
        if(i==index_a):
            distance = 0
        distance = getDistance(index_a,i,data_matrix,mapped_matrix,current_dimension-1)
        #store the furthest distance
        if(distance > dis_bucket[1]):
            dis_bucket[0] = i
            dis_bucket[1] = distance
    #get the object_b, which is furthest apart from the object a
    index_b,distance_ab = dis_bucket[0],dis_bucket[1]
    object_b = data_matrix[index_b,:]
    
    #record the ids of object_a and object_b
    pivot_matrix[:,current_dimension] = [index_a,index_b]
    
    if(distance_ab == 0):
        #set all distance in this dimension = 0
        mapped_matrix[:,current_dimension] = [0] * n
    else:
        for i in range(data_matrix.shape[0]):
            distance_ai = getDistance(index_a, i, data_matrix, mapped_matrix, current_dimension-1)
            distance_bi = getDistance(index_b, i, data_matrix, mapped_matrix, current_dimension-1)
            new_pos = computeMappedCoordinate(distance_ab, distance_ai, distance_bi)
            mapped_matrix[i,current_dimension] = new_pos
                
    return fastmap(data_matrix, k, current_dimension+1, mapped_matrix, pivot_matrix)
        
def computeMappedCoordinate(distance_ab,distance_ai,distance_bi):
    return ((distance_ai ** 2.0) +  (distance_ab ** 2.0) - (distance_bi ** 2.0)) / (2*distance_ab)
    
def getDistance(index_i,index_j,data_matrix,mapped_matrix,current_dimension):
    if current_dimension == -1:
        x = data_matrix[index_i,:]
        y = data_matrix[index_j,:]
        d = np.array(x - y) ** 2
        return np.sum(d) ** 0.5
    current_x_pos = mapped_matrix[index_i,current_dimension]
    current_y_pos = mapped_matrix[index_j,current_dimension]
    return ((getDistance(index_i,index_j,data_matrix,mapped_matrix, current_dimension-1) ** 2) - ((current_x_pos - current_y_pos) ** 2)) ** 0.5
    
#data_matrix = np.matrix([[11,202,3],[37,4,501],[5,66,7000],[7,8,99]])
#mapped_matrix,pivot_matrix = fastmap(data_matrix, 2)
#print(mapped_matrix)
filename = "/home/vitidn/Dropbox/MS/INF552/Assignment2/dims_final_project.txt"
k = 3
my_data = pd.read_csv(filename,skipinitialspace=True,header=None)
my_data.columns = ["x","y"]
data_matrix = np.matrix(my_data)
mapped_matrix,pivot_matrix = fastmap(data_matrix, k)
print(mapped_matrix)
#plt.scatter(x=mapped_matrix[:,0],y=mapped_matrix[:,1])    
