# -*- coding: utf-8 -*-
"""
Created on Wed Feb  3 20:12:07 2016

@author: vitidn
"""
import sys
import matplotlib.pyplot as plot
import matplotlib.pylab as pylab
import pandas as pd
import numpy as np

def calculateDistance(vector1,vector2,method="L2"):
    if(method=="L1"):
        return abs(vector1[0] - vector2[0]) + abs(vector1[1] - vector2[1])
    elif(method=="L2"):
        return np.linalg.norm(vector1 - vector2)

def kmeans(data,k=3,method="L2"):
    
    #pre-calculate vectors for all points
    vector_data = [np.array([row["x"],row["y"]]) for index,row in data.iterrows() ]
    
    #pick initial centroids
    #Firstly, pick randomly from the possible points
    #For subsequent picks, add the point whose minimum distance from the selected points is maximum
    row_numbers = range(0,len(my_data))
    min_distances = np.array([None] * len(my_data))
    sampling_rows = np.random.choice(row_numbers,size=1,replace=False) 
    row_numbers.remove(sampling_rows[0])
    for i in range(1,k):
        latest_centroid_row = sampling_rows[len(sampling_rows)-1]
        latest_centroid_vector = vector_data[latest_centroid_row]
        for row in row_numbers:
            vector = vector_data[row]
            distance = calculateDistance(vector,latest_centroid_vector,method = method)
            if(min_distances[row] == None or distance < min_distances[row]):
                min_distances[row] = distance
                
        selected_row = np.where(min_distances == min_distances.max())[0]
        min_distances[selected_row] = None
        row_numbers.remove(selected_row)
        sampling_rows = np.append(sampling_rows,selected_row)
                        
    centroids = [] 
    for i in range(0,len(sampling_rows)):
        x,y,label = data.iloc[ sampling_rows[i] ]["x"],data.iloc[ sampling_rows[i] ]["y"],(i+0.0)/(k-1)
        centroid_vector = np.array([x,y])
        centroids.append((centroid_vector,label))
        
    iteration_count = 0
    while(True):   
        iteration_count += 1
        print("Iteration {0} has centroids: {1}".format(iteration_count,centroids))
        
        #assign a new centroid for each point
        for index,row in data.iterrows():
            vector = vector_data[index]
            min_distance = -1
            for centroid in centroids:
                centroid_vector = centroid[0]
                label = centroid[1]
                distance = calculateDistance(vector,centroid_vector,method=method)
                    
                if(min_distance == -1 or distance < min_distance):
                    data.loc[index,"label"] = label
                    min_distance = distance
                    
        #plot the result for each iteration
        plot.scatter(data["x"],data["y"],c=plot.cm.rainbow(data["label"]))
        #also, plot each centroid as "+" symbol
        for centroid in centroids:
            centroid_vector,l = centroid
            plot.scatter(centroid_vector[0],centroid_vector[1],marker="+",s=1000)
        pylab.savefig("kmeans_{0}.png".format(iteration_count))
        plot.close()
    
        #calculate new centroids
        new_centroids = []
        num_dup_centroid = 0
        for old_centroid in centroids:
            old_centroid_vector,l = old_centroid
            cluster_data = data[data["label"] == l]
            sum_x = cluster_data["x"].sum()
            sum_y = cluster_data["y"].sum()
            new_centroid_x = sum_x / cluster_data.shape[0]
            new_centroid_y = sum_y / cluster_data.shape[0]
            centroid_vector = np.array([new_centroid_x,new_centroid_y])
            if( (old_centroid_vector == centroid_vector).all() ):
                num_dup_centroid += 1
            new_centroids.append((centroid_vector,l))
        
        #stop if we repeat all previous centroids
        if(num_dup_centroid==k):
            break
        
        centroids = new_centroids
     
if __name__ == '__main__':  
    #my_data = pd.read_csv("km-data.txt",skipinitialspace=True,header=None)     
    my_data = pd.read_csv(sys.argv[1],skipinitialspace=True,header=None)
    
    k = 3
    method = "L2"
    if(len(sys.argv)>=3):
        k = int(sys.argv[2])
    if(len(sys.argv)>=4):
        method = sys.argv[3]
        
    my_data.columns = ["x","y"]
    
    my_data["label"] = 0
         
    kmeans(my_data,k=k,method=method)  
    my_data.to_csv("result.csv",index=False)         
