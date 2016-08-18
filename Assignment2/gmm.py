'''
Created on Mar 8, 2016

@author: vitidn
'''
import sys
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def gmm_fit(vector_data,k,ll_threshold=0.0001,max_iteration=1000):
    """
    \n vector_data - np.array of data points, each element is np.array of 1 data point
    \n k - number of clusters
    \n ll_threshold - stop condition: log-likelihood increased < ll_threshold
    \n max_iteration - stop condition: number of iterations exceeds max_iteration
    \n return - (probs,mus,cov_matrixs)
    \n probs - each row is the probability that each data point belongs to each gaussian
    \n         format: np.array([[],[],...]) with size number_of_data x k
    \n mus   - list of the mean vector of each guassian
    \n cov_matrixs - list of the covariance matrix of each guassian
    """
    #calculate initial covariance-matrix
    cov_matrix = generateCovarianceMatrix(generateNormalizedMatrix( np.matrix(vector_data)))
    cov_matrixs = [cov_matrix] * k
    #change each data point to dx1 matrix
    vector_data = [np.matrix(x).transpose() for x in vector_data]
    #number of data points
    n = len(vector_data)
    #pick k initial mu randomly
    mus = np.random.choice(range(n),replace=False,size=k)
    mus = [vector_data[i] for i in mus]
    #pick the initial amplitude
    a = (1.0 / k )
    a_s = [a] * k
    #construct n x k weight matrix
    weights = np.zeros((n,k))
    iteration = 0
    #store pas log-likelihood for stopping condition
    current_ll = 0.0
    while(iteration < max_iteration):
        iteration += 1
        #E-Step
        #update weights of each data point
        for i in range(weights.shape[0]):
            pdfs = [ gaussianPdf(vector_data[i], mus[j], cov_matrixs[j])*a_s[j] for j in range(k) ]
            sumPdf = np.sum(pdfs)
            sub_weights = pdfs/sumPdf
            weights[i] = sub_weights
        
        #M-Step
        #assign new value to amplitudes
        n_k = np.apply_along_axis(np.sum, 0, weights)
        a_s = n_k / n
        #compute new mus
        for i in range(k):
            mus[i] = (1.0/n_k[i]) * sum([weights[j,i] * vector_data[j] for j in range(n)])
        #compute new covariance_matrixs
        for i in range(k):
            cov_matrixs[i] = (1.0/n_k[i]) * sum([weights[j,i] * (vector_data[j]-mus[i]) * (vector_data[j]-mus[i]).transpose() for j in range(n)])
    
        #compute log-likelihood
        ll = 0.0
        for j in range(n):
            sub_ll = sum([a_s[i] * gaussianPdf(vector_data[j], mus[i], cov_matrixs[i] ) for i in range(k)])
            ll += np.log(sub_ll)
        print("iteration {0}, log-likelihood:{1}".format(iteration,ll))
        #stop the algorithm if it doesn't improve much
        if(current_ll!=0 and ll-current_ll <= ll_threshold):
            break
        
        current_ll = ll
        
    return (weights,mus,cov_matrixs)        

def generateNormalizedMatrix(data_matrix):
    
    normalized_matrix = data_matrix.copy()
    means = np.apply_along_axis(np.mean, 0, data_matrix)
    
    for i in range(data_matrix.shape[1]):
        normalized_matrix[:,i] = data_matrix[:,i] - means[i]
        
    return normalized_matrix
        
def generateCovarianceMatrix(normalized_matrix):
    """
    \n normalized_matrix - matrix of size num_observation x num_dimension
    \n format: supposed num_observation=3, num_dimension = 5
    \n np.matrix([[A1,A2,A3,A4,A5],[B1,B2,B3,B4,B5],[C1,C2,C3,C4,C5]])
    """
    num_observation = normalized_matrix.shape[0]
    covariance_matrix = normalized_matrix.transpose() * normalized_matrix
    return covariance_matrix / (num_observation - 1.0)

def gaussianPdf(x,mu,covariance_matrix):
    """
    \n x - data point, size dx1
    \n mu - the center of gaussian, size dx1
    \n covariance_matrix - size dxd
    """
    d = covariance_matrix.shape[0]
    scalar = ((x-mu).transpose() * np.linalg.inv(covariance_matrix) * (x-mu))[0,0]
    pdf_value = (np.e ** (-0.5 * scalar )) / ((((2*np.pi) ** d) * (np.linalg.det(covariance_matrix))) ** 0.5)
    return pdf_value

if __name__ == "__main__":
    #filename = sys.argv[1]
    k = 3 
    filename = "/home/vitidn/Dropbox/MS/INF552/Assignment2/clusters.txt"
    my_data = pd.read_csv(filename,skipinitialspace=True,header=None)
    my_data.columns = ["x","y"]
    vector_data = [np.array([row["x"],row["y"]]) for index,row in my_data.iterrows() ]
    vector_data = np.array(vector_data)   
    weights,mus,cov_matrixs = gmm_fit(vector_data, k)
    #assign label according to the class that has max prob.
    labels = np.argmax(weights,axis=1)
    my_data["label"] = labels
    #plot the data
    labels = (labels + 0.0) / k
    plt.scatter(x=my_data["x"],y=my_data["y"],c=plt.cm.rainbow(labels))
    plt.scatter(x=[x[0] for x in mus],y=[x[1] for x in mus],marker="+",s=1000)
    #output the probability each data belong
    prob_data = pd.DataFrame(weights)
    prob_data.columns = ["cluster:" + str(i) for i in range(k)]
    prob_data.to_csv("result_1.csv",index=False)  
    