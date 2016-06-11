import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plot
import seaborn as sns
import cvxopt
import cvxopt.solvers
import argparse

class SVM:
    def __init__(self,cutoff = 1e-5):
        self.weights = None
        self.bias = None
        self.support_vector_ids = []
        self.cutoff = cutoff
        
    def fitSVM(self,data,labels):
        num_data = len(data)
        dimension = len(data[0])
        P = np.zeros((num_data,num_data))
        for i in range(num_data):
            for j in range(num_data):
                P[i,j] = labels[i] * labels[j] * np.dot(data[i] , data[j])
        
        #re-arrange to QP Solver: (1/2)x(T)Px + q(T)x
        #optimization for x (alpha)
        #under constrains:
        P = cvxopt.matrix(P)
        q = cvxopt.matrix(np.ones(num_data) * -1)
        
        #(1) Gx <= h
        h = cvxopt.matrix(np.zeros(num_data))
        G = np.zeros((num_data,num_data))
        np.fill_diagonal(G, -1.0)
        G = cvxopt.matrix(G)
        #(2) Ax = b
        A = cvxopt.matrix(labels,(1,num_data))
        b = cvxopt.matrix(0.0)
        
        solver = cvxopt.solvers.qp(P, q, G, h, A, b)
        alphas = np.ravel(solver['x'])
        
        self.support_vector_ids = np.argwhere(alphas > self.cutoff).transpose()[0]
        #calculate weights
        self.weights = np.zeros((1,dimension))[0]
        for id in np.nditer(self.support_vector_ids):
            self.weights += np.array(alphas[id] * labels[id] * data[id])
        #calculate biases
        sv_id = self.support_vector_ids[0]
        self.bias = (1.0/labels[sv_id]) - np.dot(self.weights,data[sv_id])
    
    def predict(self,data):
        predicted = [np.dot(d,self.weights) + self.bias for d in data]    
        return [-1 if p < 0 else 1 for p in predicted]
           
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Test SVM')
    parser.add_argument("file",metavar="file_path", type=argparse.FileType('r', 0), help="path of the tested data file")
    parser.add_argument("-cutoff", type=float, help="cutoff weight to consider to be a support vector (default: %(default)s)",default = 1e-5)
 
    args = parser.parse_args()
 
    my_data = pd.read_csv(args.file,skipinitialspace=True,header=None,delimiter=" ")
    my_data.columns = ["x","y","label"]  
    
    data = [np.array([row["x"]**2,row["y"]**2]) for index,row in my_data.iterrows() ]
    labels = np.array(my_data["label"])
    labels = [float(l) for l in labels]
    
    #train SVM
    svm = SVM(args.cutoff)
    svm.fitSVM(data, labels)
    
    #predict with the transform data
    predicted_labels = svm.predict(data)
    pairs = zip(labels,predicted_labels)
    t = [1 for p in pairs if p[0] != p[1]]
    print("")
    print("Accuracy: {0}".format(1.0 - (sum(t) + 0.0)/len(data)))
    
    #visualize the result
    
    actual_label_colors = ["r" if l == -1 else "g" for l in labels]
    #transformed data
    plot.scatter(my_data["x"]**2,my_data["y"]**2,c=actual_label_colors)
    #original data
    plot.scatter(my_data["x"],my_data["y"],c=actual_label_colors)
    
    predicted_label_colors = ["r" if l == -1 else "g" for l in predicted_labels]
    #transformed data
    plot.scatter(my_data["x"] ** 2,my_data["y"] ** 2,c=predicted_label_colors)
    #plot the decision boundary
    plot.plot([200,(-svm.bias - 200.0 * svm.weights[1]) / svm.weights[0]],[(-svm.bias - 200.0*svm.weights[0]) / svm.weights[1],200])
    plot.xlim([-100,700])
    plot.ylim([-100,700])
    sv_x = [data[point_id][0] for point_id in svm.support_vector_ids]
    sv_y = [data[point_id][1] for point_id in svm.support_vector_ids]
    plot.scatter(sv_x,sv_y,c="y")
    #original data
    plot.scatter(my_data["x"],my_data["y"],c=predicted_label_colors)
    plot.scatter(np.array(my_data.iloc[svm.support_vector_ids,[0]]),np.array(my_data.iloc[svm.support_vector_ids,[1]]),c="y",s=50)
    x = np.linspace(-10, 10, 100)
    y = ((-svm.bias - svm.weights[0] * (x**2) ) / svm.weights[1]) ** 0.5
    plot.plot(x,y,c="b")
    plot.plot(x,-y,c="b")
    
    #print out the model
    print("")
    print("Weights: {0}".format(svm.weights))
    print("Bias: {0}".format(svm.bias))
    print("Support Vector Id: {0}".format(svm.support_vector_ids))
    print("")
    print("Support Vector:")
    for p in zip(sv_x,sv_y):
        print("({0},{1})".format(p[0],p[1]))
    print("")
    print("Pre-image of Support Vector")
    print(my_data.iloc[svm.support_vector_ids,[0,1]])
