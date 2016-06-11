import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plot
import seaborn as sns
import argparse

class NeuralNetwork:
    
    def __init__(self,num_input_nodes,num_output_nodes,num_node_hidden_layers,learning_rate = 0.1,num_batch=10000,sgd_size=10):
        self.num_nodes = list(num_node_hidden_layers)
        self.num_nodes.insert(0, num_input_nodes)
        self.num_nodes.append(num_output_nodes)
        self.learning_rate = learning_rate
        self.num_batch = num_batch
        self.sgd_size = sgd_size
        self.use_sgd = (self.sgd_size > 0)
        self.initWeightAndBias()
        self.mispredict_ratio = -1.0
        
    def initWeightAndBias(self):
        #list of weight matrices in each layer from (0) to (num_layer - 2)
        self.weight_matrix_list = [None] * (len(self.num_nodes) - 1)
        #list of biases in each layer from (0) to (num_layer - 2)
        self.bias_list = [None] * (len(self.num_nodes) - 1)
        
        for i in range(0,len(self.num_nodes)-1):
            num_node_l_plus_1 = self.num_nodes[i+1]
            num_node_l = self.num_nodes[i]
            self.weight_matrix_list[i] = np.random.rand(num_node_l_plus_1,num_node_l)
            self.bias_list[i] = np.random.rand(1,self.num_nodes[i+1])[0,:]
            
    def hyperbolicTangentFunction(self,x):
        t = np.e ** (2 * x)
        return (t - 1.0) / (t + 1.0)
    
    def initActivationSignal(self,init_signals):
        """
        list of activation signal 
        a(0) are plain input values and a(num_layer - 1) are output values
        e.g. [[0,0],[0,0,0],[0]] corresponding to 3 layers, input layer has x1 and x2
        """
        a_list = [None] * (len(self.num_nodes))
        for i in range(len(a_list)):
            a_list[i] = [0.0] * self.num_nodes[i]
            
        a_list[0] = init_signals    
            
        return a_list    
        
    def trainNeuralNetwork(self,data,labels):
        """
        data & labels - a list of list of observation
        """
        #run for maximum num_batch
        for k in range(self.num_batch):
            delta_weights = [None] * len(self.weight_matrix_list)
            delta_biases = [None] * len(self.bias_list)
            
            if(self.use_sgd):
                row_ids = np.random.choice(range(0,len(data)),replace=False,size=self.sgd_size)
            else:    
                row_ids = range(0,len(data))
            num_row_per_batch = len(row_ids)   
            
            #update weights and biases for one mini-batch
            for i in range(num_row_per_batch):
                record = data[row_ids[i]]
                output_label = labels[row_ids[i]]
                a_list = self.initActivationSignal(record)
                #update activation signals
                self.forwardSignal(a_list)
                #delta_list for updating weights and biases
                delta_list = self.backPropagation(a_list, output_label)
                #update delta weight and delta bias
                for j in range(len(delta_weights)):
                    delta_vector = np.copy(delta_list[j+1])
                    delta_vector.shape = (delta_vector.shape[0],1)
                    gradient_weight = np.dot(delta_vector,np.array([a_list[j]]))
                    gradient_bias = delta_vector
                    
                    if(delta_weights[j] == None):
                        delta_weights[j] = gradient_weight
                        delta_biases[j] = gradient_bias
                    else:
                        delta_weights[j] = delta_weights[j] + gradient_weight
                        delta_biases[j] = delta_biases[j] + gradient_bias
            
            #update weight and bias
            for i in range(len(self.weight_matrix_list)):
                self.weight_matrix_list[i] = self.weight_matrix_list[i] - ((self.learning_rate / num_row_per_batch) * delta_weights[i])
                self.bias_list[i] = self.bias_list[i] - ((self.learning_rate / num_row_per_batch) * delta_biases[i].transpose())[0,:]
            
            #calculate current error
            num_mispredicted = 0.0
            for i in range(len(data)):
                predicted_labels = self.predictLabels(data[i])
                if predicted_labels != labels[i]:
                    num_mispredicted += 1
            self.mispredict_ratio = num_mispredicted / len(data) 
            if(num_mispredicted == 0):
                break
               
    def forwardSignal(self,a_list):
        for i in range(1,len(a_list)):
            for j in range(len(a_list[i])):
                a_vector = np.array(a_list[i-1])
                row_vector = self.weight_matrix_list[i-1][j,:]
                a_list[i][j] = self.hyperbolicTangentFunction(a_vector.dot(row_vector) + self.bias_list[i-1][j])
                
    def backPropagation(self,a_list,output_label): 
        #delta_list[0] will not be used
        delta_list = [None] * len(a_list)
        #back propagate from the output layer 
        for i in range(len(a_list)-1,0,-1):
            activations = np.array(a_list[i])
            
            #compute derivative f'(z) = a(1-a) (f is a hyperbolic tangent function)
            f_primes = [ 1.0 - (a ** 2) for a in activations]
            
            #delta for output layer is updated differently
            if(i == len(a_list) - 1):
                output_labels = np.array(output_label)
                delta_layer = -(output_labels - activations) * f_primes
                delta_list[i] = delta_layer
                
            else:
                weight_matrix = self.weight_matrix_list[i]
                delta_layer = np.dot(weight_matrix.transpose(),delta_list[i+1]) * f_primes
                delta_list[i] = delta_layer
        return delta_list        
    
    def predictLabels(self,features):
        a_list = self.initActivationSignal(features)
        self.forwardSignal(a_list)
        predicted = a_list[-1]
        return [ 1 if p > 0 else -1 for p in predicted]

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Test NeuralNetwork')
    parser.add_argument("file",metavar="file_path", type=argparse.FileType('r', 0), help="path of the tested data file")
    parser.add_argument("hidden_layer_nodes",metavar="N",nargs='+', type=int, help="number of nodes in each hidden layer")
    parser.add_argument("-learning_rate", type=float, help="learning rate for the model (default: %(default)s)",default = 0.1)
    parser.add_argument("-num_batch", type=int, help="maximum number of batch to perform the update (default: %(default)s)",default = 10000)
    parser.add_argument("-sgd_size", type=int, help="number of batch size for Stochastic Gradient Descent(specify as -1 if you do not want to use SGD) (default: %(default)s)",default = 10)
    
    args = parser.parse_args()
    
    my_data = pd.read_csv(args.file,skipinitialspace=True,header=None,delimiter=" ")
    my_data.columns = ["x","y","label"]  
      
    train_data = my_data.iloc[45:70,:]
    data = [np.array([row["x"],row["y"]]) for index,row in train_data.iterrows() ]
    labels = np.array(train_data["label"])
    labels = [[l] for l in labels]
    num_input_nodes = len(data[0])
    num_output_nodes = len(labels[0])
    #train the neural network
    neuralNetwork = NeuralNetwork(num_input_nodes, num_output_nodes,
                                   num_node_hidden_layers = args.hidden_layer_nodes,
                                   learning_rate = args.learning_rate,
                                   num_batch = args.num_batch,
                                   sgd_size = args.sgd_size)
    neuralNetwork.trainNeuralNetwork(data, labels)
    
    print("Training Accuracy: {0}".format(1.0 - neuralNetwork.mispredict_ratio))
    print("*"*50)   
    #Run neural network on test data
    row_ids = range(0,45)
    row_ids.extend(range(70,100)) 
    
    test_data = my_data.iloc[row_ids,:]  
    data = [np.array([row["x"],row["y"]]) for index,row in test_data.iterrows() ]
    labels = np.array(test_data["label"])
    labels = [[l] for l in labels]
    
    num_mispredicted = 0.0
    predicted_labels = []
    for i in range(len(data)):
        predicted_label = neuralNetwork.predictLabels(data[i])
        if(predicted_label != labels[i]):
            num_mispredicted += 1
        predicted_labels.append(predicted_label[0])
    num_mispredicted_ratio = num_mispredicted / len(data) 
    print("Testing Accuracy: {0}".format(1.0 - num_mispredicted_ratio))
    
    #plot the original 75 data points
    actual_labels = ["r" if l[0] == -1 else "g" for l in labels]
    plot.scatter(test_data["x"],test_data["y"],c=actual_labels)
    
    #plot the predicted 75 data points
    predicted_label_colors = ["r" if l == -1 else "g" for l in predicted_labels]
    plot.scatter(test_data["x"],test_data["y"],c=predicted_label_colors)
    
    test_data["predicted_label"] = predicted_labels
