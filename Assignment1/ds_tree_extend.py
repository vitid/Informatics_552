# -*- coding: utf-8 -*-
"""
Created on Fri Feb 19 23:53:27 2016

@author: vitidn
"""

import pandas as pd
import math
import treenode as tree
import sys

def calculateDataFrameEntropy(d,flag_column):
    prob_array = []
    total = len(d) + 0.0   
    for value in d[flag_column].unique():
        occurence = len(d[d[flag_column] == value])
        prob_array.append(occurence/total)
    return entropy(prob_array)
    
def entropy(a):
    return sum([x * math.log(1/x,2) if x != 0 else 0 for x in a])

def splitTree(data,original_data,parent_split_columns=[],flag_column = "Enjoy"):
    #stop conditions:
    #1. no column to split further
    #2. stop if Tree's entropy is 0 already
    if(len(data.columns) - len(parent_split_columns) == 2 or calculateDataFrameEntropy(data,flag_column=flag_column) == 0):
        most_occured_flag = data[flag_column].value_counts().index[0]
        return tree.Node(most_occured_flag)
        
    num_all_data = len(data)
    split_column = ""
    min_entropy = -1
    for name in data.columns[0:(len(data.columns)-1)]:
        if name in parent_split_columns:
            continue
        
        unique_values = data[name].unique()
        entropies = []
        for value in unique_values:
            sub_data = data[data[name]==value]
            num_sub_data = len(sub_data) + 0.0
            sub_entropy = calculateDataFrameEntropy(sub_data,flag_column=flag_column)
            entropies.append( (num_sub_data/num_all_data) * sub_entropy)
        split_column_entropy = sum(entropies)
        if(min_entropy == -1 or split_column_entropy < min_entropy):
            min_entropy = split_column_entropy
            split_column = name
    
    current_node = tree.Node(split_column)
    unique_values = data[split_column].unique()
    original_unique_values = original_data[split_column].unique()
    for value in original_unique_values:
        if(value not in unique_values):
            most_occured_flag = data[flag_column].value_counts().index[0]
            leaf_node = tree.Node(most_occured_flag)
            current_node.addChild(leaf_node,value)
        else:
            sub_data = data[data[split_column]==value]
            parent = parent_split_columns[:]
            parent.append(split_column)
            child_node = splitTree(sub_data,data,parent_split_columns = parent, flag_column = flag_column )
            current_node.addChild(child_node,value)
    return current_node
    
if __name__ == "__main__":

    my_data = pd.read_csv(sys.argv[1],skipinitialspace=True,header=None)
    
    column_name = my_data[0:1]
    column_name = [(s).replace("(","").replace(")","") for s in column_name.unstack()]
    
    my_data = my_data[1:len(my_data)]
    
    my_data.columns = column_name
    
    my_data["Size"] = my_data["Size"].str[4:]
    my_data["Enjoy"] = my_data["Enjoy"].apply(lambda x: str(x).replace(";",""))
    
    final_tree = splitTree(my_data,my_data,flag_column = "Enjoy")
    
    paths = final_tree.getPaths()
    #generate decision rules
    for path in paths:
        rule = "if"
        for i in range(len(path)-1,0,-1):
            splitter,condition = path[i]
            rule = rule + " {0} = {1}".format(splitter,condition)
            if(i!=1):
                rule = rule + " and"
        rule += " then predict {0}".format(path[0])
        print(rule)
