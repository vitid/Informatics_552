# -*- coding: utf-8 -*-
"""
Created on Tue Feb  2 00:04:54 2016

@author: vitidn
"""
import json

class Node:
    def __init__(self, divider):
        self.divider = divider
        self.options = []
        
    def addChild(self, child_node, matched_value):
        self.options.append((matched_value,child_node))
        
    def toJson(self):
        return json.dumps(self,default = lambda o:o.__dict__)
       
    #get all paths from leafs to a root
    def getPaths(self):
        if(len(self.options) == 0):
            return [[(self.divider)]]
            
        paths = []    
        for option in self.options:
            conditon,child = option
            child_paths = child.getPaths()
            for child_path in child_paths:
                child_path.append((self.divider,conditon))
                paths.append(child_path)
        return paths

if __name__ == "__main__":        
    a = Node("A")
    b = Node("B")
    a.addChild(b,"link1")
    c = Node("C")
    a.addChild(c,"link2")
    d = Node("D")
    e = Node("E")
    b.addChild(d,"link3")
    b.addChild(e,"link4")
    print(a.toJson())
    a.getPaths()
    