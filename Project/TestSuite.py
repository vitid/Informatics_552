import os
import maze
import time
import matplotlib
import matplotlib.pyplot as pyplot
import matplotlib.pylab as pylab
import numpy as np
import pandas as pd
from project.AStar import EuclideanDistanceAStar, FastmapDistanceAStar
from project import ModifiedFastMapMaze
from project import VanillaFastmapMaze

def generateTravelPathImage(maze_matrix,paths,start_cell,end_cell,img_name):
    path_matrix = maze_matrix.copy()
    for path in paths:
        path_matrix[path] = 2
    path_matrix[start_cell] = 3    
    path_matrix[end_cell] = 4
    pyplot.imshow(path_matrix, cmap=matplotlib.colors.ListedColormap(['black', 'white','green','blue','red']), interpolation='nearest')
    pylab.savefig("./img/" + img_name)
    pyplot.close()
    
def runTestSuite(maze_width_heights,k_tests,initial_maze_id = 0,maze_complexity = 0.9,maze_density = 0.9,is_use_modified_fastmap = True):
    """
    Run test suite based on provided parameters
    
    @param maze_width_heights
    @param k_tests
    """
    
    maze_id = initial_maze_id
    
    for maze_width_height in maze_width_heights:
        maze_width = maze_width_height[0]
        maze_height = maze_width_height[1] 
    
        maze_matrix = maze.maze(maze_width, maze_height,complexity = maze_complexity, density = maze_density)
        print("Finish generating a maze...")
        maze_width = maze_matrix.shape[1]
        maze_height = maze_matrix.shape[0]
            
        #randomly generate the start cell within 20% of distance from (0,0)
        while(True):
            i = np.random.choice(range( int(maze_matrix.shape[0] * 0.2) ))
            j = np.random.choice(range( int(maze_matrix.shape[1] * 0.2) ))
            if(maze_matrix[i,j] == 1):
                break
        start_cell = (i,j)
        #randomly generate the end cell within 20% of distance from (last_row,last_column)
        while(True):
            i = np.random.choice(range( int(maze_matrix.shape[0] * 0.8) , maze_matrix.shape[0] ))
            j = np.random.choice(range( int(maze_matrix.shape[1] * 0.8) , maze_matrix.shape[1] ))
            if(maze_matrix[i,j] == 1):
                break
        end_cell = (i,j)
            
        #extract wall coordinate from the generated maze
        walls = maze.extractWalls(maze_matrix)
        
        #Normal A* with L2 distance
        print("Start Normal A*")
        normalAStar = EuclideanDistanceAStar()
        normalAStar.init_grid(maze_matrix.shape[0], maze_matrix.shape[1], walls, start_cell, end_cell)
        begin_time = time.time()
        pathNormals = normalAStar.solve() 
        end_time = time.time()
        normal_time = (end_time - begin_time)*1000.0
        print("length: {0}, path:{1}".format(len(pathNormals),pathNormals)) 
        print("Normal A* with L2 distance take: {0} ms.".format(normal_time))
        normal_image = str(maze_id) + "_normal_" + str(maze_width) + "_" + str(maze_height)
        generateTravelPathImage(maze_matrix, pathNormals,start_cell,end_cell,normal_image)
        
        #collect the run result and save it into a file
        report_data = pd.read_csv("result_data.template.csv")
        #maze_id,maze_width,maze_height,alg_type,k,gen_coordinate_time,alg_time,alg_num_steps,alg_path,alg_image
        report_data.loc[0] = [maze_id,maze_matrix.shape[1],maze_matrix.shape[0],"normal",-1.0,0.0,normal_time,len(pathNormals),str(pathNormals),normal_image]
        report_data.to_csv("result_data.csv",mode="a",header=False,index=False)
        
        #FastMap A*
        for k in k_tests:
            print("generate dictionary for coordinates in k dimension...")
            begin_time = time.time()
            if(is_use_modified_fastmap):
                coordinate_dict = ModifiedFastMapMaze.getFastmapCoordinate(maze_matrix,k)
            else:
                coordinate_dict = VanillaFastmapMaze.getFastmapCoordinate(maze_matrix, k)
                
            end_time = time.time()
            gen_coordinate_time = (end_time - begin_time)*1000.0
            print("Finish generating dictionary for coordinates in {0} dimension, take: {1} ms.".format(k,gen_coordinate_time))
            
            print("Start FastMap A*")
            fastMapAStar = FastmapDistanceAStar(coordinate_dict)
            fastMapAStar.init_grid(maze_matrix.shape[0], maze_matrix.shape[1], walls, start_cell, end_cell)
            begin_time = time.time()
            pathFastMaps = fastMapAStar.solve()  
            end_time = time.time()
            fastmap_time = (end_time - begin_time)*1000.0
            print("length: {0}, path:{1}".format(len(pathFastMaps),pathFastMaps)) 
            print("FastMap A* take: {0} ms.".format(fastmap_time))
            fastmap_image = str(maze_id) + "_fastmap_" + str(maze_width) + "_" + str(maze_height) + "_" + str(k)
            generateTravelPathImage(maze_matrix, pathFastMaps,start_cell,end_cell,fastmap_image)
            
            #collect the run result and save it into a file
            report_data = pd.read_csv("result_data.template.csv")
            #maze_id,maze_width,maze_height,alg_type,k,gen_coordinate_time,alg_time,alg_num_steps,alg_path,alg_image   
            report_data.loc[0] = [maze_id,maze_matrix.shape[1],maze_matrix.shape[0],"fastmap",k,gen_coordinate_time,fastmap_time,len(pathFastMaps),str(pathFastMaps),fastmap_image]
            report_data.to_csv("result_data.csv",mode="a",header=False,index=False)
    
        maze_id = maze_id + 1        
            
if __name__ == "__main__":
    #change to the current working directory
    os.chdir("./")
    
    maze_width_heights = [(30,30),(50,50)]
    k_tests = [2,3]
    
    runTestSuite(maze_width_heights, k_tests,is_use_modified_fastmap = True)
