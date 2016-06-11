import numpy as np
import Queue

class MazeCell(object):
    def __init__(self, i, j, num_step = 0):
        self.i = i 
        self.j = j
        self.num_step = num_step
        
def fastmap(maze_matrix,k,current_dimension=0,coordinate_dict=None,pivot_cell=None):
    """
    
    @param coordinate_dict dictionary of coordinate of cell (i,j)
            ex. {
                (0,0):[1,2,3]
                (0,1):[4,5,6]
                }
    @param pivot_cell list of the index tuples of pivot cells
            ex. [
                ((i0,j0),(i1,j1)),
                ((i2,j2),(i3,j3)),
                ((i4,j4),(i5,j5)),
                ]
            here, cell (i0,j0) and (i1,j1) are use as pivots for the 1st dimension
            
    @return (coordinate_dict,pivot_cell)        
    """
    if(current_dimension == k):
        return (coordinate_dict,pivot_cell)
    
    #the first time method is called
    if(coordinate_dict==None):
        coordinates = [(i,j) for i in range(maze_matrix.shape[0]) for j in range(maze_matrix.shape[1]) if maze_matrix[i,j] == 1]
        coordinate_dict = {}
        for coordinate in coordinates:
            coordinate_dict[coordinate] = []
        pivot_cell = []
    #n - number of walkable cells
    n = len(coordinate_dict) 
    
    #pick objectA
    index_a = np.random.choice(range(n))
    #index_a is (i,j) of cell A
    index_a = coordinate_dict.keys()[index_a]
    #pre-compute distance matrix with A as its center
    dist_matrix_a = performBFS(maze_matrix, index_a)
    
    #get the index of cell B, which is furthest apart from the cell A
    max_distance = -1
    index_b = None
    for i_j_index in coordinate_dict.keys():
        if(i_j_index == index_a):
            distance = 0
        else:
            distance = getDistance(index_a,dist_matrix_a, i_j_index, coordinate_dict, current_dimension-1)
        
        if(distance >= max_distance):
                index_b = i_j_index
                max_distance = distance                 
    
    #pre-compute distance matrix with B as its center
    dist_matrix_b = performBFS(maze_matrix, index_b)    
    
    #NOW!!, re-select cell A again(there is a chance that cell A firstly obtained is not the furthest one)        
    max_distance = -1
    index_a = None
    dist_matrix_a = None
    for i_j_index in coordinate_dict.keys():
        if(i_j_index == index_b):
            distance = 0
        else:
            distance = getDistance(index_b,dist_matrix_b, i_j_index, coordinate_dict, current_dimension-1)
        
        if(distance >= max_distance):
                index_a = i_j_index
                max_distance = distance                 
    
    #pre-compute distance matrix with A as its center
    dist_matrix_a = performBFS(maze_matrix, index_a) 
    #swap cell A and cell B
    index_a,index_b = index_b,index_a
    dist_matrix_a,dist_matrix_b = dist_matrix_b,dist_matrix_a
    
    #record the ids of object_a and object_b
    pivot_cell.append((index_a,index_b))
    
    if(max_distance == 0):
        #set all coordinates in this dimension = 0
        for i_j_index in coordinate_dict.keys():
            coordinate_dict[i_j_index].append(0)
    else:
        for i_j_index in coordinate_dict.keys():
#             distance_ai = getDistance(index_a,dist_matrix_a, i_j_index, coordinate_dict, current_dimension-1)
#             distance_bi = getDistance(index_b,dist_matrix_b, i_j_index, coordinate_dict, current_dimension-1)
#             new_pos = computeMappedCoordinate(max_distance, distance_ai, distance_bi)
#             #if(np.isnan(new_pos)):
#             #    new_pos = 0
#             if(distance_ai > max_distance or distance_bi > max_distance):
#                 print("somethings wrong...")    
#             coordinate_dict[i_j_index].append(new_pos)
            
            distance_ai_sq = getDistanceSquare(index_a,dist_matrix_a, i_j_index, coordinate_dict, current_dimension-1)
            distance_bi_sq = getDistanceSquare(index_b,dist_matrix_b, i_j_index, coordinate_dict, current_dimension-1)
            new_pos = computeMappedCoordinateII(max_distance, distance_ai_sq, distance_bi_sq)
            #if(np.isnan(new_pos)):
            #    new_pos = 0
            #if(distance_ai > max_distance or distance_bi > max_distance):
                #print("somethings wrong...")    
            coordinate_dict[i_j_index].append(new_pos)
                
    return fastmap(maze_matrix, k, current_dimension+1, coordinate_dict, pivot_cell)
        
def computeMappedCoordinate(distance_ab,distance_ai,distance_bi):
    return ((distance_ai ** 2.0) +  (distance_ab ** 2.0) - (distance_bi ** 2.0)) / (2*distance_ab)

def computeMappedCoordinateII(distance_ab,distance_ai_sq,distance_bi_sq):
    return (distance_ai_sq +  (distance_ab ** 2.0) - distance_bi_sq) / (2.0*distance_ab)
    
def getDistance(index_i,dist_matrix_i,index_j,coordinate_dict,current_dimension):
    """
    Calculate distance in current_dimension for the center cell:index_i and a cell:index_j
    
    @param index_i (i,j) of a center cell A
    @param dist_matrix_i distance matrix of a maze, with A as its center
    @param index_j (i,j) of a cell B
    @param coordinate_dict
    @param current_dimension has value -1 for the 1st dimension and so on...
    """
    if current_dimension == -1:
        return dist_matrix_i[index_j]
    
    current_x_pos = coordinate_dict[index_i][current_dimension]
    current_y_pos = coordinate_dict[index_j][current_dimension]
    new_distance = ((getDistance(index_i,dist_matrix_i,index_j,coordinate_dict,current_dimension - 1) ** 2) - ((current_x_pos - current_y_pos) ** 2)) ** 0.5
    return new_distance

def getDistanceSquare(index_i,dist_matrix_i,index_j,coordinate_dict,current_dimension):
    """
    Calculate distance in current_dimension for the center cell:index_i and a cell:index_j
    
    @param index_i (i,j) of a center cell A
    @param dist_matrix_i distance matrix of a maze, with A as its center
    @param index_j (i,j) of a cell B
    @param coordinate_dict
    @param current_dimension has value -1 for the 1st dimension and so on...
    """
    if current_dimension == -1:
        return dist_matrix_i[index_j] ** 2.0
    
    current_x_pos = coordinate_dict[index_i][current_dimension]
    current_y_pos = coordinate_dict[index_j][current_dimension]
    new_distance_sq = (getDistanceSquare(index_i,dist_matrix_i,index_j,coordinate_dict,current_dimension - 1) - ((current_x_pos - current_y_pos) ** 2))
    return new_distance_sq

def performBFS(maze_matrix,center):
    """
    Perform Breadth First Search, starting from the center cell, fill each walkable cell with
    a number of steps from that cell to the center cell
    
    @param maze_matrix - 0 means blocked wall and 1 mean walkable path
    @param center - index (i,j) of the center cell
    
    @retur dist_matrix - distance matrix from each cell to the center cell
    """
    i,j = center[0],center[1]
    center_cell = MazeCell(i,j)
    
    #distance_matrix contains a number of steps from each cell to the center
    dist_matrix = np.zeros(maze_matrix.shape,dtype="int32")
    dist_matrix.fill(-1)
    
    q = Queue.Queue()
    q.put(center_cell)
    
    while(not q.empty()):
        current_cell = q.get(False)
        dist_matrix[current_cell.i,current_cell.j] = current_cell.num_step
        #put adjacent cells of the current_cell into queue if the number of step from it
        #to the center cell is not yet calculated
        
        #upper cell
        if(current_cell.i > 0):
            i = current_cell.i - 1
            j = current_cell.j
            #it must not be a wall and not yet calculated
            if(maze_matrix[i,j] == 1 and
               dist_matrix[i,j] == -1 ):
                adjacent_cell = MazeCell(i,j,current_cell.num_step + 1)
                q.put(adjacent_cell)
        #lower cell
        if(current_cell.i < maze_matrix.shape[0] - 1):
            i = current_cell.i + 1
            j = current_cell.j
            #it must not be a wall and not yet calculated
            if(maze_matrix[i,j] == 1 and
               dist_matrix[i,j] == -1 ):
                adjacent_cell = MazeCell(i,j,current_cell.num_step + 1)
                q.put(adjacent_cell)
        #left cell
        if(current_cell.j > 0):
            i = current_cell.i
            j = current_cell.j - 1
            #it must not be a wall and not yet calculated
            if(maze_matrix[i,j] == 1 and
               dist_matrix[i,j] == -1 ):
                adjacent_cell = MazeCell(i,j,current_cell.num_step + 1)
                q.put(adjacent_cell)
        #right cell
        if(current_cell.j < maze_matrix.shape[1] - 1):
            i = current_cell.i
            j = current_cell.j + 1
            #it must not be a wall and not yet calculated
            if(maze_matrix[i,j] == 1 and
               dist_matrix[i,j] == -1 ):
                adjacent_cell = MazeCell(i,j,current_cell.num_step + 1)
                q.put(adjacent_cell)   
    
    return dist_matrix

def getFastmapCoordinate(maze_matrix,k=5):
    coordinate_dict,pivot_cell = fastmap(maze_matrix, k)
    return coordinate_dict 

def testFastMap():
    """
    Perform test on the maze:
    1 1 1 1 1
    1 0 0 0 1
    1 0 1 0 1
    1 0 1 0 1
    1 1 1 1 1
    """ 
    maze_matrix = np.matrix([[1,1,1,1,1],[1,0,0,0,1],[1,0,1,0,1],[1,0,1,0,1],[1,1,1,1,1,]])
    coordinate_dict,pivot_cell = fastmap(maze_matrix, k=3)
    print(coordinate_dict) 
          
def testPerformBFS():
    """
    Perform test on the maze:
    1 1 1 1 1
    1 0 0 0 1
    1 0 1 0 1
    1 0 1 0 1
    1 1 1 1 1
    
    ceneter_cell is (0,4)
    """    
    maze_matrix = np.matrix([[1,1,1,1,1],[1,0,0,0,1],[1,0,1,0,1],[1,0,1,0,1],[1,1,1,1,1,]])
    dist_matrix = performBFS(maze_matrix, (0,4))
    print(dist_matrix)
if __name__ == "__main__":
    testPerformBFS()
    testFastMap()
        
    