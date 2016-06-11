import numpy
from numpy.random import random_integers as rand

def maze(width=81, height=51, complexity=.75, density=.75):
    """
    Generate a maze matrix
    0 - wall
    1 - walkable cell
    @return maze_matrix 
    """
    # Only odd shapes
    shape = ((height // 2) * 2 + 1, (width // 2) * 2 + 1)
    # Adjust complexity and density relative to maze size
    complexity = int(complexity * (5 * (shape[0] + shape[1])))
    density    = int(density * ((shape[0] // 2) * (shape[1] // 2)))
    # Build actual maze
    Z = numpy.ones(shape, dtype="int32")
    # Fill borders
    Z[0, :] = Z[-1, :] = 0
    Z[:, 0] = Z[:, -1] = 0
    # Make aisles
    for i in range(density):
        x, y = rand(0, shape[1] // 2) * 2, rand(0, shape[0] // 2) * 2
        Z[y, x] = 0
        for j in range(complexity):
            neighbours = []
            if x > 1:             neighbours.append((y, x - 2))
            if x < shape[1] - 2:  neighbours.append((y, x + 2))
            if y > 1:             neighbours.append((y - 2, x))
            if y < shape[0] - 2:  neighbours.append((y + 2, x))
            if len(neighbours):
                y_,x_ = neighbours[rand(0, len(neighbours) - 1)]
                if Z[y_, x_] == 1:
                    Z[y_, x_] = 0
                    Z[y_ + (y - y_) // 2, x_ + (x - x_) // 2] = 0
                    x, y = x_, y_
    return Z

def extractWalls(maze_matrix):
    """
    Get all coordinate: (i,j) of wall cells
    @return list of cooridnates (i,j) - [(0,1),(1,2),...]
    """
    walls = []
    for i in range(maze_matrix.shape[0]):
        for j in range(maze_matrix.shape[1]):
            if(maze_matrix[i,j] == 0):
                walls.append((i,j))
    return walls
    