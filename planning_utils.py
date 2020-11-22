from enum import Enum
from queue import PriorityQueue
import numpy as np

from bresenham import bresenham
from scipy.spatial import Voronoi
import networkx as nx

from sklearn.neighbors import KDTree
from shapely.geometry import Polygon, Point, LineString

def create_grid(data, safety_distance):
    """
    Returns a grid representation of a 2D configuration space
    based on given obstacle data, drone altitude and safety distance
    arguments.
    """

    # minimum and maximum north coordinates
    north_min = np.floor(np.min(data[:, 0] - data[:, 3]))
    north_max = np.ceil(np.max(data[:, 0] + data[:, 3]))

    # minimum and maximum east coordinates
    east_min = np.floor(np.min(data[:, 1] - data[:, 4]))
    east_max = np.ceil(np.max(data[:, 1] + data[:, 4]))

    # given the minimum and maximum coordinates we can
    # calculate the size of the grid.
    north_size = int(np.ceil(north_max - north_min))
    east_size = int(np.ceil(east_max - east_min))

    # Initialize an empty grid
    grid = np.zeros((north_size, east_size))

    # Populate the grid with obstacles
    for i in range(data.shape[0]):
        north, east, alt, d_north, d_east, d_alt = data[i, :]
        obstacle = [
            int(np.clip(north - d_north - safety_distance - north_min, 0, north_size-1)),
            int(np.clip(north + d_north + safety_distance - north_min, 0, north_size-1)),
            int(np.clip(east - d_east - safety_distance - east_min, 0, east_size-1)),
            int(np.clip(east + d_east + safety_distance - east_min, 0, east_size-1)),
        ]

        # HACK TO WORK AROUND BUG - SIM ALTITUDE DATA <> COLLIDERS ALTITUDE DATA 
        # FOR BUILDING AROUND GRID COORDINATES (383, 658)
        if alt < safety_distance and \
            (north > -15 and north < 160) and \
            (east > 120 and east < 310):
                alt = 25
                d_alt = 25

        grid[obstacle[0]:obstacle[1]+1, obstacle[2]:obstacle[3]+1] = alt + d_alt

    return grid, int(north_min), int(east_min)


def create_graph(data, grid, north_min, east_min, safety_distance):
    """
    Returns a graph from the Voronoi graph edges given obstacle data and the
    drone's altitude.
    """
    # Define a list to hold Voronoi points
    points = []
    # Populate the grid with obstacles
    for i in range(data.shape[0]):
        north, east, alt, d_north, d_east, d_alt = data[i, :]

        if alt > safety_distance:           
            # add center of obstacles to points list
            points.append([north - north_min, east - east_min])

    # TODO: create a voronoi graph based on
    # location of obstacle centres
    graph = Voronoi(points)

    north_size = grid.shape[0]
    east_size = grid.shape[1]
    
    # TODO: check each edge from graph.ridge_vertices for collision
    sd = 2
    edges = []
    for v in graph.ridge_vertices:
        p1 = graph.vertices[v[0]]
        p2 = graph.vertices[v[1]]

        # Then you can test each pair p1 and p2 for collision using Bresenham
        # (need to convert to integer if using prebuilt Python package)
        # If the edge does not hit an obstacle
        # add it to the list
        bList = bresenham(int(p1[0]), int(p1[1]), int(p2[0]), int(p2[1]))
        in_collision = False
        for pt in bList:
            n = int(pt[0])
            e = int(pt[1])
        
            if (n < 0 or e < 0):
                 in_collision = True
                 break
                
            if (n >= north_size or e >= east_size):
                 in_collision = True
                 break
            
            # grid[n, e] is the altitude
            if (grid[n-sd:n+sd, e-sd:e+sd] > safety_distance).any():
                 in_collision = True
                 break
                     
        if not in_collision:
            # convert from array to tuple for future graph creation step - otherwise you will get an error
            p1 = (int(p1[0]), int(p1[1]))
            p2 = (int(p2[0]), int(p2[1]))
            edges.append((p1, p2))        
    
    G = nx.Graph()
    for e in edges:
        p1 = e[0]
        p2 = e[1]
        dist = np.linalg.norm(np.array(p2) - np.array(p1))
        G.add_edge(p1, p2, weight=dist)    
        
    return G

# Assume all actions cost the same.
class Action(Enum):
    """
    An action is represented by a 3 element tuple.

    The first 2 values are the delta of the action relative
    to the current grid position. The third and final value
    is the cost of performing the action.
    """

    WEST = (0, -1, 1)
    EAST = (0, 1, 1)
    NORTH = (-1, 0, 1)
    SOUTH = (1, 0, 1)

    # diagonal actions
    NORTHEAST = (-1, 1, 1.414)
    NORTHWEST = (-1, -1, 1.414)
    SOUTHEAST = (1, 1, 1.414)
    SOUTHWEST = (1, -1, 1.414)

    @property
    def cost(self):
        return self.value[2]

    @property
    def delta(self):
        return (self.value[0], self.value[1])


def valid_actions(grid, current_node, safety_distance):
    """
    Returns a list of valid actions given a grid and current node.
    """
    valid_actions = list(Action)
    n, m = grid.shape[0] - 1, grid.shape[1] - 1
    x, y = current_node

    # check if the node is off the grid or
    # it's an obstacle

    if x - 1 < 0 or grid[x - 1, y] > safety_distance:
        valid_actions.remove(Action.NORTH)
        valid_actions.remove(Action.NORTHEAST)
        valid_actions.remove(Action.NORTHWEST)
    if x + 1 > n or grid[x + 1, y] > safety_distance:
        valid_actions.remove(Action.SOUTH)
        valid_actions.remove(Action.SOUTHEAST)
        valid_actions.remove(Action.SOUTHWEST)

    if y - 1 < 0 or grid[x, y - 1] > safety_distance:
        valid_actions.remove(Action.WEST)
        if Action.NORTHWEST in valid_actions:
            valid_actions.remove(Action.NORTHWEST)
        if Action.SOUTHWEST in valid_actions:
            valid_actions.remove(Action.SOUTHWEST)
    if y + 1 > m or grid[x, y + 1] > safety_distance:
        valid_actions.remove(Action.EAST)
        if Action.NORTHEAST in valid_actions:
            valid_actions.remove(Action.NORTHEAST)
        if Action.SOUTHEAST in valid_actions:
            valid_actions.remove(Action.SOUTHEAST)

    return valid_actions


def grid_a_star(grid, h, start, goal, safety_distance):

    path = []
    path_cost = 0
    queue = PriorityQueue()
    queue.put((0, start))
    visited = set(start)

    branch = {}
    found = False
    
    while not queue.empty():
        item = queue.get()
        current_node = item[1]
        if current_node == start:
            current_cost = 0.0
        else:              
            current_cost = branch[current_node][0]
            
        if current_node == goal:        
            found = True
            break
        else:
            for action in valid_actions(grid, current_node, safety_distance):
                # get the tuple representation
                da = action.delta
                next_node = (current_node[0] + da[0], current_node[1] + da[1])
                branch_cost = current_cost + action.cost
                queue_cost = branch_cost + h(next_node, goal)
                
                if next_node not in visited:                
                    visited.add(next_node)               
                    branch[next_node] = (branch_cost, current_node, action)
                    queue.put((queue_cost, next_node))
             
    if found:
        # retrace steps
        n = goal
        path_cost = branch[n][0]
        path.append(goal)
        while branch[n][1] != start:
            path.append(branch[n][1])
            n = branch[n][1]
        path.append(branch[n][1])
    else:
        print('**********************')
        print('Failed to find a path!')
        print('**********************') 
    return path[::-1], path_cost

def heuristic(position, goal_position):
    return np.linalg.norm(np.array(position) - np.array(goal_position))

def graph_a_star(nxGraph, h, start, goal, failIfPathNotFound=False):
    path = []
    path_cost = 0
    queue = PriorityQueue()
    queue.put((0, start))
    visited = set(start)

    branch = {}
    found = False

    # disjointed edges, find node nearest to goal that has a path to start
    foundPath2Goal = nx.has_path(nxGraph,start,goal)
    if (not foundPath2Goal) and failIfPathNotFound:
        print('**********************')
        print('Failed to find a path!')
        print('**********************') 
        return foundPath2Goal, None, path_cost

    listNodes = list(nxGraph)
    if not foundPath2Goal:
        closestNodesIndices = np.argsort(np.linalg.norm(goal - np.array(listNodes), axis=1))
        for idx in closestNodesIndices:
            node = listNodes[idx]
            if (start == node or goal == node):
                continue
                
            if nx.has_path(nxGraph,start,node):
                goal = node
                break

    while not queue.empty():
        item = queue.get()
        current_node = item[1]
        if current_node == start:
            current_cost = 0.0
        else:              
            current_cost = branch[current_node][0]
            
        if current_node == goal:        
            # print('Found a path.')
            found = True
            break
        else:
            for next_node in nxGraph[current_node]:
                cost = nxGraph[current_node][next_node]['weight']
                branch_cost = current_cost + cost          
                queue_cost = branch_cost + h(next_node, goal)
                
                if next_node not in visited:                
                    visited.add(next_node)               
                    branch[next_node] = (branch_cost, current_node)
                    queue.put((queue_cost, next_node))
             
    if found:
        # retrace steps
        n = goal
        path_cost = branch[n][0]
        path.append(goal)
        while branch[n][1] != start:
            path.append(branch[n][1])
            n = branch[n][1]
        path.append(branch[n][1])
    else:
        print('**********************')
        print('Failed to find a path!')
        print('**********************') 
    return foundPath2Goal, path[::-1], path_cost

def closestNode(graph, node):
    listNodes = list(graph.nodes)
    closestNodeIndex = np.linalg.norm(node - np.array(listNodes), axis=1).argmin()
    return listNodes[closestNodeIndex]

def local_position_2_grid_coord(localPos, grid, north_offset, east_offset):
    north_size = grid.shape[0]
    grid_north = int(np.clip(localPos[0] - north_offset, 0, north_size-1))
    east_size = grid.shape[1]
    grid_east = int(np.clip(localPos[1] - east_offset, 0, east_size-1))
    return (grid_north, grid_east)

def grid_coord_2_local_position(grid_Coord, north_offset, east_offset):
    north = grid_Coord[0] + north_offset
    east = grid_Coord[1] + east_offset
    return (north, east)

# TODO - change to 3d collinearity test. ok for now since we increase altitudes linearly
def are_collinear(p1, p2, p3): 
    epsilon = 0.01
    collinear = False
    # TODO: Calculate the determinant of the matrix using integer arithmetic 
    det = p1[0]*(p2[1]-p3[1]) + p2[0]*(p3[1]-p1[1]) + p3[0]*(p1[1]-p2[1])
    # TODO: Set collinear to True if the determinant is equal to zero
    if abs(det) < epsilon:
        collinear = True
        
    return collinear

# borrowed from https://www.geeksforgeeks.org/bresenhams-algorithm-for-3-d-line-drawing/
def Bresenham3D(x1, y1, z1, x2, y2, z2): 
    ListOfPoints = [] 
    ListOfPoints.append((x1, y1, z1)) 
    dx = abs(x2 - x1) 
    dy = abs(y2 - y1) 
    dz = abs(z2 - z1) 
    if (x2 > x1): 
        xs = 1
    else: 
        xs = -1
    if (y2 > y1): 
        ys = 1
    else: 
        ys = -1
    if (z2 > z1): 
        zs = 1
    else: 
        zs = -1
  
    # Driving axis is X-axis" 
    if (dx >= dy and dx >= dz):         
        p1 = 2 * dy - dx 
        p2 = 2 * dz - dx 
        while (x1 != x2): 
            x1 += xs 
            if (p1 >= 0): 
                y1 += ys 
                p1 -= 2 * dx 
            if (p2 >= 0): 
                z1 += zs 
                p2 -= 2 * dx 
            p1 += 2 * dy 
            p2 += 2 * dz 
            ListOfPoints.append((x1, y1, z1)) 
  
    # Driving axis is Y-axis" 
    elif (dy >= dx and dy >= dz):        
        p1 = 2 * dx - dy 
        p2 = 2 * dz - dy 
        while (y1 != y2): 
            y1 += ys 
            if (p1 >= 0): 
                x1 += xs 
                p1 -= 2 * dy 
            if (p2 >= 0): 
                z1 += zs 
                p2 -= 2 * dy 
            p1 += 2 * dx 
            p2 += 2 * dz 
            ListOfPoints.append((x1, y1, z1)) 
  
    # Driving axis is Z-axis" 
    else:         
        p1 = 2 * dy - dz 
        p2 = 2 * dx - dz 
        while (z1 != z2): 
            z1 += zs 
            if (p1 >= 0): 
                y1 += ys 
                p1 -= 2 * dz 
            if (p2 >= 0): 
                x1 += xs 
                p2 -= 2 * dz 
            p1 += 2 * dy 
            p2 += 2 * dx 
            ListOfPoints.append((x1, y1, z1)) 

    return ListOfPoints 

# this function assumes that samplePt1 and samplePt2 are inside the grid
def can_connect_3d(grid, p1, p2, safety_distance):
    bList = Bresenham3D(int(p1[0]), int(p1[1]), int(p1[2]), 
                        int(p2[0]), int(p2[1]), int(p2[2]))
    sd = 2
    in_collision = False
    for pt in bList:
        n = int(pt[0])
        e = int(pt[1])
        a = int(pt[2])
        
        # grid[n, e] is the altitude
        if (grid[n-sd:n+sd, e-sd:e+sd] + safety_distance > a).any():
            in_collision = True
            break
                     
    return not in_collision        

def prune_path_collinearity(path):
    idx = 0
    while idx < len(path)-2:
        if are_collinear(path[idx], path[idx+1], path[idx+2]):
            path.pop(idx+1)
            continue
        
        idx += 1

    return path

def prune_path_bresenham(path, grid, safety_distance):
    idx = 0
    while idx < len(path)-2:
        if can_connect_3d(grid, path[idx], path[idx+2], safety_distance):
            path.pop(idx+1)
            continue
        
        idx += 1

    return path

def createProbabilisticRoadMap(grid, grid_start, grid_goal, safety_distance):
    nmin = grid_start[0] if grid_start[0] < grid_goal[0] else grid_goal[0]
    nmax = grid_start[0] if grid_start[0] > grid_goal[0] else grid_goal[0]
    emin = grid_start[1] if grid_start[1] < grid_goal[1] else grid_goal[1]
    emax = grid_start[1] if grid_start[1] > grid_goal[1] else grid_goal[1]

    amin = grid_start[2] + safety_distance
    amax = grid[nmin-safety_distance:nmax+safety_distance, 
                emin-safety_distance:emax+safety_distance].max() + 2*safety_distance

    # we are tackling situation where start and goal are on opposite sides of a barrier
    # only reachable by going over the top or through some opening
    num_of_samples = 300

    nsamples = np.random.random_integers(nmin, nmax, num_of_samples)
    esamples = np.random.random_integers(emin, emax, num_of_samples)
    asamples = np.random.random_integers(amin, amax, num_of_samples)
    samplePoints = list(zip(nsamples, esamples, asamples))

    # throw out ones that are inside or too close to an obstruction
    sd = 2
    idx = 0
    while idx < len(samplePoints):
        n = samplePoints[idx][0]
        e = samplePoints[idx][1]
        a = samplePoints[idx][2]
        if (grid[n-sd:n+sd, e-sd:e+sd] + safety_distance > a).any():
            samplePoints.pop(idx)
            continue
        idx += 1

    # add the grid start and goal to the list of sample points
    samplePoints.append(grid_start)
    samplePoints.append(grid_goal)

    # construct a navigable graph using the sample points
    g = nx.Graph()   
    tree = KDTree(samplePoints)
    for samplePt in samplePoints:
        idxs = tree.query([samplePt], k=12, return_distance=False)[0]
        for i in idxs:
            if samplePt == samplePoints[i]:
                continue
            
            if can_connect_3d(grid, samplePt, samplePoints[i], safety_distance):
                dist = np.linalg.norm(np.array(samplePt) - np.array(samplePoints[i]))
                g.add_edge(samplePt, samplePoints[i], weight=dist)

    if g.number_of_nodes() == 0:
        return False, None

    foundPath2Goal, path, _ = graph_a_star(g, heuristic, grid_start, grid_goal, True)

    if foundPath2Goal:
        # don't return grid start and grid_goal. 
        # grid_start is already in the path. We will be adding grid_goal later
        return foundPath2Goal, path[1:-1]
    else:
        return foundPath2Goal, None

