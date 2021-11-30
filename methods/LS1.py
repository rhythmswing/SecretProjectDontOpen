import math
import random
import time
import numpy as np


start_time = 0

# readfile
def readFile(path, nodelist, dimension):
    # Open input file
    infile = open(path, 'r')

    # Read instance header
    Name = infile.readline().strip().split()[1]  # NAME
    Comment = infile.readline().strip().split()[1]  # COMMENT
    dimension = int(infile.readline().strip().split()[1])  # DIMENSION
    EdgeWeightType = infile.readline().strip().split()[1]  # EDGE_WEIGHT_TYPE
    infile.readline()

    # Read node list
    N = int(dimension)
    for i in range(0, int(dimension)):
        line = infile.readline().strip().split()
        id = int(line[0])
        x, y = [float(x) for x in line[1:]]
        nodelist.append([id, x, y])

    # Close input file
    infile.close()

    return nodelist, dimension


# calculate the distance from two cities
def dist_between_cities(coor_dict, city_id_u, city_id_v):
    coor_u = coor_dict[city_id_u] # [x, y] of city u
    coor_v = coor_dict[city_id_v] # [x, y] of city v

    diff = [x-y for x,y in zip(coor_u, coor_v)]
    diff = [x**2 for x in diff]
    dist = math.sqrt(sum(diff))
    return dist


# build the graph [city u, city v, distance]
def buildGraph(coor_dict):
    graph = []
    N = len(coor_dict)
    for i in range(1, N+1):
        for j in range(i+1, N+1):
            distance = dist_between_cities(coor_dict,i,j)
            graph.append([i, j, distance])
    return graph


# calculate the total distance for the route
def calculateTotalDistance(route, coor_graph):
    best_distance = 0
    routeLength = len(route)

    for i in range(routeLength - 1):
        j = i + 1
        if (route[i] < route[j]):
            best_distance += coor_graph[(route[i], route[j])]
        else:
            best_distance += coor_graph[(route[j], route[i])]

    # the circle: add the last edge(from the last node to the first node)
    if (route[0] < route[-1]):
        best_distance += coor_graph[(route[0], route[-1])]
    else:
        best_distance += coor_graph[(route[-1], route[0])]

    return best_distance


# the two opt swap
def twoOptSwap(route, i, k):
    if (i == 0):
        new_route =  route[k::-1] + route[k+1:]
    else:
        new_route = route[:i] + route[k:i-1:-1] + route[k+1:]
    return new_route


# implement the twoOpt method to find the optimal distance
def twoOpt(route, graph):
    global start_time
    routeLength = len(route)
    # compute the total distance of exsiting route
    best_distance = calculateTotalDistance(route, graph)
    best_route = route
    improved = True # Optimization

    # repeat until no improvement is made
    while improved:
        improved = False
        for i in range(routeLength-1):
            for k in range(i+1, routeLength):
                if k - i == 1: continue
                new_route = twoOptSwap(route, i ,k)
                new_distance = calculateTotalDistance(new_route, graph)
                if (new_distance < best_distance):
                    with open('output', 'a+') as fp:
                        fp.write("{:.2f}, {}\n".format(
                            time.time() - start_time,
                            best_distance))
                    best_route = new_route
                    best_distance = new_distance
                    improved = True
        route = best_route

    return best_distance


# define the main function
def main(path):
    start_time = time.time()
    # extract the data
    nodelist, dimension = [], 0
    nodelist, dimension = readFile(path, nodelist, dimension)

    # build the dictionary of the coordinate
    coor_dict = {}
    for city in nodelist:
        city_id = city[0]
        x = city[1]
        y = city[2]
        coor_dict[city_id] = [x, y]

    # build the list of graph
    graph = buildGraph(coor_dict)

    # buile the dictionary of the graph
    coor_graph = {}
    for edge in graph:
        city_u = edge[0]
        city_v = edge[1]
        distance = edge[2]
        coor_graph[(city_u, city_v)] = distance

    # randomly generate the initial route
    route = [i for i in range(1, dimension + 1)]
    random.shuffle(route)

    # the total running time of the two opt mthod of local search
    res = twoOpt(route, coor_graph)
    delta = time.time() - start_time

    print("For the city "+ path +":")
    print("The sorting time is " + str(delta))
    print("The best distance is " + str(res))

if __name__ == "__main__":
    with open("output", "w+") as f:
        pass
    path = 'Atlanta.tsp'
    start_time = time.time()
    main(path)

def solve(args):
    main(args.inst)










