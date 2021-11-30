import math
import random
import time
import numpy as np


start_time = 0
# readfile
def readFile(path, nodelist):
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
    coor_u = coor_dict[city_id_u]  # [x, y] of city u
    coor_v = coor_dict[city_id_v]  # [x, y] of city v

    diff = [x - y for x, y in zip(coor_u, coor_v)]
    diff = [x ** 2 for x in diff]
    dist = math.sqrt(sum(diff))
    return dist


# build the graph [city u, city v, distance]
def buildGraph(coor_dict):
    graph = []
    N = len(coor_dict)
    for i in range(1, N + 1):
        for j in range(i + 1, N + 1):
            distance = dist_between_cities(coor_dict, i, j)
            graph.append([i, j, distance])
    return graph


# the Three opt swap
def threeOptSwap(route, coor_dict, i, j, k):
    A, B, C, D, E, F = route[i - 1], route[i], route[j - 1], route[j], route[k - 1], route[k % len(route)]
    d0 = dist_between_cities(coor_dict, A, B) + dist_between_cities(coor_dict, C, D) + dist_between_cities(coor_dict, E, F)
    d1 = dist_between_cities(coor_dict, A, C) + dist_between_cities(coor_dict, B, D) + dist_between_cities(coor_dict, E, F)
    d2 = dist_between_cities(coor_dict, A, B) + dist_between_cities(coor_dict, C, E) + dist_between_cities(coor_dict, D, F)
    d3 = dist_between_cities(coor_dict, A, D) + dist_between_cities(coor_dict, E, B) + dist_between_cities(coor_dict, C, F)
    d4 = dist_between_cities(coor_dict, F, B) + dist_between_cities(coor_dict, C, D) + dist_between_cities(coor_dict, E, A)

    if d0 > d1:
        route[i:j] = reversed(route[i:j])
        return -d0 + d1
    elif d0 > d2:
        route[j:k] = reversed(route[j:k])
        return -d0 + d2
    elif d0 > d4:
        route[i:k] = reversed(route[i:k])
        return -d0 + d4
    elif d0 > d3:
        tmp = route[j:k] + route[i:j]
        route[i:k] = tmp
        return -d0 + d3
    return 0



# three_opt method
def three_opt(route, coor_dict, coor_graph):
    while True:
        delta = 0
        best_distance = float('inf')
        # Generate all segments combinations#
        all_segments = []
        for i in range(len(route)):
            for j in range(i + 2, len(route)):
                for k in range(j + 2, len(route) + (i > 0)):
                    all_segments.append((i, j, k))

        for (a, b, c) in all_segments:

            delta += threeOptSwap(route, coor_dict, a, b, c)
            cur_dist = calculateTotalDistance(route, coor_graph)
            if cur_dist < best_distance:
                best_distance = cur_dist
                with open('output', 'a+') as fp:
                    fp.write("{:.2f}, {}\n".format(
                        time.time() - start_time,
                        best_distance))
        if delta >= 0:
            break
    return route


# calculate the total distance for the route
def calculateTotalDistance(route, coor_graph):
    best_distance = 0
    routeLength = len(route)

    for i in range(routeLength - 1):
        j = i + 1
        if route[i] < route[j]:
            best_distance += coor_graph[(route[i], route[j])]
        else:
            best_distance += coor_graph[(route[j], route[i])]

    # the circle: add the last edge(from the last node to the first node)
    if route[0] < route[-1]:
        best_distance += coor_graph[(route[0], route[-1])]
    else:
        best_distance += coor_graph[(route[-1], route[0])]


    return best_distance


def main(path):
    # extract the data
    global start_time
    nodelist, dimension = [], 0
    nodelist, dimension = readFile(path, nodelist)

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
    print("For the city " + path + ":")
    print("The original route would be: " + str(route))

    start = time.time()
    threeopt = three_opt(route, coor_dict, coor_graph)
    delta = time.time() - start
    print("The final route would be: " + str(threeopt))
    print("The sorting time is " + str(delta))

    distance = calculateTotalDistance(threeopt, coor_graph)
    print("The best distance is " + str(distance))

import random
import argparse

if __name__ == "__main__":
    with open("output", "w+") as f:
        pass
    path = 'Atlanta.tsp'
    start_time = time.time()
    main(path)


def solve(args):
    main(args.inst)