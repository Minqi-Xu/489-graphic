import numpy as np
import random
import operator
import pandas as pd
import matplotlib.pyplot as plt
import tsplib95


class Vertex:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def dist(self, v):
        xDist = abs(self.x - v.x)
        yDist = abs(self.y - v.y)
        distance = np.sqrt(xDist * xDist + yDist * yDist)
        return distance

    def __repr__(self):
        return "(" + str(self.x) + "," + str(self.y) + ")"


class Fitness:
    def __init__(self, walk):
        self.walk = walk
        self.length = 0.0
        self.fitness = 0.0

    def walkLength(self):
        if self.length == 0:
            tempL = 0
            for i in range(len(self.walk)):
                u = self.walk[i]
                v = None
                if i + 1 < len(self.walk):
                    v = self.walk[i+1]
                else:
                    v = self.walk[0]
                tempL = tempL + u.dist(v)
            self.length = tempL
        return self.length

    def walkFitness(self):
        if self.fitness == 0:
            self.fitness = 1 / float(self.walkLength())
        return self.fitness


def createWalk(vList):
    walk = random.sample(vList, len(vList))
    return walk


def initializePopulation(size, vList):
    population = []
    for i in range(size):
        population.append(createWalk(vList))
    return population


def sortWalks(population):
    fitnesses = {}
    for i in range(len(population)):
        fitnesses[i] = Fitness(population[i]).walkFitness()
    return sorted(fitnesses.items(), key = operator.itemgetter(1), reverse = True)


def selectParents1(ranks, size):
    # This is Fitness proportionate selection approach
    result = []
    data = pd.DataFrame(np.array(ranks), columns=["index","fitness"])
    data['cum_sum'] = data.fitness.cumsum()
    data['prob'] = data.cum_sum/data.fitness.sum()

    for i in range(len(ranks)):
        if i < size:
            # for the first size, select the most fitness walks
            result.append(ranks[i][0])
        else:
            # for the rest ones, using roulette wheel selection
            rand_num = random.random()
            for j in range(len(ranks)):
                if rand_num <= data.iat[j,3]:
                    result.append(ranks[j][0])
                    break
    return result


def selectParents2(ranks, size):
    # This is Stochastic universal sampling approach
    result = []
    data = pd.DataFrame(np.array(ranks), columns=["index", "fitness"])
    data['cum_sum'] = data.fitness.cumsum()
    fitness_step = data.fitness.sum() / len(ranks)
    rand_num = random.random() * fitness_step

    current_ptr = rand_num
    pre_position = 0
    for index in range(len(ranks)):
        if index < size:
            result.append(ranks[index][0])
            continue
        for position in range(pre_position, len(ranks)):
            if data.cum_sum[position] >= current_ptr:
                result.append(ranks[position][0])
                pre_position = position
                break
        current_ptr = current_ptr + fitness_step
    return result


def pooling(population, selection):
    pool = []
    for i in range(len(selection)):
        index = selection[i]
        pool.append(population[index])
    return pool


def crossover1(p1, p2):
    # partially mapped crossover operator
    # but only half, copy the fragment from p1, and rest are with the order in p2
    start_v = p1[0]
    p1 = p1[1:len(p1)-1]
    p2 = p2[1:len(p2)-1]
    child = [None]*len(p1)
    o1 = []
    o2 = []

    walk_len = len(p1)
    cut_point1 = int(random.random() * walk_len)
    cut_point2 = int(random.random() * walk_len)

    while cut_point1 == cut_point2:
        cut_point2 = int(random.random() * walk_len)

    if cut_point1 > cut_point2:
        cut_point1, cut_point2 = cut_point2, cut_point1

    for i in range(cut_point1, cut_point2):
        o1.append(p1[i])

    o2 = [item for item in p2 if item not in o1]

    for i in range(len(p1)):
        if i < cut_point1:
            if p2[i] not in o1:
                child[i] = p2[i]
        elif i < cut_point2:
            child[i] = o1[i - cut_point1]
        else:
            if p2[i] not in o1:
                child[i] = p2[i]

    while None in child:
        ind = child.index(None)
        temp_ind = ind
        while p2[temp_ind] in child:
            temp_ind = p1.index(p2[temp_ind])
        child[ind] = p2[temp_ind]
    child.insert(0, start_v)
    child.append(start_v)
    return child


def crossover2(p1, p2):
    # cycle crossover
    child = [None] * len(p1)
    while None in child:
        ind = child.index(None)
        indices = []
        values = []
        while ind not in indices:
            val = p1[ind]
            indices.append(ind)
            values.append(val)
            ind = p1.index(p2[ind])
        for ind, val in zip(indices, values):
            child[ind] = val
        p1, p2 = p2, p1
    return child


def crossoverPopulation(pool, size, crossoverMethod):
    children = []
    length = len(pool) - size
    pool = random.sample(pool, len(pool))

    for i in range(size):
        children.append(pool[i])

    for i in range(length):
        if crossoverMethod == 1:
            child = crossover1(pool[i], pool[len(pool)-i-1])
        else:
            child = crossover2(pool[i], pool[len(pool)-i-1])
        children.append(child)
    return children


def mutate(walk, mutationRate):
    for i in range(len(walk)):
        if i == 0 or i == (len(walk)-1):
            continue
        if random.random() < mutationRate:
            swapWith = int(random.random() * len(walk))
            while swapWith == 0 or swapWith == (len(walk)-1):
                swapWith = int(random.random() * len(walk))
            walk[i], walk[swapWith] = walk[swapWith], walk[i]
    return walk


def mutatePopulation(population, mutationRate):
    result = []
    for i in range(len(population)):
        mutated = mutate(population[i], mutationRate)
        result.append(mutated)
    return result


def nextGeneration(currentGeneration, size, mutationRate, selectionMethod, crossoverMethod):
    rank = sortWalks(currentGeneration)
    if selectionMethod == 1:
        selection = selectParents1(rank, size)
    else:
        selection = selectParents2(rank, size)
    pool = pooling(currentGeneration, selection)
    children = crossoverPopulation(pool, size, crossoverMethod)
    nextGeneration = mutatePopulation(children, mutationRate)
    return nextGeneration


def evolve(vList, popSize, size, mutationRate, generations, selectionMethod, crossoverMethod):
    # vList: list of vertices
    # popSize: the size of population
    # size: elite size, the size that we want to guarantee the best from the previous generation
    # mutationRate: mutation rate should in [0,1]
    # generations: number of generations we want to do
    # selectionMethod: the selection method, should be 1 or 2
    #   1: fitness proportionate selection    2: Stochastic universal sampling
    # crossoverMethod: the crossover method, should be 1 or 2
    #   1: partially mapped crossover         2: cycle crossover
    population = initializePopulation(popSize, vList)
    progress = []
    progress.append(1 / sortWalks(population)[0][1])

    for i in range(generations):
        population = nextGeneration(population, size, mutationRate, selectionMethod, crossoverMethod)
        progress.append(1 / sortWalks(population)[0][1])
    if(selectionMethod == 1):
        sm = "Fitness Proportionate Selection"
    else:
        sm = "Stochastic Universal Sampling"
    if(crossoverMethod == 1):
        cm = "Partially Mapped Crossover"
    else:
        cm = "Cycle Crossover"
    title_str = sm + " with " + cm
    plt.title(title_str)
    plt.plot(progress)
    plt.ylabel('Distance')
    plt.xlabel('Generation')
    plt.show()


tsp = tsplib95.load('data/dj38.tsp')
#tsp = tsplib95.load('data/qa194.tsp')
length = len(list(tsp.get_nodes()))
vList = []
for i in range(1, length+1):
    vList.append(Vertex(tsp.node_coords[i][0], tsp.node_coords[i][1]))

evolve(vList=vList, popSize=100, size=20, mutationRate=0.01, generations=5000, selectionMethod=1, crossoverMethod=2)