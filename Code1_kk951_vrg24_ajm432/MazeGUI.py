import pygame, sys, random, numpy, math, time
from collections import deque
import Strategy_1 as s1
import Strategy2 as s2
import Strategy_3 as s3

'''
This is the main code where we will run the visualizing for the various graph algorithms for dfs, bfs, a star
and this is where the calls to strategy 1, 2, 3 will happen.
'''

# Color Graphics used in the Maze Visualizer
BLACK = (0, 0, 0)
YELLOW = (255, 255, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
RED = (255, 0, 0)

'''

MazeGUI class

Description: Where the visualization of the mazing solving algorithms is placed including DFS, BFS and A star.

'''
class MazeGUI:
    x, y = 0, 0 # these are to draw the rectangle sizes to create the maze
    cell_size = 5 # determines the cell size to draw 
    dim = 10 # default value for the dim size
    tracking_obstacles = [] # tracks obstacles created in the build maze function
    tracking_array = [] # copy of tracking_obstacles
    fringe = [] # fringe utilized in dfs
    visited = [] # visited array utilized in dfs
    display = None # screen to draw the GUI

    # this is where the logic to build the maze is based to create based on a certain obstacle density
    def build_maze(self, screen, size, probability):
        self.dim = size
        self.display = screen
        self.fire_array = numpy.zeros((self.dim, self.dim))
        obstacle_num = 0  # See if the amount of obstacles required are 0 or not
        # if the maze area is 100 then there should be only 10 obstacles
        obstacles = (size*size)*probability
        # track where the obstacles are places so it doesn't double count
        tracking_array = numpy.zeros((size, size))
        dim_array = list(range(0, size))
        # iterate based on the amount of obstacles that are left, when there are no obstacles left then draw the maze
        while obstacles != 0:
            i = random.choice(dim_array)
            j = random.choice(dim_array)
            if i == 0 and j == 0:  # this is what we will define as a start node with yellow
                pass # cant place an obstacle in the starting
            elif i == size - 1 and j == size - 1: 
                pass # can't place an obstacle in the ending
            else:
                arr = [0, 1]  # these will represent random choices
                if random.choice(arr) == 0 and obstacles != 0 and tracking_array[i][j] == 0:
                    tracking_array[i][j] = 1
                    obstacles -= 1

        # draw the visual
        for k in range(0, size):
            self.x = 5
            self.y += 5
            for b in range(0, size):
                if k == 0 and b == 0:  # this is what we will define as a start node with yellow
                    cell = pygame.Rect(
                        self.x, self.y, self.cell_size, self.cell_size)
                    pygame.draw.rect(screen, YELLOW, cell)
                elif k == size-1 and b == size-1: # this will define the ending node (G)
                    cell = pygame.Rect(
                        self.x, self.y, self.cell_size, self.cell_size)
                    pygame.draw.rect(screen, GREEN, cell)
                elif tracking_array[k][b] == 1: # this will define an obstacle
                    cell = pygame.Rect(
                        self.x, self.y, self.cell_size, self.cell_size)
                    pygame.draw.rect(screen, BLACK, cell)
                else: # anything else draw the maze design
                    cell = pygame.Rect(
                        self.x, self.y, self.cell_size, self.cell_size)
                    pygame.draw.rect(screen, BLACK, cell, 1)

                self.x += 5

        pygame.display.update()
        self.tracking_obstacles = tracking_array # update tracking_obstacles the global maze
        self.tracking_array = tracking_array # update this one for DFS
        return self.tracking_obstacles

    # this is for dfs only because it creates a path to start from and a path to end
    def create_maze_dfs(self, screen, size, probability, start, ending):
        # variables to create the visual
        self.x = 0  # reset x on creation
        self.y = 0  # reset y on creation
        self.dim = size
        self.display = screen

        obstacle_num = 0  # See if the amount of obstacles required are 0 or not
        # if the maze area is 100 then there should be only 10 obstacles
        obstacles = (size*size)*probability
        # track where the obstacles are places so it doesn't double count
        tracking_array = numpy.zeros((size, size))
        dim_array = list(range(0, size))
        while obstacles != 0:
            i = random.choice(dim_array)
            j = random.choice(dim_array)
            if i == 0 and j == 0:  # this is what we will define as a start node with yellow
                pass
            elif i == size - 1 and j == size - 1:
                pass
            else:
                arr = [0, 1]  # these will represent random choices
                if random.choice(arr) == 0 and obstacles != 0 and tracking_array[i][j] == 0:
                    tracking_array[i][j] = 1
                    obstacles -= 1

        # draw positions in the maze
        for k in range(0, size):
            self.x = 5
            self.y += 5
            for b in range(0, size):
                # this is what we will define as a start node with yellow
                if k == start[0] and b == start[1]:
                    cell = pygame.Rect(
                        self.x, self.y, self.cell_size, self.cell_size)
                    pygame.draw.rect(screen, YELLOW, cell)
                elif k == ending[0] and b == ending[1]: # ending node
                    cell = pygame.Rect(
                        self.x, self.y, self.cell_size, self.cell_size)
                    pygame.draw.rect(screen, GREEN, cell)
                elif tracking_array[k][b] == 1: # obstacles
                    cell = pygame.Rect(
                        self.x, self.y, self.cell_size, self.cell_size)
                    pygame.draw.rect(screen, BLACK, cell)
                else: # cell design
                    cell = pygame.Rect(
                        self.x, self.y, self.cell_size, self.cell_size)
                    pygame.draw.rect(screen, BLACK, cell, 1)
                self.x += 5
        pygame.display.update()

        self.tracking_obstacles = tracking_array # update global representation
        self.tracking_array = tracking_array # copy
        return self.tracking_obstacles

    def distance_calculator(self, start):  # calculates the eucledian distance between current point and goal
        end = len(self.tracking_obstacles) - 1
        x_diff = abs(start[0] - end)
        y_diff = abs(start[1] - end)
        return math.sqrt(x_diff**2 + y_diff**2)

    # puts the new child in the priority queue depending on it's cost to get there and distance to goal
    def sorting(self, fringe, child, cost):
        return_array = []
        child_dist = self.distance_calculator(child[0])

        if len(fringe) == 0:
            fringe.append(child[0])
            return fringe

        # iterates the fringe to find the smallest child and sort based off that for the future states
        for i in range(0, len(fringe)):
            curr_child_dist = self.distance_calculator(fringe[i])
            if child_dist + cost[child[0][0]][child[0][1]] <= curr_child_dist + cost[fringe[i][0]][fringe[i][1]] and child[0] not in return_array:
                return_array.append(child[0])
                return_array.append(fringe[i])
                i += 2
            elif i == len(fringe) - 1 and child[0] not in return_array:
                return_array.append(fringe[i])
                return_array.append(child[0])
            else:
                return_array.append(fringe[i])

        return return_array

    def a_star(self):  # A* algo
        maze_array = self.tracking_obstacles
        fringe = []  # priority queue
        visited = [[-1, -1, -1]]   # keeps track of all the visited cells
        child1 = []
        child2 = []
        child3 = []
        child4 = []
        n = len(maze_array)
        start = [0, 0]
        cost = numpy.zeros([n, n])
        goal = [n - 1, n - 1]
        tracker = []  # array for final path
        fringe.append(start)
        # 3 top, 4 right, 1 bottom, 2 left - to keep track of the parent of each node
        parent = numpy.zeros([n, n])
        while len(fringe) > 0:
            current = fringe.pop(0)  # take out the child with highest priority
            if len(child1) != 0:  # empty the child arrays
                child1.pop()
            if len(child2) != 0:
                child2.pop()
            if len(child3) != 0:
                child3.pop()
            if len(child4) != 0:
                child4.pop()
            if current not in visited:  # only continue if the current node is not visited before
                if not fringe:  # if the fringe is empty it does not check for child in fringe
                    # checks if it's not the top row and that there is not an obstacle on top of it
                    if current[0] != 0 and maze_array[current[0] - 1][current[1]] != 1:
                        child1.append([current[0] - 1, current[1]])  # top cell
                        if child1[0] not in visited and cost[current[0] - 1][[current[1]]] >= cost[current[0]][  # if the child is not visited
                                [current[1]]] + 1 or cost[current[0] - 1][[current[1]]] == 0:  # and the path it took is shorter than before or it's the firt time getting there
                            # then add it to the fringe with priority queue
                            cost[current[0] - 1][[current[1]]
                                                 ] = cost[current[0]][[current[1]]] + 1
                            fringe = self.sorting(fringe, child1, cost)
                            # set its parent to the top cell
                            parent[current[0] - 1][[current[1]]] = 3
                    if current[1] != n - 1 and maze_array[current[0]][current[1] + 1] != 1:
                        # right cell
                        child2.append([current[0], current[1] + 1])
                        if child2[0] not in visited and cost[current[0]][[current[1] + 1]] >= cost[current[0]][
                                [current[1]]] + 1 or cost[current[0]][[current[1] + 1]] == 0:
                            cost[current[0]][[current[1] + 1]
                                             ] = cost[current[0]][[current[1]]] + 1
                            fringe = self.sorting(fringe, child2, cost)
                            parent[current[0]][[current[1] + 1]] = 2
                    if current[0] != n - 1 and maze_array[current[0] + 1][current[1]] != 1:
                        # bottom cell
                        child3.append([current[0] + 1, current[1]])
                        if child3[0] not in visited and cost[current[0] + 1][[current[1]]] >= cost[current[0]][
                                [current[1]]] + 1 or cost[current[0] + 1][[current[1]]] == 0:
                            cost[current[0] + 1][[current[1]]
                                                 ] = cost[current[0]][[current[1]]] + 1
                            fringe = self.sorting(fringe, child3, cost)
                            parent[current[0] + 1][[current[1]]] = 1
                    if current[1] != 0 and maze_array[current[0]][current[1] - 1] != 1:
                        # left cell
                        child4.append([current[0], current[1] - 1])
                        if child4[0] not in visited and cost[current[0]][[current[1] - 1]] >= cost[current[0]][
                                [current[1]]] + 1 or cost[current[0]][[current[1] - 1]] == 0:
                            cost[current[0]][[current[1] - 1]
                                             ] = cost[current[0]][[current[1]]] + 1
                            fringe = self.sorting(fringe, child4, cost)
                            parent[current[0]][[current[1] - 1]] = 4
                else:
                    if current not in fringe:  # if current is not in fringe we go through its neighbours
                        if current[0] != 0 and maze_array[current[0] - 1][current[1]] != 1:
                            child1.append(
                                [current[0] - 1, current[1]])  # top cell
                            if child1[0] not in visited and child1[0] not in fringe and cost[current[0] - 1][
                                    # if the child is not visited before and not in the fringe and there is no obstacle there
                                    [current[1]]] >= cost[current[0]][[current[1]]] + 1 or cost[current[0] - 1][
                                    [current[1]]] == 0:  # if the cost to get there is less than before or it's 0 - meaning it's first time getting there
                                # set cost to get to child: cost ot get to current + 1
                                cost[current[0] - 1][[current[1]]
                                                     ] = cost[current[0]][[current[1]]] + 1
                                # add child to the fringe - priority queue
                                fringe = self.sorting(fringe, child1, cost)
                                # set it's parent to top cell
                                parent[current[0] - 1][[current[1]]] = 3
                        if current[1] != n - 1 and maze_array[current[0]][current[1] + 1] != 1:
                            # right cell
                            child2.append([current[0], current[1] + 1])
                            if child2[0] not in visited and child2[0] not in fringe and cost[current[0]][
                                    [current[1] + 1]] >= cost[current[0]][[current[1]]] + 1 or cost[current[0]][
                                    [current[1] + 1]] == 0:
                                cost[current[0]][[current[1] + 1]
                                                 ] = cost[current[0]][[current[1]]] + 1
                                fringe = self.sorting(fringe, child2, cost)
                                parent[current[0]][[current[1] + 1]] = 2
                        if current[0] != n - 1 and maze_array[current[0] + 1][current[1]] != 1:
                            # bottom cell
                            child3.append([current[0] + 1, current[1]])
                            if child3[0] not in visited and child3[0] not in fringe and cost[current[0] + 1][
                                    [current[1]]] >= cost[current[0]][[current[1]]] + 1 or cost[current[0] + 1][
                                    [current[1]]] == 0:
                                cost[current[0] + 1][[current[1]]
                                                     ] = cost[current[0]][[current[1]]] + 1
                                fringe = self.sorting(fringe, child3, cost)
                                parent[current[0] + 1][[current[1]]] = 1
                        if current[1] != 0 and maze_array[current[0]][current[1] - 1] != 1:
                            # left cell
                            child4.append([current[0], current[1] - 1])
                            if child4[0] not in visited and child4[0] not in fringe and cost[current[0]][
                                    [current[1] - 1]] >= cost[current[0]][[current[1]]] + 1 or cost[current[0]][
                                    [current[1] - 1]] == 0:
                                cost[current[0]][[current[1] - 1]
                                                 ] = cost[current[0]][[current[1]]] + 1
                                fringe = self.sorting(fringe, child4, cost)
                                parent[current[0]][[current[1] - 1]] = 4
                    visited.append(current)

            if current == goal:  # takes the "parent" array and tracks back to the start using the cell value
                y = n - 1
                x = n - 1
                tracker.append([y, x])
                while True:
                    if parent[y][x] == 1:  # parent is top cell
                        tracker.append([y - 1, x])
                        y -= 1
                    elif parent[y][x] == 2:  # parent is right cell
                        tracker.append([y, x - 1])
                        x -= 1
                    elif parent[y][x] == 3:  # parent is bottom cell
                        tracker.append([y + 1, x])
                        y += 1
                    elif parent[y][x] == 4:  # parent is left cell
                        tracker.append([y, x + 1])
                        x += 1
                    if x == 0 and y == 0:  # when it reaches start it breaks out of the loop
                        break

                tracker.reverse()

                self.draw_path(tracker)  # draws the path
                return True

        return False

    # check if the bounds are valid for the given maze
    def check_valid_bounds(self, i, j, pop_value, arr):
        # arg i indicate direction +1 or -1 for up and down
        i = pop_value[0] + i
        # arg j indicates direction +1 or -1 for left and right
        j = pop_value[1] + j
        if i >= 0 and i < len(arr) and j >= 0 and j < len(arr):
            return True  # in bounds, return true
        else:
            return False  # not in bounds, return false

    def bfs_tree_search(self):
        arr = self.tracking_obstacles

        # now define the start and end node which in our case is the first indicies and the last indicies respectively
        start = (0, 0)
        goal = (len(arr) - 1, len(arr) - 1)

        # now because we are working with bfs, we know bfs calls for a fringe in the form of a queue because of the queue's policy (FIFO)
        fringe = deque()
        fringe.append(start)

        # keep an array to represent the visited arrays
        visited = numpy.zeros((len(arr), len(arr)), dtype=bool)

        # for this implementation of bfs we want to keep track of the parents to obtain the shortest path
        path = []

        # now iterate through the fringe to check for the path
        while len(fringe) > 0:
            current = fringe.popleft()
            visited[current[0]][current[1]] = True
            if current == goal:
                path.append(current)
                self.bfs_nodes = len(path)
                path.reverse()
                # now that we found the end node, let's perform a recursive backtracking algorithm to find the actual path
                bfs_route = []
                while path[0] != start:
                    new_curr = path.pop(0)
                    if not bfs_route:
                        bfs_route.append(new_curr)
                    # check if its a straight path up
                    elif new_curr[1] == bfs_route[len(bfs_route) - 1][1] + 1 and new_curr[0] == bfs_route[len(bfs_route) - 1][0]:
                        bfs_route.append(new_curr)
                    # check if its a straight path right
                    elif new_curr[1] == bfs_route[len(bfs_route) - 1][1] and new_curr[0] == bfs_route[len(bfs_route) - 1][0] + 1:
                        bfs_route.append(new_curr)
                    # check if its a straight path down
                    elif new_curr[1] == bfs_route[len(bfs_route) - 1][1] - 1 and new_curr[0] == bfs_route[len(bfs_route) - 1][0]:
                        bfs_route.append(new_curr)
                    # check if its a straight path left
                    elif new_curr[1] == bfs_route[len(bfs_route) - 1][1] and new_curr[0] == bfs_route[len(bfs_route) - 1][0] - 1:
                        bfs_route.append(new_curr)

                # append the last node because that won't be included (we check until the starting node)
                bfs_route.append(start)
                bfs_route.reverse()
                # once the final path is recieved, draw it for the visualization
                self.draw_path(bfs_route)
                return bfs_route

            else:
                # first check the up direction
                if self.check_valid_bounds(-1, 0, current, arr) and arr[current[0] - 1][current[1]] == 0 and visited[current[0] - 1][current[1]] == False and (current[0] - 1, current[1]) not in fringe:
                    fringe.append((current[0] - 1, current[1]))
                    if current not in path:  # only append the parent if its not seen in the path already
                        path.append(current)

                # now check the down direction
                if self.check_valid_bounds(1, 0, current, arr) and arr[current[0] + 1][current[1]] == 0 and visited[current[0] + 1][current[1]] == False and (current[0] + 1, current[1]) not in fringe:
                    fringe.append((current[0] + 1, current[1]))
                    if current not in path:  # only append the parent if its not seen in the path already
                        path.append(current)

                # now we can check the left direction
                if self.check_valid_bounds(0, -1, current, arr) and arr[current[0]][current[1] - 1] == 0 and visited[current[0]][current[1] - 1] == False and (current[0], current[1] - 1) not in fringe:
                    fringe.append((current[0], current[1] - 1))
                    if current not in path:  # only append the parent if its not seen in the path already
                        path.append(current)

                # finally check the right side
                if self.check_valid_bounds(0, 1, current, arr) and arr[current[0]][current[1] + 1] == 0 and visited[current[0]][current[1] + 1] == False and (current[0], current[1] + 1) not in fringe:
                    fringe.append((current[0], current[1] + 1))
                    if current not in path:  # only append the parent if its not seen in the path already
                        path.append(current)

        # for no given path return [] indicating a blocked route or no path
        return []

    def dfs(self, beginning, goal):

        #checks whether either the goal or beginning points are blocked, if so return false
        if self.tracking_array[int(beginning[0])][int(beginning[1])] == 1 or self.tracking_array[goal[0]][goal[1]] == 1:
            return False

        #If they are the same point then return true
        if beginning == goal:
            return True

        #If not false, then add the beginning point to the fringe
        self.fringe.append((int(beginning[0]), int(beginning[1])))

        #loops through the fringe
        while len(self.fringe) > 0:

            #sets current to the topmost element of the fringe
            current = self.fringe.pop()

            #Terminating case in which current is equal to the goal
            if current == (goal[0], goal[1]):
                return True

            #Current not equal to goal
            else:
                #current has not been explored yet (haven't added surrounding valid children to the fringe)
                if current not in self.visited:
                    #All columns other than the first column
                    if current[1] > 0:

                        #Checks validity of left child
                        if self.tracking_array[current[0]][current[1]-1] == 0 and (current[0], current[1]-1) not in self.fringe and (current[0], current[1]-1) not in self.visited:
                            # left child is valid
                            self.fringe.append((current[0], current[1]-1))

                        #Checks whether the row is not the last row and also validity of bottom child
                        if current[0] != self.dim-1 and self.tracking_array[current[0]+1][current[1]] == 0 and (current[0]+1, current[1]) not in self.fringe and (current[0]+1, current[1]) not in self.visited:
                            # bottom child is valid
                            self.fringe.append((current[0]+1, current[1]))

                        #Checks whether the column is not the last column and validity of right child
                        if current[1] != self.dim-1 and self.tracking_array[current[0]][current[1]+1] == 0 and (current[0], current[1]+1) not in self.fringe and (current[0], current[1]+1) not in self.visited:
                            # right child is valid
                            self.fringe.append((current[0], current[1]+1))

                        #Checks whether the row is not the first row and validity of top child
                        if current[0] != 0 and self.tracking_array[current[0]-1][current[1]] == 0 and (current[0]-1, current[1]) not in self.fringe and (current[0]-1, current[1]) not in self.visited:
                            # top child is valid
                            self.fringe.append((current[0]-1, current[1]))

                    #The first column
                    else:
                        #Checks whether the row is not the last row and also validity of bottom child
                        if current[0] != self.dim-1 and self.tracking_array[current[0]+1][current[1]] == 0 and (current[0]+1, current[1]) not in self.fringe and (current[0]+1, current[1]) not in self.visited:
                            # bottom child is valid
                            self.fringe.append((current[0]+1, current[1]))

                        #Checks validity of right child
                        if self.tracking_array[current[0]][current[1]+1] == 0 and (current[0], current[1]+1) not in self.fringe and (current[0], current[1]+1) not in self.visited:
                            # right child is valid
                            self.fringe.append((current[0], current[1]+1))

                        #Checks whether the row is not the first row and validity of top child
                        if current[0] != 0 and self.tracking_array[current[0]-1][current[1]] == 0 and (current[0]-1, current[1]) not in self.fringe and (current[0]-1, current[1]) not in self.visited:
                            # top child is valid
                            self.fringe.append((current[0]-1, current[1]))

                #Adds the current node to visited (all valid children have been added to the fringe)
                self.visited.append(current)

        #In the case that the fringe is empty and you could not find a path
        return False

    # this method draws the path given by bfs or a star
    def draw_path(self, arr):  # arr contains the coordinates of the path to draw
        self.x = 0  # reset x for drawing
        self.y = 0  # reset y for drawing
        screen = self.display
        size = self.dim

        # use the global tracking obstacles to draw
        tracking_array = self.tracking_obstacles
        curr = None  # this is where we will pop one element at a time for the array

        # pop one element at a time and store that element inside tracking array to draw the path
        for i in range(0, len(tracking_array)):
            for j in range(0, len(tracking_array)):
                if len(arr) > 0:
                    curr = arr.pop(0)
                tracking_array[curr[0]][curr[1]] = 2

        # same mechanism as in drawing the maze but now it just includes drawing the full path given by one of the algos used
        for k in range(0, size):
            self.x = 5
            self.y += 5
            for b in range(0, size):
                if k == 0 and b == 0:  # this is what we will define as a start node with yellow
                    cell = pygame.Rect(
                        self.x, self.y, self.cell_size, self.cell_size)
                    pygame.draw.rect(screen, YELLOW, cell)
                elif k == size - 1 and b == size - 1:  # this is what we will define as a ending node with green
                    cell = pygame.Rect(
                        self.x, self.y, self.cell_size, self.cell_size)
                    pygame.draw.rect(screen, GREEN, cell)
                # this is the cell we will define to be an obstacle
                elif tracking_array[k][b] == 1:
                    cell = pygame.Rect(
                        self.x, self.y, self.cell_size, self.cell_size)
                    pygame.draw.rect(screen, BLACK, cell)
                # these are the cells that correspond to the path
                elif tracking_array[k][b] == 2:
                    cell = pygame.Rect(
                        self.x, self.y, self.cell_size, self.cell_size)
                    pygame.draw.rect(screen, BLUE, cell)
                else:
                    cell = pygame.Rect(
                        self.x, self.y, self.cell_size, self.cell_size)
                    pygame.draw.rect(screen, BLACK, cell, 1)
                pygame.display.update()
                self.x += 5

# this is how we will start the MazeGUI visualization
def start():
    # inital conditions to start pygame
    pygame.init()
    pygame.mixer.init()
    screen = pygame.display.set_mode((700, 700))
    screen.fill('white')
    pygame.display.set_caption("Python Maze Generator")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont('Comic Sans MS', 30)

    # this is where the main logic will be setup
    maze = MazeGUI()  # we need this to start the GUI up

    # now based on the user's request we will run the specific algorithm if they want to run DFS, BFS, or A Star
    if sys.argv[3] == "bfs":  # runs bfs
        maze.build_maze(screen, int(sys.argv[1]), float(
            sys.argv[2]))  # start off with building the maze
        maze.bfs_tree_search()  # execute bfs
    elif sys.argv[3] == 'a_star':  # runs a star
        maze.build_maze(screen, int(sys.argv[1]), float(
            sys.argv[2]))  # start off with building the maze
        maze.a_star()  # execute a star
    # if the user command isn't bfs or a star, it will automatically run dfs
    elif sys.argv[3] == 'dfs':
        beginning = input("Enter a start node in the form of x,y: ")
        Position1 = beginning.split(",")
        B_T = (int(Position1[0]), int(Position1[1]))
        goal = input("Enter a final node in the form of x,y: ")
        Position2 = goal.split(",")
        E_T = (int(Position2[0]), int(Position2[1]))
        maze.create_maze_dfs(screen, int(
            sys.argv[1]), float(sys.argv[2]), B_T, E_T)
        # prints true or false if there is a given path in the console
        print(maze.dfs(B_T, E_T))

    # if we reach this point this means that the third argument is a strategy we are running s1, s2, or s3
    if sys.argv[3] == 's1':  # run s1
        s1.start()
    elif sys.argv[3] == 's2':  # run s2
        s2.start()
    elif sys.argv[3] == 's3':  # otherwise run s3
        s3.start()

    # pygame variables in order to create the visualization and to run pygame in general
    running = True
    index = 0
    while running:
        clock.tick(60)
        events = pygame.event.get()
        for event in events:
            if event.type == pygame.QUIT:
                running = False

        # update pygame's display to display everything
        pygame.display.update()


# this is where the start method will be launched from
if __name__ == "__main__":

    # check the arguments given (has to start with atleast 4)
    if len(sys.argv[1]) >= 4:
        print("Incorrect Usage: Has to be either python MazeGUI.py <dimension size> <probability of generating an obstacle> <algorithm> or python MazeGUI.py <dimension size> <probability of generating an obstacle> <strategy number> <probability of generating fire>")
        exit(1)

    # else start
    start()
