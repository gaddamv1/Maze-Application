import pygame, sys, re, random, numpy, math
from pygame_widgets import Button, TextBox
from collections import deque, OrderedDict
import threading, time

# Color Graphics used in the Maze Visualizer
BLACK = (0, 0, 0)
YELLOW = (255, 255, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
RED = (255, 0, 0)
current1 = 0
dimensions = 0

class MazeGUI:
    x, y = 0, 0
    cell_size = 5

    dim = 10
    tracking_obstacles = []
    display = None
    fire_array = None

    # this is where the logic to build the maze is based to create based on a certain obstacle density (same logic as in for the static mazes)
    def build_maze(self, screen, size, probability):
        self.x = 0 # reset x upon creating the maze again
        self.y = 0 # reset y upon creating the maze again
        self.dim = size
        self.display = screen
        obstacle_num = 0  # See if the amount of obstacles required are 0 or not
        obstacles = (size*size)*probability  # if the maze area is 100 then there should be only 10 obstacles
        tracking_array = numpy.zeros((size, size))  # track where the obstacles are places so it doesn't double count
        dim_array = list(range(0, size))
        self.fire_array = numpy.zeros((size, size))
        # iterate based on the amount of obstacles that are left, when there are no obstacles left then draw the maze
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

        for k in range(0, size):
            self.x = 5
            self.y += 5
            for b in range(0, size):
                if k == 0 and b == 0:  # this is what we will define as a start node with yellow
                    cell = pygame.Rect(self.x, self.y, self.cell_size, self.cell_size)
                    pygame.draw.rect(screen, YELLOW, cell)
                elif k == size-1 and b == size-1:
                    cell = pygame.Rect(self.x, self.y, self.cell_size, self.cell_size)
                    pygame.draw.rect(screen, GREEN, cell)
                elif tracking_array[k][b] == 1:
                    cell = pygame.Rect(self.x, self.y, self.cell_size, self.cell_size)
                    pygame.draw.rect(screen, BLACK, cell)
                else:
                    cell = pygame.Rect(self.x, self.y, self.cell_size, self.cell_size)
                    pygame.draw.rect(screen, BLACK, cell, 1)
                self.x += 5

        pygame.display.update()

        self.tracking_obstacles = tracking_array
        return self.tracking_obstacles

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

    def generate_fire_maze1(self, probability, bln):  # bln is to check if it's the first fire
        q = probability
        fire_maze = self.tracking_obstacles  # the actual maze
        fire_array = self.fire_array  # array that keeps track of fire only in the maze
        fire_array_copy = numpy.zeros((len(fire_maze), len(fire_maze)))  # a copy of the fire_array to keep track of old fires, so the new ones are not counted when calculating probabilities
        for x in range(0, len(fire_maze)):
            for y in range(0, len(fire_maze)):
                fire_array_copy[x][y] = fire_array[x][y]
        if bln:  # if it's the first fire then we chose a stop randomly
            while bln:  # for the first one does a random fire
                y = random.randint(0, len(fire_maze) - 1)  # random x spot for fire
                x = random.randint(0, len(fire_maze) - 1)  # random y spot for fire
                if fire_maze[x][y] != 2 and fire_maze[x][y] != 1 and (x != 0 and y != 0) and (
                        x != len(fire_maze) - 1 and y != len(fire_maze) - 1):  # only generate fire if there is no obstacle there and it's not the start or goal
                    fire_array[x][y] = 2
                    self.tracking_obstacles[x][y] = 2
                    return self.tracking_obstacles  # return the maze array
        else:
            for i in range(0, len(self.tracking_obstacles)):  # if it's not the first time then we traverse through every cell
                for j in range(0, len(self.tracking_obstacles)):  # for each cell we calculate the probability fo it catching fire depending on how many of it's neighbours are on fire
                    fire = 0
                    if fire_maze[i][j] != 1 and fire_array[i][j] != 2:
                        if i != len(self.tracking_obstacles) - 1 and fire_array_copy[i + 1][j] == 2:  # we use the copy of fire array to make sure a new fire is not counted in the calculations
                            fire += 1  # bottom cell
                        if fire_array_copy[i - 1][j] == 2 and i != 0:
                            fire += 1  # top cell
                        if j != len(self.tracking_obstacles) - 1 and fire_array_copy[i][j + 1] == 2:
                            fire += 1  # right cell
                        if fire_array_copy[i][j - 1] == 2 and j != 0:
                            fire += 1  # left cell
                        prob = 1 - ((1 - q) ** fire)  # calculate the probability with given formula
                        if fire > 0 and random.random() <= prob and prob > 0:  # if it catches on fire
                            fire_array[i][j] = 2  # update the fire tracking array
                            self.tracking_obstacles[i][j] = 2  # update the actual maze array

        return self.tracking_obstacles

    # this is where strategy 1 wil start from
    def strategy1(self):
        self.generate_fire_maze1(float(sys.argv[4]), True)  # ignite the first fire in the maze
        time.sleep(1.5)
        path = self.bfs_tree_search()  # run a bfs search for the path from start to goal
        path.reverse()  # path is reversed for it to be used in the for loop
        dimension = len(self.tracking_obstacles) - 1
        x = len(path)

        if not path:  # if there is no path return false
            return False
        curr = path.pop()  # pop the start cell from the path array

        for i in range(0, x):  # the loop is run for every step of the path
            self.draw_path(curr)  # draw the current step on the maze
            curr = path.pop()  # replace the current cell with next step in path
            if self.tracking_obstacles[curr[0]][curr[1]] == 2:  # is the next step agent is going to take is already on fire we stop the code
                return False
            if curr[0] == dimension and curr[1] == dimension:  # if current is at the goal cell then we return true and agent has reached the goal
                return True
            self.generate_fire_maze1(float(sys.argv[4]), False)  # update the fire in maze
            time.sleep(.5)  # makes it easier to visualize how the fire and agent are moving in the maze

    # modified BFS for the purposes of strategy 1
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
                path.reverse()
                # now that we found the end node, let's perform a recursive backtracking algorithm to find the actual path
                bfs_route = []
                while path[0] != start:
                    new_curr = path.pop(0)
                    if not bfs_route:
                        bfs_route.append(new_curr)
                    # top
                    elif new_curr[1] == bfs_route[len(bfs_route) - 1][1] + 1 and new_curr[0] == \
                            bfs_route[len(bfs_route) - 1][0]:
                        bfs_route.append(new_curr)
                    # right
                    elif new_curr[1] == bfs_route[len(bfs_route) - 1][1] and new_curr[0] == \
                            bfs_route[len(bfs_route) - 1][0] + 1:
                        bfs_route.append(new_curr)
                    # bottom
                    elif new_curr[1] == bfs_route[len(bfs_route) - 1][1] - 1 and new_curr[0] == \
                            bfs_route[len(bfs_route) - 1][0]:
                        bfs_route.append(new_curr)
                    # left
                    elif new_curr[1] == bfs_route[len(bfs_route) - 1][1] and new_curr[0] == \
                            bfs_route[len(bfs_route) - 1][0] - 1:
                        bfs_route.append(new_curr)

                bfs_route.append(start)

                bfs_route.reverse()

                return bfs_route

            else:
                # first check the up direction
                if self.check_valid_bounds(-1, 0, current, arr) and arr[current[0] - 1][current[1]] == 0 and \
                        visited[current[0] - 1][current[1]] == False and (current[0] - 1, current[1]) not in fringe:
                    fringe.append((current[0] - 1, current[1]))
                    if current not in path:
                        path.append(current)

                # now check the down direction
                if self.check_valid_bounds(1, 0, current, arr) and arr[current[0] + 1][current[1]] == 0 and \
                        visited[current[0] + 1][current[1]] == False and (current[0] + 1, current[1]) not in fringe:
                    fringe.append((current[0] + 1, current[1]))
                    if current not in path:
                        path.append(current)

                # now we can check the left direction
                if self.check_valid_bounds(0, -1, current, arr) and arr[current[0]][current[1] - 1] == 0 and \
                        visited[current[0]][current[1] - 1] == False and (current[0], current[1] - 1) not in fringe:
                    fringe.append((current[0], current[1] - 1))
                    if current not in path:
                        path.append(current)

                # finally check the right side
                if self.check_valid_bounds(0, 1, current, arr) and arr[current[0]][current[1] + 1] == 0 and \
                        visited[current[0]][current[1] + 1] == False and (current[0], current[1] + 1) not in fringe:
                    fringe.append((current[0], current[1] + 1))
                    if current not in path:
                        path.append(current)
        return []

    # function to draw the path one spot at a time
    def draw_path(self, position):  # arr contains the coordinates of the path to draw
        self.x = 0
        self.y = 0
        screen = self.display
        size = self.dim
        tracking_array = self.tracking_obstacles

        tracking_array[position[0]][position[1]] = 3

        for k in range(0, size):
            self.x = 5
            self.y += 5
            for b in range(0, size):
                if k == 0 and b == 0:  # this is what we will define as a start node with yellow
                    cell = pygame.Rect(
                        self.x, self.y, self.cell_size, self.cell_size)
                    pygame.draw.rect(screen, YELLOW, cell)
                elif k == size - 1 and b == size - 1:
                    cell = pygame.Rect(
                        self.x, self.y, self.cell_size, self.cell_size)
                    pygame.draw.rect(screen, GREEN, cell)
                elif tracking_array[k][b] == 1:
                    cell = pygame.Rect(
                        self.x, self.y, self.cell_size, self.cell_size)
                    pygame.draw.rect(screen, BLACK, cell)
                elif tracking_array[k][b] == 2:
                    cell = pygame.Rect(
                        self.x, self.y, self.cell_size, self.cell_size)
                    pygame.draw.rect(screen, RED, cell)
                elif tracking_array[k][b] == 3:
                    cell = pygame.Rect(
                        self.x, self.y, self.cell_size, self.cell_size)
                    pygame.draw.rect(screen, BLUE, cell)
                else:
                    cell = pygame.Rect(
                        self.x, self.y, self.cell_size, self.cell_size)
                    pygame.draw.rect(screen, BLACK, cell, 1)
                pygame.display.update()
                self.x += 5

# this is where the the strategy will start from
def start():

    # command line arguments
    dim = int(sys.argv[1])
    probability = float(sys.argv[2])

    # inital conditions to start pygame
    pygame.init()
    pygame.mixer.init()
    screen = pygame.display.set_mode((700, 700))
    screen.fill('white')
    pygame.display.set_caption("Python Maze Generator")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont('Comic Sans MS', 30)

    # this is the class that will assist in starting the maze
    maze = MazeGUI()

    # first build the starting maze
    maze.build_maze(screen, dim, probability)
    print(maze.strategy1()) # run strategy 1

    # here are some extra factors that pygame needs in order to run properly
    running = True
    while running:
        clock.tick(60)
        events = pygame.event.get()
        for event in events:
            if event.type == pygame.QUIT:
                running = False

        # update pygame's display to display everything
        pygame.display.update()

# main method to start the program
if __name__ == "__main__":
    start()
