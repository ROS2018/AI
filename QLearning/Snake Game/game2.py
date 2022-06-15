import pygame, sys, random, time
from pygame.math import Vector2 as vc
import numpy as np
import torch



# initiation:
cell_size = 40
width = 6
height = width

pygame.init()
screen = None
pygame.display.set_caption('Snake')
clock = pygame.time.Clock()  # to control the speed of displaying
apple = pygame.image.load('apple.png')  #
game_font = pygame.font.Font(None, 45)



# My Classes and Functions:

class FRUIT:
    def __init__(self):
        self.pos = vc(random.randint(0, width - 1), random.randint(0, height - 1))

    def draw(self):
        # creat rectangle
        fruit_rect = pygame.Rect(self.pos.x * cell_size, self.pos.y * cell_size, cell_size, cell_size)
        screen.blit(apple, fruit_rect)
        # pygame.draw.rect(screen, (250, 150, 70), fruit_rect)

    def update_position(self):
        self.pos = vc(random.randint(0, width - 1), random.randint(0, height - 1))


class SNAKE:
    def __init__(self):
        y, x =  int(height/2), int(width/2)
        self.body = [vc(x, y), vc(x+1, y), vc(x+2, y)]
        self.direction = vc(-1, 0) #random.choice([vc(-1, 0), vc(1, 0), vc(0, -1), vc(0, 1)])
        self.score = 0

    def draw(self):
        # border_bottom_right_radius=15)
        for block in self.body[1:]:
            rect = pygame.Rect(block.x * cell_size, block.y * cell_size, cell_size, cell_size)
            pygame.draw.rect(screen, (0, 250, 0), rect)
        pygame.draw.rect(screen, (0 + 255, 250 * 0, 0),
                         [self.body[0].x * cell_size, self.body[0].y * cell_size, cell_size, cell_size], 0,
                         border_radius=10)  # , border_top_left_radius=0,

    def move(self):
        body_copy = self.body[:-1]
        body_copy.insert(0, body_copy[0] + self.direction)
        self.body = body_copy[:]


    def munch(self, fruit_pos):
        self.body.insert(0, fruit_pos)
        self.score += 1

    def initialize(self):
        y, x = int(height/2), int(width/2)
        self.body = [vc(x, y), vc(x+1, y), vc(x+2, y)]
        self.direction = vc(-1, 0)
        self.score = 0


class ENVIRONMENT:
    def __init__(self):
        # Define Environment components:
        self.fruit = FRUIT()
        self.snake = SNAKE()
        self.num_actions = 3
        self.state_len = width*height
        # update state:
        self.state = self.compute_state()

    def get_state(self):
        return self.state # normalization

    def update(self):
        reward = torch.tensor([1])
        done = torch.tensor([0])
        self.snake.move()
        # if there is no collision
        if self.check_collision(): # i.e of done
            #print('Game Over !!!')
            self.snake.initialize()
            self.fruit.update_position()
            reward = torch.tensor([-100])
            done = torch.tensor([1])
        # check if the fruit is reached
        if self.fruit.pos == self.snake.body[0]:
            # update the snake length
            self.snake.munch(self.fruit.pos)
            # randomize the fruit again
            self.fruit.update_position()
            reward = torch.tensor([100])
        self.state = self.compute_state()
        return self.state, reward, done, self.snake.score

    def compute_state(self):
        state = torch.zeros((1,width, height))
        for block in self.snake.body[1:]:
           # print(self.snake.body)
            state[0,int(block.x),int(block.y)] = .2
        state[0,int(self.snake.body[0].x),int(self.snake.body[0].y)] = .5 # add the head
        state[0, int(self.fruit.pos.x), int(self.fruit.pos.y)] = 1
        return state

    def draw(self):
        self.fruit.draw()
        self.snake.draw()
        self.draw_score()

    def change_direction(self, value):  # value = 0: right , 1: up, 2: left, 3: down
        if value == 0:  # right
            self.snake.direction = np.matmul(np.array([[0,1],[-1,0]]), self.snake.direction).tolist() # rotate the direction toward right
        if value == 1:  # left
            self.snake.direction = np.matmul( np.array([[0,-1],[1,0]]),self.snake.direction).tolist() # rotate the direction toward left
        #if value == 3:  # don nothing

        state, reward, done, score = self.update()
        return state, reward, done, score


    def check_collision(self):
        head = self.snake.body[0]
        self.done = head.x > width - 1 or head.x < 0 or head.y > height - 1 or head.y < 0 or head in self.snake.body[2:]
        return self.done

    def draw_score(self):
        text = str(self.snake.score)
        score_surface = game_font.render(text, True, (0, 0, 0))
        x = cell_size * (width - 2)
        y = cell_size
        rect = score_surface.get_rect(center=(x, y))
        screen.blit(score_surface, rect)

    def render(self):
        # EVENTS SECTION:
        global screen
        if screen == None:
            screen = pygame.display.set_mode((cell_size * width, cell_size * height))  # (screen_width, screen_height)
        for event in pygame.event.get():  # that happens each 150 ms
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        # DRAWING SECTION: happens each 1000/60= 16 ms
        screen.fill(pygame.Color('white'))  # (50,50,30)) #pygame.Color('green')
        self.draw()
        pygame.display.update()
        clock.tick(70)  # this is for the graphic update, the program will never run at more than 10 frames per second.

