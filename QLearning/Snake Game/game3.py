import pygame, sys, random, time
from pygame.math import Vector2 as vc
import numpy as np
import torch



# initiation:
cell_size = 40
width = 5
height = width

pygame.init()
screen = None
pygame.display.set_caption('Snake')
clock = pygame.time.Clock()  # to control the speed of displaying
apple = pygame.image.load('apple.png')  #
game_font = pygame.font.Font(None, 45)



# My Classes and Functions:

class FRUIT:
    def __init__(self):#, Transist = False):
        #
        # if Transist == False:
        #     random.setstate(self.prev_random_state)# if you dont transit then set the previous random.sate
        #
        # self.prev_random_state = random.getstate()  # save the taken random.state. (the main loop cannot begain with transist = False)
        self.pos = vc(random.randint(0, width - 1), random.randint(0, height - 1))
        #self.pos = vc(random.choice([0,2]), 0)

    def draw(self):
        # creat rectangle
        fruit_rect = pygame.Rect(self.pos.x * cell_size, self.pos.y * cell_size, cell_size, cell_size)
        screen.blit(apple, fruit_rect)
        # pygame.draw.rect(screen, (250, 150, 70), fruit_rect)


class SNAKE:
    def __init__(self):
        y, x =  int(height/2), int(width/2)
        self.body = [vc(x, y)]
        #head_dir = random.choice([ (vc(-1, 0) , 0.2), (vc(1, 0),0.4), (vc(0, -1),0.6), ( vc(0, 1), 0.8) ])
        #self.direction = head_dir[0]
        #self.head = head_dir[1]
        self.direction = vc(-1, 0)
        self.head = 0.2
        self.score = 0
        self.steps = 0

    def draw(self):
        x = self.body[0].x
        y = self.body[0].y
        e = cell_size
        if self.direction == vc(-1,0):
            a = [x*e, (y + 0.5)*e]
            b = [(x + 1)*e, (y + 1)*e]
            c = [(x + 1)*e, y*e]
        elif self.direction == vc(0,1):
            a = [(x + 0.5)*e, (y + 1)*e]
            b = [(x + 1)*e, y*e]
            c = [x*e, y*e]
        elif self.direction == vc(1, 0) :
            a = [(x + 1)*e, (y + 0.5)*e]
            b = [x*e, y*e]
            c = [x*e, (y + 1)*e]
        elif self.direction == vc(0, -1) :
            a = [(x + 0.5)*e, y*e]
            b = [x*e, (y + 1)*e]
            c = [(x + 1)*e, (y + 1)*e]
        pygame.draw.rect(screen, (0, 0, 139),
                         [self.body[0].x * cell_size, self.body[0].y * cell_size, cell_size, cell_size], 0,
                         border_radius=10)  # , border_top_left_radius=0,
        pygame.draw.polygon(screen, (255,255,255), [a, b, c], 3)

    def code_direction(self):
        if self.direction == vc(-1, 0):
            self.head = 0.2
        elif self.direction == vc(0, 1):
            self.head = 0.4
        elif self.direction == vc(0, -1):
            self.head = 0.6
        elif self.direction == vc(1, 0):
            self.head = 0.8


    def move(self):
        self.body = [self.body[0] + self.direction]
        self.steps += 1
    #    self.done = self.steps < 500


    def munch(self, fruit_pos):
        #self.body.insert(0, fruit_pos)
        self.score += 1
        self.steps = 0


class ENVIRONMENT:
    def __init__(self):
        # Define Environment components:
        self.fruit = FRUIT()#(rand_seed)
        self.snake = SNAKE()
        self.num_actions = 3
        self.state_len = width*height
        self.done = torch.tensor([0])
        # update state:
        self.state = self.compute_state()

    def get_state(self):
        return self.state

    def update(self):
        reward = torch.tensor([0])
        self.snake.move()
        # if there is no collision
        if self.check_collision().item(): # i.e of done
            #print('Game Over !!!')
            self.snake.__init__()
            self.fruit.__init__()
            reward = torch.tensor([-100])
            #self.done = torch.tensor([1])
        # check if the fruit is reached
        if self.fruit.pos == self.snake.body[0]:
            # Reward the snake:
            # reward = torch.tensor([100/( self.snake.steps + 1)])
            reward = torch.tensor([100 - self.snake.steps])
            # update the snake length
            self.snake.munch(self.fruit.pos)
            # randomize the fruit again
            self.fruit.__init__()
        # print('reward: ', reward.item())

        self.state = self.compute_state()
        return self.state, reward, self.done, self.snake.score

    def compute_state(self):
        # code the snake head direction to numbers:
        state = torch.zeros((1,width, height))
        for block in self.snake.body[1:]:
           # print(self.snake.body)
            state[0,int(block.x),int(block.y)] = .2
        state[0,int(self.snake.body[0].x),int(self.snake.body[0].y)] = self.snake.head # add the head
        state[0, int(self.fruit.pos.x), int(self.fruit.pos.y)] = 1
        return state

    # def change_direction(self, value):  # value = 0: right , 1: up, 2: left, 3: down  #### This version for one dimension map
    #     if value == 0:  # right
    #         self.snake.direction = vc(1,0)
    #     if value == 1:  # left
    #         self.snake.direction = vc(-1,0)
    #     #if value == 3:  # don nothing
    #
    #     state, reward, done, score = self.update()
    #     return state, reward, done, score


    # def change_direction(self, value):  # value = 0: right , 1: up, 2: left, 3: down  #### This version for two dimension map
    #     if value == 0:  # right
    #         self.snake.direction = vc(1,0)
    #     if value == 1:  # left
    #         self.snake.direction = vc(-1,0)
    #     if value == 2:  # up
    #         self.snake.direction = vc(0,-1)
    #     if value == 3:  # down
    #         self.snake.direction = vc(0,1)
    #     #if value == 4:  # don nothing
    #
    #     state, reward, done, score = self.update()
    #     return state, reward, done, score

    def change_direction(self, value):  # value = 0: right , 1: up, 2: left, 3: down
        if value == 0:  # right
            self.snake.direction = np.matmul(np.array([[0,1],[-1,0]]), self.snake.direction).tolist() # rotate the direction toward right
        if value == 1:  # left
            self.snake.direction = np.matmul( np.array([[0,-1],[1,0]]),self.snake.direction).tolist() # rotate the direction toward left
        #if value == 3:  # don nothing
        self.snake.code_direction()

        state, reward, done, score = self.update()
        return state, reward, done, score


    def check_collision(self):
        head = self.snake.body[0]
        done = head.x > width - 1 or head.x < 0 or head.y > height - 1 or head.y < 0  or self.snake.steps> 500 #or head in self.snake.body[2:]
        self.done =  torch.tensor([int(done)])
        return self.done

    def draw_score(self):
        text = str(self.snake.score)
        score_surface = game_font.render(text, True, (0, 0, 0))
        x = cell_size + 1
        y = cell_size / 2
        rect = score_surface.get_rect(center=(x, y))
        screen.blit(score_surface, rect)

    def draw_game_over(self):
        over_surface = game_font.render('Over!', True, (200, 0, 0))
        x = cell_size * width / 2
        y = cell_size * height / 2
        rect = over_surface.get_rect(center=(x, y))
        screen.blit(over_surface, rect)
        pygame.display.update()
        time.sleep(2)

    def draw(self):
        self.fruit.draw()
        self.snake.draw()
        self.draw_score()
        if self.done: self.draw_game_over()

    def render(self, tick):
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
        clock.tick(tick)  # this is for the graphic update, the program will never run at more than 10 frames per second.

