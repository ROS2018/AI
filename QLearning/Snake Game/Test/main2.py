import pygame, sys, random
from pygame.math import Vector2 as vc
import numpy as np

# My Classes and Functions:

class FRUIT:
    def __init__(self):
        self.pos = vc(random.randint(0, cell_number), random.randint(0,cell_number))

    def draw(self):
        # creat rectangle
        fruit_rect = pygame.Rect(self.pos.x*cell_size, self.pos.y*cell_size, cell_size, cell_size)
        pygame.draw.rect(screen, (126, 50, 70), fruit_rect)

class SNAKE:
    def __init__(self):
        self.body = [vc(5,5),vc(6,5)]
        print(self.body)
        self.direction = vc(-1,0)
    def draw(self):
        for block in self.body :
            rect = pygame.Rect(block.x*cell_size, block.y*cell_size, cell_size, cell_size)
            pygame.draw.rect(screen, (0, 250, 0), rect)

    def move(self):
        body_copy = self.body[:-1]
        body_copy.insert(0,body_copy[0]+self.direction)
        self.body = body_copy[:]

    def munch(self):
        if fruit.pos == self.body[0]:
            self.body.insert(0,fruit.pos)



# initiation:
pygame.init()
cell_size = 40
cell_number = 20
screen = pygame.display.set_mode((cell_number*cell_size, cell_number*cell_size)) # (screen_width, screen_height)
pygame.display.set_caption('Snake')
clock = pygame.time.Clock() # to control the speed of displaying

# surfaces (test):
#test_surface  = pygame.Surface((cell_size,cell_size)) # surface dimension
#test_surface.fill((0,50,100)) #  color
#test_rect = pygame.Rect(100, 200, 70,70 ) # (x,y, w, h) #  make rectangle
#test_rect = test_surface.get_rect(center = (200,250)) # topright # make rectangle on test_surface
#x_pos = 0

# Define Environment components:
fruit = FRUIT()
snake = SNAKE()

# Events:
SCREEN_UPDATE =  pygame.USEREVENT # this is a custom event that could trigger
pygame.time.set_timer(SCREEN_UPDATE,150)

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
        if event.type == SCREEN_UPDATE:
            snake.move()
            snake.munch()

        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_UP:
                snake.direction = vc(0, -1)
            if event.key == pygame.K_DOWN:
                snake.direction = vc(0,1)
            if event.key == pygame.K_LEFT:
                snake.direction = vc(-1,0)
            if event.key == pygame.K_RIGHT:
                snake.direction = vc(1,0)

    screen.fill((150,200,70)) #pygame.Color('green')

    fruit.draw()
    snake.draw()
    #fruit.pos += pygame.math.Vector2(1,1)

    #x_pos +=1
    #screen.blit(test_surface,(200,250))
    #pygame.draw.rect(screen, pygame.Color('black'), test_rect )
    #test_rect.right +=1
    #screen.blit(test_surface, test_rect)
    pygame.display.update()
    clock.tick(60)


