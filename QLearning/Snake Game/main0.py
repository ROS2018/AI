import time
from pygame.math import Vector2 as vc
import numpy as np
import random
import game0 as game

env = game.ENVIRONMENT()
# while True:
#     env.render()

while True:
    action = random.randint(0,1)
    state, reward, done, score = env.change_direction(action)
    env.render()


print('finished')


