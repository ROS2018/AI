# Python stuff:
from itertools import count #--------
import torch
# import game3 as game
import game4 as game

import dqn_tools
import time


########################################## 3.2 Essentiel Objects :
# setup the device :
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# the environement manager:
em = game.ENVIRONMENT()

# Policy Network:
policy_net = dqn_tools.DN(em.state_len, em.num_actions)#.to('CPU')
#policy_net.load_state_dict(torch.load('models/policy_net(3x3)_state_dict_2021-04-05_08h24m23s.pth'))

############################################## 3.3 Test Loop :

num_episodes = 30000000000000
steps = []
best_score = 0
win = False
random_state = None

for episode in range(num_episodes):  # 4 for each episode

 policy_net.load_state_dict(torch.load('models/model.pth', map_location= device))
 policy_net.eval()
 state = em.get_state()
 for step in count():  # range(20): #count(): # B) for each step:
  with torch.no_grad():
   action = policy_net(state).argmax(dim=1).to(device)
  state,reward,done,score = em.change_direction(action)  # b. execute the agent selected decision.
  em.render(tick= 10)

  # show the progression:
  print('Episode: ', episode, '| Step: ', step,  ' | Reward: ', reward.item() ,' | Score: ', em.snake.score, ' | best_score: ', best_score)  # , ' | exp_size: ', len(memory.memory))

  if done:
   print('*'*60)
   steps.append(step)
   break

  # save the best score:
  if best_score < em.snake.score:
   best_score = em.snake.score
   #torch.save(policy_net.state_dict(), 'models/best_model.pth')  # save model

 #if episode % 5 :
  #time.sleep(5)


print('>>> finished ...')

