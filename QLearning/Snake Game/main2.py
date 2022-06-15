# Python stuff:
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt  #--------
from collections import namedtuple  #--------
from itertools import count #--------
from PIL import Image #--------
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
from torchsummary import summary
import game2 as game
import dqn_tools
import time
from IPython.display import clear_output

# 2) Define Some Classes:
## 2.1 Define the DQN class:

class DN(nn.Module):
 def __init__(self, state_len, numb_actions):
  super(DN, self).__init__()
  self.state_len = state_len
  self.fc1 = nn.Linear(in_features=state_len, out_features=state_len * 2)
  self.fc2 = nn.Linear(in_features=state_len * 2, out_features=state_len * 4)
  self.fc3 = nn.Linear(in_features=state_len * 4, out_features=state_len)
  self.out = nn.Linear(in_features=state_len, out_features=numb_actions)

 def forward(self, t):
  t = t.reshape(-1, self.state_len)
  t = self.fc1(t)
  t = F.relu(t)
  t = self.fc2(t)
  t = F.relu(t)
  t = self.fc3(t)
  t = F.relu(t)
  qvalues = self.out(t)
  #qvalues = F.softmax(qvalues,dim = 1) # qvalues.shape = [batch_length, 2]
  return qvalues

print('finished ...')

## 2.2 Define the experience tuple:

Experience = namedtuple('experience', ('state', 'action', 'reward', 'next_state', 'done'))

## 2.3 Replay Memory:

class replaymemory():

 # define the memory parameters:
 def __init__(self, capacity):
  self.capacity = capacity
  self.memory = []
  self.push_count = 0

 # feeding function

 def push(self, experience):
  # if self.push_count < self.capacity:
  if len(self.memory) < self.capacity:
   self.memory.append(experience)
  else:
   self.memory[self.push_count % self.capacity] = experience
  self.push_count += 1

 # outputting function:

 def sample(self, batch_size):
  return random.sample(self.memory, batch_size)

 def can_provide_sample(self, batch_size):
  return len(self.memory) >= batch_size

  # print(random.sample(range(10),3))

print('finished ...')

## 2.4 Epsilon Greedy Strategy :

class EpsilonGreedyStrategy():
 def __init__(self, start, end, decay):
  self.start = start
  self.end = end
  self.decay = decay
  self.history = []

 def get_exploration_rate(self, episode):
  rate = self.end + (self.start - self.end) * np.exp(-self.decay * episode)
  self.history.append(rate)
  return rate


print('finished ...')

## 2.5 Reinforcement Learning Agent Class:

class Agent():
 def __init__(self, strategy, num_actions, device):
  self.num_actions = num_actions
  self.strategy = strategy
  self.current_steps = 0
  self.device = device

 # trade off: exploration vs exploitation
 def decide(self, policy_net, state):
  rate = self.strategy.get_exploration_rate(self.current_steps)
  self.current_steps += 1

  if rate > random.random():  # explore
   action = random.randrange(self.num_actions)
   action = torch.tensor([action]).to(self.device)
   decision = 'exploration'

  else:  # otherwise exploite:
   with torch.no_grad():
    action = policy_net(state).argmax(dim=1).to(self.device)  # policy_net(self.state).argmax(dim=1).item()
   decision = 'exploitation'
  return action, decision, rate


print('finished ...')


# 3) Define some functions:
## 3.1 Tensor exrators

def extract_tensors(experiences):
 # Convert batch of Experiences to Experience of batches
 batch = Experience(*zip(*experiences))
 t1 = torch.cat(batch.state)
 t2 = torch.cat(batch.action)
 t3 = torch.cat(batch.reward)
 t4 = torch.cat(batch.next_state)
 t5 = torch.cat(batch.done)
 return (t1, t2, t3, t4, t5)



## 3.2 ################################QValues class

class QValues():
 device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

 @staticmethod
 def get_current_qvalues(policy_net, states, actions):
  return policy_net(states).gather(dim=1, index=actions.unsqueeze(-1))

 @staticmethod
 def get_next_qvalues(target_net, next_states, dones):
  #final_state_locations = next_states.flatten(start_dim=1) \
  non_final_state_locations = (dones == 0)
  non_final_states = next_states[non_final_state_locations]
  batch_size = next_states.shape[0]
  values = torch.zeros(batch_size).to(QValues.device)
  values[non_final_state_locations] = target_net(non_final_states).max(dim=1)[0].detach()
  return values


print('finished!')

# 3) #######################################Main Program

# %% md

## 3.1 hyperparameters:

# %%

batch_size = 256
gamma = 0.99 # it is the 0.999 that made me crazy :P
eps_start = 1
eps_end = 0.1
eps_decay = 0.000001
target_update = 75
memory_size = 50000
net_learning_rate = 0.001
num_episodes = 1000000

print('finished ...')


########################################## 3.2 Essentiel Objects :

# setup the device :
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# the environement manager:
em = game.ENVIRONMENT()

# strategy (exploration rate updater)
strategy = EpsilonGreedyStrategy(eps_start, eps_end, eps_decay)

# Agent:
agent = Agent(strategy, em.num_actions, device)

# Memory:
memory = replaymemory(memory_size)

# Target Network and Policy Network:
policy_net = DN(em.state_len, em.num_actions).to(device)
target_net = DN(em.state_len, em.num_actions).to(device)  # target_net =  policy_net
# --
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()
# set the optimizer :
optimizer = optim.Adam(params=policy_net.parameters(), lr=net_learning_rate)
#torch.save(policy_net,'model.pth')
print('>>> finished ...')

# %% md

###################################################### 3.3 Training Loop :

# %%

# array to store scores during training in order to plot them using the plot():
steps = []
exploration_rate = []
loss_history = []
# em.get_state()
for episode in range(num_episodes):  # 4 for each episode
 state = em.get_state()  # A) initialize the starting state

 for step in count():  # range(20): #count(): # B) for each step:
  action, decision, rate = agent.decide(policy_net, state)  # a. let the agetn select an action based on the exploration rate
  new_state,reward,done,score = em.change_direction(action)  # b. execute the agent selected decision.
  experience = Experience(state, action, reward, new_state, done)
  memory.push(experience)  # d. store the experience in the reaplay memory
  state = new_state  # update for the next upcoming step.
  #if episode%10 == 0:
  em.render()
  ## Learn from the stored experiences:
  if memory.can_provide_sample(batch_size):
   batch = memory.sample(batch_size)  # e. Sample random batch from replay memory.
   states, actions, rewards, next_states, dones = extract_tensors(batch)  # f...
   current_qvalues = QValues.get_current_qvalues(policy_net, states, actions)  # g. Pass batch of preprocessed states to POLICY network.
   next_qvalues = QValues.get_next_qvalues(target_net, next_states, dones)  # g2. Pass batch of preprocessed states to TARGET network.
   target_qvalues = rewards + gamma * next_qvalues

   loss = F.mse_loss(current_qvalues, target_qvalues.unsqueeze(1))  # h. the loss between target q-values and output q-values
   optimizer.zero_grad()  # i. backpropagation process
   loss.backward()
   optimizer.step()
  else:
   loss = torch.tensor([-1])

  # display.clear_output(wait=True)

  if done:
   steps.append(step)
   exploration_rate.append(rate*10)
   loss_history.append(loss.item())
   dqn_tools.plot(exploration_rate,loss_history,steps, 100)
   break

  # show the state:
  # show_state(episode, step, em)
  # update_progress('Episodes: ', episode , num_episodes)

 if episode % target_update == 0:
  target_net.load_state_dict(policy_net.state_dict())

 if episode % 500 :
  torch.save(policy_net,'policy_net.pth')

 # show the progression:
 print('Episode: ', episode, '| Step: ', steps[-1], ' | Score: ', em.snake.score)  # , ' | exp_size: ', len(memory.memory))


print('>>> finished ...')

