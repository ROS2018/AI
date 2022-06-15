# Python stuff:
from itertools import count #--------
import torch
import torch.optim as optim
import torch.nn.functional as F
import game3 as game
import dqn_tools


# 3) #######################################Main Program

## 3.1 hyperparameters:

batch_size = 256
gamma = 0.95 # it is the 0.999 that made me crazy :P
eps_start = 1
eps_end = 0.1
eps_decay = 0.0000005
target_update = 75
memory_size = 100000
net_learning_rate = 0.001
num_episodes = 1000000

print('finished ...')


########################################## 3.2 Essentiel Objects :

# setup the device :
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# the environement manager:
em = game.ENVIRONMENT()
# strategy (exploration rate updater)
strategy = dqn_tools.EpsilonGreedyStrategy(eps_start, eps_end, eps_decay)

# Agent:
agent = dqn_tools.Agent(strategy, em.num_actions, device)

# Memory:
memory = dqn_tools.replaymemory(memory_size)

# Target Network and Policy Network:
policy_net = dqn_tools.DN(em.state_len, em.num_actions).to(device)
policy_net.load_state_dict(torch.load('models/policy_net(3x3)_state_dict.pth'))
policy_net.eval()
filename =  'models/policy_net(3x3)_state_dict_' + dqn_tools.tag_now() + '.pth'
# --
target_net = dqn_tools.DN(em.state_len, em.num_actions).to(device)  # target_net =  policy_net
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()
# set the optimizer :
optimizer = optim.Adam(params=policy_net.parameters(), lr=net_learning_rate)

###################################################### 3.3 Training Loop :
steps = []
exploration_rate = []
loss_history = []



for episode in range(num_episodes):  # 4 for each episode
 state = em.get_state()  # A) initialize the starting state

 for step in count():  # range(20): #count(): # B) for each step:
  action, decision, rate = agent.decide(policy_net, state)  # a. let the agetn select an action based on the exploration rate
  new_state,reward,done,score = em.change_direction(action)  # b. execute the agent selected decision.
  experience = dqn_tools.Experience(state, action, reward, new_state, done)
  memory.push(experience)  # d. store the experience in the reaplay memory
  state = new_state  # update for the next upcoming step.
  #em.render(tick = 100)
  ## Learn from the stored experiences:
  if memory.can_provide_sample(batch_size):
   batch = memory.sample(batch_size)  # e. Sample random batch from replay memory.
   states, actions, rewards, next_states, dones = dqn_tools.extract_tensors(batch)  # f...
   current_qvalues = dqn_tools.QValues.get_current_qvalues(policy_net, states, actions)  # g. Pass batch of preprocessed states to POLICY network.
   next_qvalues = dqn_tools.QValues.get_next_qvalues(target_net, next_states, dones)  # g2. Pass batch of preprocessed states to TARGET network.
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
   break

  # show the state:
  # show_state(episode, step, em)
  # update_progress('Episodes: ', episode , num_episodes)

 if episode % target_update == 0:
  target_net.load_state_dict(policy_net.state_dict())

 if episode % 100 == 0:
  torch.save(policy_net.state_dict(), filename)
  # show the progression:
  dqn_tools.plot(exploration_rate, loss_history, steps, 100)
  print('Episode: ', episode, '| Step: ', steps[-1], ' | Score: ', em.snake.score)  # , ' | exp_size: ', len(memory.memory))


print('>>> finished ...')

