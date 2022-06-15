import random
import numpy as np
from collections import namedtuple  #--------
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from datetime import datetime
import h5py
import os
is_ipython = 'inline' in plt.get_backend()
if is_ipython:
    from IPython import display
    print('from IPython, display is imported')


# The Policy network
class DN(nn.Module):
    def __init__(self, state_len, numb_actions):
        super(DN, self).__init__()
        self.state_len = state_len
        self.fc1 = nn.Linear(in_features=state_len, out_features=state_len * 10)
        self.fc2 = nn.Linear(in_features=state_len * 10, out_features=state_len*5)
        self.out = nn.Linear(in_features=state_len*5, out_features=numb_actions)

    def forward(self, t):
        t = t.reshape(-1, self.state_len)
        t = self.fc1(t)
        t = F.relu(t)
        t = self.fc2(t)
        t = F.relu(t)
        qvalues = self.out(t)
        #qvalues = F.softmax(qvalues,dim = 1) # qvalues.shape = [batch_length, 2]
        return qvalues
class DN2(nn.Module):
    def __init__(self, state_len, numb_actions):
        super(DN2, self).__init__()
        self.state_len = state_len
        self.fc1 = nn.Linear(in_features=state_len, out_features=state_len * 5)
        self.fc2 = nn.Linear(in_features=state_len * 5, out_features=state_len * 10)
        self.fc3 = nn.Linear(in_features=state_len * 10, out_features=state_len*5)
        self.out = nn.Linear(in_features=state_len*5, out_features=numb_actions)

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

## 2.2 Define the experience tuple:
Experience = namedtuple('experience', ('state', 'action', 'reward', 'next_state', 'done'))

## 2.3 Replay Memory:
class replaymemory():
    # define the memory parameters:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.push_count = 0
        self.last_hund_expr = []

    # feeding function
    def push(self, experience):
        # if self.push_count < self.capacity:
        last_exp_indx = len(self.memory)
        if last_exp_indx < self.capacity:
            self.memory.append(experience)
        else:
            last_exp_indx = self.push_count % self.capacity
            self.memory[last_exp_indx] = experience

        self.push_count += 1

        if len(self.last_hund_expr) < 100:
            self.last_hund_expr.append(last_exp_indx)
        else:
            self.last_hund_expr.append(last_exp_indx % 100)


    # outputting function:
    def sample(self, batch_size):
        # rate = .8
        # rate_ = 1-rate
        # rand_sample = random.sample(self.memory, int(batch_size*rate))
        # last_hun_sample = [self.memory[i] for i in self.last_hund_expr[0:int(batch_size*rate_)]]
        # return rand_sample + last_hun_sample
        return random.sample(self.memory, batch_size)


    def can_provide_sample(self, batch_size):
        return len(self.memory) >= batch_size

## 2.6 Tensor exrators
def extract_tensors(experiences):
    # Convert batch of Experiences to Experience of batches
    batch = Experience(*zip(*experiences))
    t1 = torch.cat(batch.state)
    t2 = torch.cat(batch.action)
    t3 = torch.cat(batch.reward)
    t4 = torch.cat(batch.next_state)
    t5 = torch.cat(batch.done)
    return (t1, t2, t3, t4, t5)

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


## 2.5 Reinforcement Learning Agent Class:
class Agent():
    def __init__(self, strategy, num_actions, device):
        self.num_actions = num_actions
        self.strategy = strategy
        self.current_steps = 0
        self.device = device
        self.exploration_rate = .3
    # trade off: exploration vs exploitation
    def decide(self, policy_net, state, memory = None):
        self.exploration_rate = self.strategy.get_exploration_rate(self.current_steps)
        self.current_steps += 1

        if self.exploration_rate > random.random():  # explore
            action = random.randrange(self.num_actions)
            action = torch.tensor([action]).to(self.device)
            decision = 'exploration'

        else:  # otherwise exploite:
            with torch.no_grad():
                action = policy_net(state).argmax(dim=1).to(self.device)  # policy_net(self.state).argmax(dim=1).item()
            decision = 'exploitation'

        if self.exploration_rate < self.strategy.end*1.1 :
            self.current_steps = 0

        return action, decision, self.exploration_rate, 1



class Agent2():
    def __init__(self, strategy, num_actions, device):
        self.num_actions = num_actions
        self.strategy = strategy
        self.current_steps = 0
        self.exploration_steps = 0
        self.device = device

 # trade off: exploration vs exploitation
    def decide(self, policy_net, state, memory):
        #states, actions, _, _, _ = extract_tensors(memory)  # f...
        push_in_memory = False
        # checke the non executed actions in states,actions history
        Ane = non_executed_actions(state, self.num_actions, memory)
        # compute the exporation rate:
        rate = self.strategy.get_exploration_rate(self.exploration_steps)
        if rate > random.random() :  # If it's likely to explore AND there are some non executed actions, then explore:
            if Ane != [] :
                action = random.choice(Ane)
                action = torch.tensor([action]).to(self.device)
                decision = 'exploration'
                push_in_memory = True #***
        else:  # otherwise exploite:
            with torch.no_grad():
                action = policy_net(state).argmax(dim=1).to(self.device)  # policy_net(self.state).argmax(dim=1).item()
                push_in_memory = action in Ane #***
            decision = 'exploitation'

        self.exploration_steps += push_in_memory
        #***# push_in_memory is False if :  1) the state is fully explored (all actions already executed ), when the agent tries to explore, and
            #                               2) the action, taken by exploiting the policy, is also executed , i.e not in Ane
        return action, decision, rate, push_in_memory




## 2.7  ################################ QValues class
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


########################## Plot Funtions ##################################################

def get_moving_average(period, values):
    values = torch.tensor(values, dtype=torch.float)
    if len(values) >= period:
        moving_avg = values.unfold(dimension=0, size=period, step=1) \
            .mean(dim=1).flatten(start_dim=0)
        moving_avg = torch.cat((torch.zeros(period-1), moving_avg))
        return moving_avg.numpy()
    else:
        moving_avg = torch.zeros(len(values))
        return moving_avg.numpy()


def plot(episode, rate, loss_history, steps,score_history, moving_avg_period):
    steps_avg = get_moving_average(moving_avg_period, steps)
    loss_avg = get_moving_average(moving_avg_period, loss_history)
    score_avg =  get_moving_average(moving_avg_period, score_history)

    plt.figure(2)
    plt.clf()
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')

    #plt.plot(steps)
    plt.plot(steps_avg)
    plt.plot(rate)
    plt.plot(loss_avg/loss_avg.max())
    plt.plot(score_avg)

    print('Episode: ', episode, '| Score average: ',score_avg[-1], '| Steps average: ', steps_avg[-1], ' | loss_avg: ',
          loss_avg[-1])  # , ' | exp_size: ', len(memory.memory))


    plt.pause(0.0001)
    if is_ipython: display.clear_output(wait=True)

    return loss_avg[-1], steps_avg[-1]



#################### IO functions ########################################################

def tag_now():
    return datetime.now().strftime("%Y-%m-%d_%Hh%Mm%Ss")


def save_h5_dataset(data, filename):
    states, actions, rewards, next_states, dones = extract_tensors(data)
    # ---- Create h5 file
    with h5py.File(filename, "w") as f:
        f.create_dataset("states", data = states)
        f.create_dataset("actions", data=actions)
        f.create_dataset("rewards", data=rewards)
        f.create_dataset("next_states", data=next_states)
        f.create_dataset("dones", data=dones)
    # ---- done
    size = os.path.getsize(filename) / (1024 * 1024)
   # print('Dataset : {:24s}  state_shape : {:22s} size : {:6.1f} Mo   (saved)'.format(filename, str(states.shape), size))

def load_h5_dataset(filename):
    with  h5py.File(filename, 'r') as f:
        states = f['states'][:]
        actions = f['actions'][:]
        rewards = f['rewards'][:]
        next_states = f['next_states'][:]
        dones = f['dones'][:]
    return states, actions, rewards, next_states, dones


#def check_memory(state, action, states, actions):

def non_executed_actions(state, possible_actions, memory):
    A = range(possible_actions)
    Ane = A

    if memory.memory == []:
        # the memory is empyt, no exploration done, i.e all actions are non executed.
        return Ane
    else: # the memory is not empty, then check for the tupe state action is explored or not, if yes remove the action from Ane
        states, actions, rewards, next_states, dones = extract_tensors(memory.memory)  # f...
        states = states.tolist()
        actions = actions.tolist()
        memory = zip(states,actions)
        for action in A:
            eventual_experience = (state, action)
            if eventual_experience in memory:
                Ane.remove(action)
        return Ane