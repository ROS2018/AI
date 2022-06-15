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
gamma = 0.99 # it is the 0.999 that made me crazy :P
eps_start = 1
eps_end = 0.1
eps_decay = 0.00001
target_update = 150
memory_size = 15000
net_learning_rate = 0.0002
num_episodes = 1000000

print('finished ...')


########################################## 3.2 Essentiel Objects :
# IO parameters:
now = dqn_tools.tag_now()
model_name = 'models/model0.pth'

data_name = 'data/' + now + '.h5'
# setup the device :
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# the environement manager:
#transist = True
em = game.ENVIRONMENT()
# strategy (exploration rate updater)
strategy = dqn_tools.EpsilonGreedyStrategy(eps_start, eps_end, eps_decay)

# Agent:
agent = dqn_tools.Agent(strategy, em.num_actions, device)

# Memory:
memory = dqn_tools.replaymemory(memory_size)

# Target Network and Policy Network:
policy_net = dqn_tools.DN(em.state_len, em.num_actions).to(device)
#policy_net.load_state_dict(torch.load('models/model.pth'))
#policy_net.eval()
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
score_history = [0]



for episode in range(num_episodes):
    pr_score = 0
    state = em.get_state()  # A) initialize the starting state
    for step in count():  # range(20): #count(): # B) for each step:
        action, decision, rate, push_in_memory = agent.decide( policy_net, state,memory)  # a. let the agetn select an action based on the exploration rate
        new_state, reward, done, score = em.change_direction(action)  # b. execute the agent selected decision.

        experience = dqn_tools.Experience(state, action, reward, new_state, done)
        if push_in_memory:
            memory.push(experience)  # d. store the experience in the reaplay memory
        state = new_state  # update for the next upcoming step.
        # em.render(tick = 100)
        ## Learn from the stored experiences:
        if memory.can_provide_sample(batch_size):
            batch = memory.sample(batch_size)  # e. Sample random batch from replay memory.
            states, actions, rewards, next_states, dones = dqn_tools.extract_tensors(batch)  # f...
            # Forward porpagation, qvalues, and loss
            current_qvalues = dqn_tools.QValues.get_current_qvalues(policy_net, states, actions)  # g. Pass batch of preprocessed states to POLICY network.
            next_qvalues = dqn_tools.QValues.get_next_qvalues(target_net, next_states, dones)  # g2. Pass batch of preprocessed states to TARGET network.
            target_qvalues = rewards + gamma * next_qvalues
            loss = F.mse_loss(current_qvalues, target_qvalues.unsqueeze(1))  # h. the loss between target q-values and output q-values
            # Back propagation
            optimizer.zero_grad()  # i. backpropagation process
            loss.backward()
            optimizer.step()
        else: #  this one is needed just to plot loss during the first period when there is no learning, i.e no loss computed
            loss = torch.tensor([-1])

        if done:
            steps.append(step/10)
            exploration_rate.append(rate)
            loss_history.append(loss.item()/3000)
            score_history.append(pr_score)
            #print(total_score)
            break

        pr_score = score

    if episode % target_update == 0:
        target_net.load_state_dict(policy_net.state_dict())


    if episode % 500 == 0:
        torch.save(policy_net.state_dict(), model_name) # save model
        dqn_tools.save_h5_dataset(memory.memory,data_name)
        # show the progression:
        loss_avg , steps_avg = dqn_tools.plot(episode,exploration_rate, loss_history, steps,score_history, 500)


print('>>> finished ...')
