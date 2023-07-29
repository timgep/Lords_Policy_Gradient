import logging
logging.getLogger().setLevel(logging.CRITICAL)
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import gymnasium as gym
import random
from collections import deque
import math
import copy
import pickle
import time

max_updates = 70 # maximum number of updates between episodes


#used to create random seeds in Actor -> less dependendance on the specific random seed.
def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)

#Rectified Hubber Error Loss Function
def ReHE(target, input, delta=1.0):
    k = 1.0 if input.shape[0]>=1024 else 0.45
    ae = k*torch.abs(input-target).mean()
    return delta*ae*torch.tanh(ae/delta)

#Spiking Activation Function - Neuron fires only when a normalized treshold (-0.7) is reached (reminds silicon treshold, 0.7v).
#to some extent prevents catastrofic forgetting
class Spike(nn.Module):
    def __init__(self, dim):
        super().__init__() # init the base class
        self.norm = nn.LayerNorm(dim)

    def forward(self, input):
        input = 0.7 + self.norm(input) #layer norm & shift
        input =  input * (input > 0.0).float() # relu
        return input * torch.tanh(7*input) - 0.7 #slighlty curve & shift back


class FadingMemories:
    def __init__(self, capacity=2000000):
        self.capacity = capacity
        self.cache, self.indices = [], []
        self.buffer, self.length = deque(maxlen=capacity), 0
        self.x, self.step, self.s, self.eps = 0.0, 5.0/capacity, 1.0, 0.1
        self.counter = 0


    # priority for old memories are fading gradually
    def fade(self, norm_index):
        return (1-self.s)*np.tanh(100*self.s*norm_index**2)

    #adds average between two to store more data (except transitions with dones)
    def add_average(self, transition):
        self.cache.append(transition)

        if len(self.cache)>=2:
            transition = self.cache[0]
            if self.cache[0][-1] == True or self.cache[1][-1] == True:
                del self.cache[0]
            else:
                for j, (x,y) in enumerate(zip(self.cache[0], self.cache[1])):
                    transition[j] = (x+y)/2
                self.cache = []
            self.add(transition)

    # adds to buffer
    def add(self, transition):
        self.buffer.append(transition)
        self.length = len(self.buffer)
        if self.length < self.capacity: self.indices.append(self.length-1)
            
        self.x += self.step
        self.s = (1.0 - self.eps)*math.exp(-self.x) + self.eps #exp decay from 1 to (0+eps)
        self.counter += 1



    def sample(self, device, batch_size=255):
        if len(self.buffer) <  4*batch_size: return None

        # samples big batch then re-samples smaller batch with less priority to old data
        sample_indices = random.sample(self.indices, k= 4*batch_size)
        probs = self.fade(np.array(sample_indices)/self.length)
        batch_indices = np.random.default_rng().choice(sample_indices, p=probs/np.sum(probs), size=batch_size, replace=False)
        batch = [self.buffer[indx-1] for indx in batch_indices]

        #Combined Experience Replay (the last transition is added to the batch):
        if len(batch)<=255: batch.append(self.buffer[-1])
        states, actions, rewards, next_states, dones = map(np.vstack, zip(*batch))
        
        return (
            torch.FloatTensor(states).to(device),
            torch.FloatTensor(actions).to(device),
            torch.FloatTensor(rewards).to(device),
            torch.FloatTensor(next_states).to(device),
            torch.FloatTensor(dones).to(device),
        )



    def __len__(self):
        return self.length



# Define the actor network
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=32, max_action=1.0):
        super(Actor, self).__init__()


        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            Spike(hidden_dim),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()
         )
        self.max_action = max_action
       

    def forward(self, state, mean=False):
        mu = self.net(state)
        std = 0.3*self.max_action
        lim = 3.0*std
        x = self.max_action*mu + (std*torch.randn_like(mu)).clamp(-lim,lim).to(device)
        return mu if mean else x.clamp(-self.max_action, self.max_action)

# Define the critic network
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=32):
        super(Critic, self).__init__()

        self.input = nn.Sequential(
            nn.Linear(state_dim+action_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )

        self.net1 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            Spike(hidden_dim),
            nn.Linear(hidden_dim, action_dim),
        )

        self.net2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            Spike(hidden_dim),
            nn.Linear(hidden_dim, action_dim),
        )

        self.net3 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            Spike(hidden_dim),
            nn.Linear(hidden_dim, action_dim),
        )

        self.action_dim = action_dim

    def forward(self, state, action, united=False):
        x = torch.cat([state, action], -1)
        x = self.input(x)

        x1 = self.net1(x)
        x2 = self.net2(x)
        x3 = self.net3(x)

        return torch.min(torch.stack([x1,x2,x3], dim=-1), dim=-1)[0]  if united else (x1, x2, x3)
        #return (x1+x2+x3)/3 if united else (x1, x2, x3)
       



# Define the soft actor-critic agent
class TD3(object):
    def __init__(self, state_dim, action_dim, hidden_dim, max_action, max_updates, device):
        self.actor = Actor(state_dim, action_dim, hidden_dim, max_action).to(device)
        
        self.critic = Critic(state_dim, action_dim, hidden_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=3e-4)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=7e-4)


        self.max_action = max_action
        self.max_updates = max_updates
        self.device = device
        self.state = None

    def select_action(self, state, policy_training):
        with torch.no_grad():
            state = torch.FloatTensor(state).reshape(-1,state.shape[-1]).to(device)
            action = self.actor(state, mean=policy_training)
        return action.cpu().data.numpy().flatten()
    
    def train_ctrl(self, policy_training, replay_buffer, episodic=False):
        if policy_training:
            td3.train(replay_buffer.sample(self.device))
            if episodic:
                for _ in range(min(len(replay_buffer.buffer)//10240, self.max_updates)):
                    td3.train(replay_buffer.sample(self.device, batch_size=2560))
                


    def train(self, batch):
        self.critic_update(batch)
        self.actor_update()


    def critic_update(self, batch):
        if batch != None:
            self.state, action, reward, next_state, done = batch

            with torch.no_grad():
                for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
                    target_param.data.copy_(0.997 * target_param.data + 0.003 * param.data)

                next_action = self.actor(next_state)
                q_next_target = self.critic_target(next_state, next_action, united=True)
                next_q_value = reward + (1.0-done) * 0.99 * q_next_target
            
            q1, q2, q3 = self.critic(self.state, action, united=False)
            critic_loss = ReHE(next_q_value, q1) + ReHE(next_q_value, q2) + ReHE(next_q_value, q3)

            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()
        else:
            self.state = None


    def actor_update(self):
        if self.state != None:
            action = self.actor(self.state, mean=True)
            q_new_policy = self.critic(self.state, action, united=True)
            actor_loss = -q_new_policy.mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

env = gym.make('Humanoid-v4')
env_test = gym.make('Humanoid-v4', render_mode="human")

#env = gym.make('Pendulum-v1', g=9.81)
#env_test = gym.make('Pendulum-v1', g=9.81, render_mode="human")

#env = gym.make('BipedalWalkerHardcore-v3')
#env_test = gym.make('BipedalWalkerHardcore-v3', render_mode="human")

#env = gym.make("LunarLander-v2", continuous=True)
#env_test = gym.make("LunarLander-v2", continuous=True, render_mode="human")


state_dim = env.observation_space.shape[0]
action_dim= env.action_space.shape[0]

hidden_dim = 128


print('action space high', env.action_space.high)

max_action = torch.FloatTensor(env.action_space.high).to(device) if env.action_space.is_bounded() else 1.0
max_action = 0.9*max_action

replay_buffer = FadingMemories()
td3 = TD3(state_dim, action_dim, hidden_dim, max_action, max_updates, device)

num_episodes, counter, total_rewards, test_rewards, policy_training = 1000000, 0, [], [], False

#load existing models

try:
    print("loading...")
    td3.actor.load_state_dict(torch.load('actor_model.pt'))
    td3.critic.load_state_dict(torch.load('critic_model.pt'))
    td3.critic_target.load_state_dict(torch.load('critic_target_model.pt'))
    with open('replay_buffer', 'rb') as file:
        dict = pickle.load(file)
        replay_buffer = dict['buffer']
        if len(replay_buffer)>=512 and not policy_training: policy_training = True
    print('models loaded, buffer length', len(replay_buffer))
except:
    print("problem during loading models")



for i in range(num_episodes):
    
    obs = env.reset()[0]
    rewards = []

    #**************************************************************************
    #**********This part can be done with real robot before training***********
    #-------------------pre-training for big buffer > 10240--------------------

    td3.train_ctrl(policy_training, replay_buffer, episodic=True)
   

    #-------------slightly random initial configuration as in OpenAI Pendulum-------------
    action = 0.1*max_action.to('cpu').numpy()*np.random.uniform(-1.0, 1.0, size=action_dim)
      
    for t in range(0,4):
        next_obs, reward, done, info, _ = env.step(action)
        obs = next_obs
        rewards.append(reward)
        counter += 1
        if done: break
        
    #processor releave
    if policy_training: time.sleep(0.5)
    #===========================================================================
    #****************************************************************************


    #------------------------training-------------------------

    done = False
    for steps in range(1,1000000,1):
         #-------------------decreases dependence on random seed: ------------------
        if not policy_training: td3.actor.apply(init_weights)
        if len(replay_buffer)>=1024 and not policy_training: policy_training = True

        action = td3.select_action(obs, policy_training)
        next_obs, reward, done, info, _ = env.step(action)
        replay_buffer.add_average([obs, action, reward, next_obs, done])
        obs = next_obs
        rewards.append(reward)

        td3.train_ctrl(policy_training, replay_buffer)

        counter += 1
        if done: break

    total_reward = sum(rewards)
    total_rewards.append(total_reward)
    average_reward = np.mean(total_rewards[-250:])

    print(f"Ep {i}: Rtrn = {total_reward:.2f}, Avg = {average_reward:.2f}| ep steps = {steps+4}")
    #====================================================
    

    #--------------------testing-------------------------
    if policy_training and i%100==0:
        
        torch.save(td3.actor.state_dict(), 'actor_model.pt')
        torch.save(td3.critic.state_dict(), 'critic_model.pt')
        torch.save(td3.critic_target.state_dict(), 'critic_target_model.pt')
        print("saving... ", end="")
        with open('replay_buffer', 'wb') as file:
            pickle.dump({'buffer': replay_buffer}, file)
        print("> done")

        for test_episode in range(10):

            #-----------------pre-training for big buffer > 10240---------------
            td3.train_ctrl(policy_training, replay_buffer, episodic=True)

            obs = env_test.reset()[0]
            rewards = []
            done = False
            for steps in range(1,1000000,1):
                action = td3.select_action(obs, policy_training)
                next_obs, reward, done, info , _ = env_test.step(action)
                replay_buffer.add_average([obs, action, reward, next_obs, done])
                obs = next_obs
                rewards.append(reward)

                td3.train_ctrl(policy_training, replay_buffer)

                
                counter += 1
                if done: break
            total_reward = sum(rewards)
            test_rewards.append(total_reward)
        print(f"Validation, Avg Rtrn = {np.mean(test_rewards[-10:]):.2f}, buffer len {len(replay_buffer)}| all steps {counter}")

    #====================================================
    
