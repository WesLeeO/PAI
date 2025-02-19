import random
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from gymnasium.utils import seeding
from utils import ReplayBuffer, get_env, run_episode


class MLP(nn.Module):
    '''
    A simple ReLU MLP constructed from a list of layer widths.
    '''
    def __init__(self, sizes):
        super().__init__()
        layers = []
        for i, (in_size, out_size) in enumerate(zip(sizes[:-1], sizes[1:])):
            layers.append(nn.Linear(in_size, out_size))
            if i < len(sizes) - 2:
                layers.append(nn.ReLU())
        self.layers = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.layers(x)


class Critic(nn.Module):
    '''
    Simple MLP Q-function.
    '''
    def __init__(self, obs_size, action_size, num_layers, num_units):
        super().__init__()
        #####################################################################
        # TODO: add components as needed (if needed)

        self.net = MLP([obs_size + action_size] + ([num_units] * num_layers) + [1])

        #####################################################################

    def forward(self, x, a):
        #####################################################################
        # TODO: code the forward pass
        # the critic receives a batch of observations and a batch of actions
        # of shape (batch_size x obs_size) and batch_size x action_size) respectively
        # and output a batch of values of shape (batch_size x 1)
        
        # Simply concatenate actions and observations along the columns 
        obs_act =  torch.cat((x, a), dim=1)
        return self.net(obs_act)
        #####################################################################



class Actor(nn.Module):
    '''
    Simple Tanh deterministic actor.
    '''
    def __init__(self, action_low, action_high,  obs_size, action_size, num_layers, num_units):
        super().__init__()
        #####################################################################
        # TODO: add components as needed (if needed)

        self.net = MLP([obs_size] + ([num_units] * num_layers) + [action_size])

        #####################################################################
        # store action scale and bias: the actor's output can be squashed to [-1, 1]
        self.action_scale = (action_high - action_low) / 2
        self.action_bias = (action_high + action_low) / 2

    def forward(self, x):
        #####################################################################
        # TODO: code the forward pass
        # the actor will receive a batch of observations of shape (batch_size x obs_size)
        # and output a batch of actions of shape (batch_size x action_size)


        # Use hyperbolic tangent to obtain outputs in the range [-1, 1]
        actions = torch.tanh(self.net(x)) 
        # Use scales to obtain values interval wanted 
        scaled_actions = actions * self.action_scale + self.action_bias
        return scaled_actions
        #####################################################################
       


class Agent:

    #TD3 algorithm

    # automatically select compute device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    buffer_size: int = 50_000  # no need to change

    #########################################################################
    # TODO: store and tune hyperparameters here

    batch_size: int = 256
    gamma: float = 0.99  # MDP discount factor, 
    exploration_noise: float = 0.1  # epsilon for epsilon-greedy exploration

    
    #########################################################################
    tau: float = 0.005  # Polyak averaging coefficient for target networs update
    actor_lr: float = 1e-3
    critic_lr: float = 1e-3
    exploration_noise: float = 0.1
    policy_delay: int = 2
    noise_clip: float = 0.5
    policy_noise: float = 0.2

    def __init__(self, env):

        # extract informations from the environment
        self.obs_size = np.prod(env.observation_space.shape)  # size of observations
        self.action_size = np.prod(env.action_space.shape)  # size of actions
        # extract bounds of the action space
        self.action_low = torch.tensor(env.action_space.low).float()
        self.action_high = torch.tensor(env.action_space.high).float()

        #####################################################################
        # TODO: initialize actor, critic and attributes
        self.num_layers = 4
        self.dim_layers_actor = self.obs_size * 4
        self.dim_layers_critic = (self.obs_size + self.action_size) * 4

        # Define 
        self.actor = Actor(self.action_low, self.action_high, self.obs_size, self.action_size, self.num_layers, self.dim_layers_actor).to(self.device)
        self.actor_target = Actor(self.action_low, self.action_high, self.obs_size, self.action_size, self.num_layers, self.dim_layers_actor).to(self.device)
        # Guarantee weigths of neural networks are initially identical
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.actor_lr)

        self.critic_1 = Critic(self.obs_size, self.action_size, self.num_layers, self.dim_layers_critic).to(self.device)
        self.critic_target_1 = Critic(self.obs_size, self.action_size, self.num_layers, self.dim_layers_critic).to(self.device)
        self.critic_2 = Critic(self.obs_size, self.action_size, self.num_layers, self.dim_layers_critic).to(self.device)
        self.critic_target_2 = Critic(self.obs_size, self.action_size, self.num_layers, self.dim_layers_critic).to(self.device)
        self.critic_target_1.load_state_dict(self.critic_1.state_dict())
        self.critic_target_2.load_state_dict(self.critic_2.state_dict())
        self.critic_optimizer = optim.Adam(list(self.critic_1.parameters()) + list(self.critic_2.parameters()), lr=self.critic_lr)


        #####################################################################
        # create buffer
        self.buffer = ReplayBuffer(self.buffer_size, self.obs_size, self.action_size, self.device)
        self.train_step = 0
    
    def train(self):
        '''
        Updates actor and critic with one batch from the replay buffer.
        '''
        obs, action, next_obs, done, reward = self.buffer.sample(self.batch_size)

        #####################################################################
        # TODO: code training logic

        with torch.no_grad():
            noise = torch.clamp(
                torch.randn_like(action) * self.policy_noise,
                -self.noise_clip,
                self.noise_clip
            )
            # Add noise to the action to guarantee exploration
            next_action = self.actor_target(next_obs) + noise
            # Make sure action is within the range
            next_action = torch.clamp(next_action, self.action_low, self.action_high)
            target_q1 = self.critic_target_1(next_obs, next_action)
            target_q2 = self.critic_target_2(next_obs, next_action)
            #Future reward = 0 if done = 1 (episode is over)
            target_q = reward.unsqueeze(1) + self.gamma * (1 - done.unsqueeze(1)) * torch.min(target_q1, target_q2)

        #Current estimate of Q
        q1 = self.critic_1(obs, action)
        q2 = self.critic_2(obs, action)

        # Update of our critic networks
        critic_loss = nn.MSELoss()(q1, target_q) + nn.MSELoss()(q2, target_q)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Every two iterations
        if self.train_step % self.policy_delay == 0:
            # Peform backpropagation on the actor network based on the following loss 
            # Maximize value of the critic
            actor_loss = -self.critic_1(obs, self.actor(obs)).mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update target networks weights using a weighted average between current ones and estimated ones
            with torch.no_grad():
                for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                    target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
                for param, target_param in zip(self.critic_1.parameters(), self.critic_target_1.parameters()):
                    target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
                for param, target_param in zip(self.critic_2.parameters(), self.critic_target_2.parameters()):
                    target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        self.train_step += 1

        #####################################################################

    def get_action(self, obs, train):
        '''
        Returns the agent's action for a given observation.
        The train parameter can be used to control stochastic behavior.
        '''
        #####################################################################
        # TODO: return the agent's action for an observation (np.array
        # of shape (obs_size, )). The action should be a np.array of
        # shape (act_size, )

        # Unsqueeze since neural networks expect 2D inputs (process batches)
        obs = torch.tensor(obs, dtype=torch.float32).to(self.device).unsqueeze(0)
        action = self.actor(obs).squeeze(0)
        return torch.clip(action, self.action_low, self.action_high).numpy()
        
        #####################################################################

    def store(self, transition):
        '''
        Stores the observed transition in a replay buffer containing all past memories.
        '''
        obs, action, reward, next_obs, terminated = transition
        self.buffer.store(obs, next_obs, action, reward, terminated)


# This main function is provided here to enable some basic testing. 
# ANY changes here WON'T take any effect while grading.
if __name__ == '__main__':

    WARMUP_EPISODES = 10  # initial episodes of uniform exploration
    TRAIN_EPISODES = 50  # interactive episodes
    TEST_EPISODES = 300  # evaluation episodes
    save_video = False
    verbose = True
    seeds = np.arange(10)  # seeds for public evaluation

    start = time.time()
    print(f'Running public evaluation.') 
    test_returns = {k: [] for k in seeds}

    for seed in seeds:

        # seeding to ensure determinism
        seed = int(seed)
        for fn in [random.seed, np.random.seed, torch.manual_seed]:
            fn(seed)
        torch.backends.cudnn.deterministic = True

        env = get_env()
        env.action_space.seed(seed)
        env.np_random, _ = seeding.np_random(seed)

        agent = Agent(env)

        for _ in range(WARMUP_EPISODES):
            run_episode(env, agent, mode='warmup', verbose=verbose, rec=False)

        for _ in range(TRAIN_EPISODES):
            run_episode(env, agent, mode='train', verbose=verbose, rec=False)

        for n_ep in range(TEST_EPISODES):
            video_rec = (save_video and n_ep == TEST_EPISODES - 1)  # only record last episode
            with torch.no_grad():
                episode_return = run_episode(env, agent, mode='test', verbose=verbose, rec=video_rec)
            test_returns[seed].append(episode_return)

    avg_test_return = np.mean([np.mean(v) for v in test_returns.values()])
    within_seeds_deviation = np.mean([np.std(v) for v in test_returns.values()])
    across_seeds_deviation = np.std([np.mean(v) for v in test_returns.values()])
    print(f'Score for public evaluation: {avg_test_return}')
    print(f'Deviation within seeds: {within_seeds_deviation}')
    print(f'Deviation across seeds: {across_seeds_deviation}')

    print("Time :", (time.time() - start)/60, "min")
