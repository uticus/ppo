#!/usr/bin/env python

import os
import json

import numpy as np
from copy import deepcopy

import gym

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam


from tensorboardX import SummaryWriter

from ppo_loss import get_ppo_actor_loss_clipped_obj, get_ppo_actor_loss_clipped_obj_continuous, get_ppo_critic_loss

#ENV = 'LunarLander-v2'
# CONTINUOUS = False
ENV = 'LunarLanderContinuous-v2'
CONTINUOUS = True


EPISODES = 100000
EPISODES_START_WATCH = 90000
EPISODES_WATCH_STEP = 100

# losses
CFG_CLIP_EPSILON = 0.2  # clip loss
CFG_BETA = 1e-3         # exploration - loss entropy beta for discrete action space
CFG_SIGMA = 1.0         # loss exploration noise

NOISE = 1.0         # exploration noise for continues action space

MEMORY_SIZE = 256
GAMMA = 0.99        # discount rewards

NEURONS = 128
NUM_LAYERS = 2

EPOCHS = 10
BATCH_SIZE = 64

LR = 1e-4 # Lower lr stabilises training greatly



########################################################
# environment for Open AI gym
########################################################

class GymEnvironment:
    ''' openAI gym environment for single player '''
    def __init__(self, problem, **kwargs):
        self.problem = problem
        self.env = gym.make(self.problem)

        self.__state = None
        self.__reward = 0
        self.__action = None
        self.__done = False
        pass

    def get_state_size(self):
        ''' return size of states '''
        return self.env.observation_space.shape[0]

    def get_action_size(self):
        ''' return number of possible actions '''
        return self.env.action_space.n

    def get_action_space(self):
        return self.env.action_space.shape, self.env.action_space.low, self.env.action_space.high

    def render(self):
        ''' render the game - can be empty '''
        self.env.render()
        pass

    def reset(self):
        ''' reset the game '''
        # gym reset returns initial state
        state = deepcopy( self.env.reset() )
        self.__state = state

        self.__done = False
        pass

    def state(self, agentId):
        ''' return state '''
        return self.__state

    def agent_reward(self, agentId):
        ''' return last action reward '''
        #pop reward
        reward = self.__reward
        self.__reward = 0
        return reward

    def order(self, agentId, action):
        ''' set order to player '''
        # store the order
        self.__action = action
        pass

    def pulse(self):
        ''' implement all given orders '''
        # apply action
        action = self.__action
        state, reward, done, info = self.env.step(action)
        state = deepcopy(state)
        self.__state, reward, self.__done, info = (state, reward, done, info)
        self.__reward += reward
        # ignore info
        pass

    def Done(self):
        ''' check if game is finished '''
        return self.__done


########################################################
# Model
########################################################


class ActorModelBase:
    def __init__(self, state_size, action_size, **kwargs):
        self.state_size = state_size
        self.action_size = action_size
        self.batch_size = kwargs.get('batch_size', 1)

        self.DUMMY_ACTION = np.zeros((1, self.action_size))
        self.DUMMY_VALUE = np.zeros((1, 1))

        learning_rate = kwargs.get('learning_rate', 1e-4)
        self.optimizer = kwargs.get('optimizer', Adam(lr=learning_rate))

        # self.metrics = kwargs.get('metrics', ['accuracy'])
        self.metrics = kwargs.get('metrics', None)

        self.neurons = kwargs.get('neurons', self.state_size * self.batch_size)
        self.hidden_layers = kwargs.get('hidden_layers', 2)

        self.kernel_initializer = kwargs.get('kernel_initializer', 'glorot_uniform')
        # self.kernel_initializer = kwargs.get('kernel_initializer', 'he_uniform')

        self.activation = kwargs.get('activation', 'tanh')
        self.activation_last = kwargs.get('activation_last', 'softmax')

        self.Initialize(**kwargs)
        pass

    def Initialize(self, **kwargs):
        model = kwargs.get('model', None)
        if not model == None:
            self.model = model
            self.compileModel(**kwargs)
        else:
            self.model = self.createModel(**kwargs)
            self.compileModel(**kwargs)
        self.x_shape, self.y_shape = self.get_input_shapes()

    def createModel(self, **kwargs):
        """ create and return model """
        input_state = Input(shape=(self.state_size,))
        input_advantage = Input(shape=(1,))
        input_old_prediction = Input(shape=(self.action_size,))
        inputs = [input_state, input_advantage, input_old_prediction]

        x = Dense(self.neurons, kernel_initializer=self.kernel_initializer, activation=self.activation)(input_state)
        for _ in range(self.hidden_layers - 1):
            x = Dense(self.neurons, kernel_initializer=self.kernel_initializer, activation=self.activation)(x)

        out_actions = Dense(self.action_size, kernel_initializer=self.kernel_initializer,
                            activation=self.activation_last, name='output')(x)

        model = Model(inputs=inputs, outputs=[out_actions])

        # print(type(input_advantage))

        # self.loss = proximal_policy_optimization_loss(
        #                  advantage=advantage,
        #                  old_prediction=old_prediction )

        # model.compile(optimizer=Adam(lr=LR),
        #              #loss=[self.loss ])
        #              loss=self.loss)
        # model.summary()

        return model

    def createLoss(self, **kwargs):
        """ create loss function """
        raise NotImplementedError

    @property
    def continuous(self):
        """ return if continuous action space """
        raise NotImplementedError


    def compileModel(self, **kwargs):
        """ compile the existing model with optimizer, loss and metrics """
        self.optimizer = kwargs.get('optimizer', self.optimizer)
        self.metrics = kwargs.get('metrics', self.metrics)

        # make custom loss
        self.loss = self.createLoss(**kwargs)

        self.model.compile(optimizer=self.optimizer, loss=self.loss, metrics=self.metrics)
        # self.model.compile( optimizer=self.optimizer, loss=self.loss)

        self.model.summary()
        pass

    def get_input_shapes(self):
        """ return (x_shape,y_shape) """
        # return [(self.batch_size, self.state_size,), (self.batch_size, 1,), (self.batch_size, self.state_size,),
        #        (self.batch_size, self.action_size,)], (self.batch_size, self.action_size,)
        return (self.batch_size, self.state_size,), (self.batch_size, self.action_size,)

    def predict(self, state):
        x = np.array(state)
        p = self.model.predict([x.reshape(1, self.state_size), self.DUMMY_VALUE, self.DUMMY_ACTION])
        return p[0]

    pass


class ActorModelDiscrete(ActorModelBase):
    def __init__(self, state_size, action_size, **kwargs):
        # clipping for the surrogate loss
        self.loss_clip_epsilon = kwargs.get('loss_clip_epsilon', 0.2)
        self.loss_entropy_beta = kwargs.get('loss_entropy_beta', 1e-3)

        super().__init__(state_size, action_size, **kwargs)
        pass

    def createLoss(self, **kwargs):
        input_advantage_ = self.model.input[1]
        input_old_prediction_ = self.model.input[2]
        return get_ppo_actor_loss_clipped_obj(
            input_advantage_,
            input_old_prediction_,
            clip_epsilon=self.loss_clip_epsilon,
            beta=self.loss_entropy_beta)

    @property
    def continuous(self):
        """ return if continuous action space """
        return False

    pass


class ActorModelContinuous(ActorModelBase):
    def __init__(self, state_size, action_size, **kwargs):
        # clipping for the surrogate loss
        self.loss_clip_epsilon = kwargs.get('loss_clip_epsilon', 0.2)
        self.loss_sigma = kwargs.get('loss_sigma', 1.0)
        super().__init__(state_size, action_size, **kwargs)
        pass

    def createLoss(self, **kwargs):
        input_advantage_ = self.model.input[1]
        input_old_prediction_ = self.model.input[2]
        return get_ppo_actor_loss_clipped_obj_continuous(
            input_advantage_,
            input_old_prediction_,
            clip_epsilon=self.loss_clip_epsilon,
            sigma=self.loss_sigma)

    @property
    def continuous(self):
        """ return if continuous action space """
        return True


class CriticModel:
    def __init__(self, state_size, action_size, **kwargs):
        self.state_size = state_size
        self.action_size = action_size
        self.batch_size = kwargs.get('batch_size', 1)
        # self.history_callback = kwargs.get('history_callback', [])

        self.metrics = kwargs.get('metrics', None)

        self.loss_clip = float(kwargs.get('loss_clip', np.inf))
        self.loss = get_ppo_critic_loss(self.loss_clip)

        learning_rate = kwargs.get('learning_rate', 1e-4)
        self.optimizer = kwargs.get('optimizer', Adam(lr=learning_rate))

        self.neurons = kwargs.get('neurons', self.state_size * self.batch_size)
        self.hidden_layers = kwargs.get('hidden_layers', 2)
        self.kernel_initializer = kwargs.get('kernel_initializer', 'glorot_uniform')
        # self.kernel_initializer = kwargs.get('kernel_initializer', 'he_uniform')

        self.activation = kwargs.get('activation', 'tanh')
        self.Initialize(**kwargs)

    def Initialize(self, **kwargs):
        model = kwargs.get('model', None)
        if not model == None:
            self.model = model
            self.compileModel(**kwargs)
        else:
            self.model = self.createModel(**kwargs)
            self.compileModel(**kwargs)
        self.x_shape, self.y_shape = self.get_input_shapes()

    def createModel(self, **kwargs):
        """ create and return model """
        state_input = Input(shape=(self.state_size,))
        x = Dense(self.neurons, kernel_initializer=self.kernel_initializer, activation=self.activation)(state_input)
        for _ in range(self.hidden_layers - 1):
            x = Dense(self.neurons, kernel_initializer=self.kernel_initializer, activation=self.activation)(x)

        out_value = Dense(1)(x)

        model = Model(inputs=[state_input], outputs=[out_value])
        # model.compile(optimizer=Adam(lr=LR), loss=self.loss)
        return model

    def compileModel(self, **kwargs):
        """ compile the existing model with optimizer, loss and metrics """
        self.optimizer = kwargs.get('optimizer', self.optimizer)
        self.metrics = kwargs.get('metrics', self.metrics)

        self.model.compile(optimizer=self.optimizer, loss=self.loss, metrics=self.metrics)
        # self.model.compile( optimizer=self.optimizer, loss=self.loss)

        self.model.summary()
        pass

    def get_input_shapes(self):
        """ return (x_shape,y_shape) """
        return (self.batch_size, self.state_size,), (self.batch_size, 1,)

    def predict(self, state):
        p = self.model.predict(state)
        return p


########################################################
# Exploration, Noise policy
########################################################
class GreedyQPolicy:
    def __init__(self):
        super().__init__()
        pass

    def SelectAction(self, q_values: np.array):
        assert isinstance(q_values, (np.ndarray))
        assert q_values.ndim == 1, "q_values.ndim = " + str(q_values.ndim)
        return np.argmax(q_values)

    def Tick(self):
        pass


class PPOExporationPolicy:
    def __init__(self, **kwargs):

        super().__init__()
        self.interval = kwargs.get('interval', 100)
        self.__cnt = kwargs.get('cnt', 0)
        self.__r = kwargs.get('initial', True)

    def CheckIfRandom(self):
        return self.__r

    def Tick(self):
        ''' optional tick '''
        self.__cnt += 1
        if self.__cnt % self.interval == 0:
            self.__r = False
        else:
            self.__r = True
        pass

class PPONoisePolicyContinuous:
    def __init__(self, **kwargs):
        super().__init__()
        self.noise = kwargs.get('noise', 1.0)

    def SelectAction(self, policy):
        action = policy + np.random.normal(loc=0, scale=self.noise, size=policy.shape)
        return action

    def Tick(self):
        pass


class PPONoisePolicyDiscrete:
    def __init__(self, action_size, **kwargs):
        super().__init__()
        self.action_size = action_size

    def SelectAction(self, policy):
        action = np.random.choice(self.action_size, p=np.nan_to_num(policy))
        return action

    def Tick(self):
        pass


class PPOPolicy:
    def __init__(self, **kwargs):
        self.is_frozen = False
        self.__Initialize(
            kwargs.get('explorationPolicy'),
            kwargs.get('qPolicy'),
            kwargs.get('noisePolicy')
        )
        pass

    def __Initialize(self, explorationPolicy, qPolicy, noisePolicy,
                     ):
        self.explorationPolicy = explorationPolicy
        self.qPolicy = qPolicy
        self.noisePolicy = noisePolicy
        pass

    def CheckIfRandom(self):
        if self.explorationPolicy is not None:
            return self.explorationPolicy.CheckIfRandom()
        return False

    def SelectAction(self, q_values: np.array):
        if self.CheckIfRandom():
            if self.noisePolicy is not None:
                q_values = self.noisePolicy.SelectAction(q_values)
        else:
            if self.qPolicy is not None:
                q_values = self.qPolicy.SelectAction(q_values)
        return q_values

    def Tick(self):
        if self.explorationPolicy is not None:
            self.explorationPolicy.Tick()
        if self.noisePolicy is not None:
            self.noisePolicy.Tick()
        if self.qPolicy is not None:
            self.qPolicy.Tick()
        pass


########################################################
# Trajectory
########################################################

class PPOGameMemory:
    def __init__(self, **kwargs):

        self.gamma = kwargs.get('gamma', 0.99)

        self.mem = [[], [], [], []]

        self.states = []
        self.actions = []
        self.predicted_actions = []
        self.rewards = []
        pass

    def append(self, state, action, reward, predicted, done):
        self.states.append(state)
        self.actions.append(action)
        self.predicted_actions.append(predicted)
        self.rewards.append(reward)

        if done is True:
            self.apply_final_reward(reward)
            for i in range(len(self.states)):
                self.mem[0].append(self.states[i])
                self.mem[1].append(self.actions[i])
                self.mem[2].append(self.predicted_actions[i])
                self.mem[3].append(self.rewards[i])

            self.states = []
            self.actions = []
            self.predicted_actions = []
            self.rewards = []
        pass

    def HasBatch(self, batch_size):
        return (len(self.mem[0]) >= batch_size)

    def apply_final_reward(self, reward):
        for j in range(len(self.rewards) - 2, -1, -1):
            self.rewards[j] += self.rewards[j + 1] * self.gamma
        pass

    def GetBatch(self, batch_size):
        obs = np.array(self.mem[0])
        actions = np.array(self.mem[1])
        preds = np.array(self.mem[2])
        # preds = np.reshape(preds, (preds.shape[0], preds.shape[2]))
        rewards = np.reshape(np.array(self.mem[3]), (len(self.mem[3]), 1))
        obs, actions, preds, rewards = obs[:batch_size], actions[:batch_size], preds[:batch_size], rewards[:batch_size]
        return obs, actions, preds, rewards

    def PopBatch(self, batch_size):
        res = self.GetBatch(batch_size)
        self.mem = [[], [], [], []]
        return res


########################################################
# Agent
########################################################


class Agent:
    def __init__(self, actor, critic, policy, trajectory, memory_size, writer):
        self.critic = critic
        self.actor = actor
        self.policy = policy
        self.trajectory = trajectory
        self.memory_size = memory_size
        self.gradient_steps = 0
        self.writer = writer
        pass

    @property
    def continuous(self):
        return self.actor.continuous

    @property
    def action_size(self):
        return self.actor.action_size

    @property
    def state_size(self):
        return self.actor.state_size

    def get_action(self, p):
        return self.policy.SelectAction(p)

    def get_action_matrix(self, action):
        if self.continuous is False:
            action_matrix = np.zeros(self.action_size)
            action_matrix[action] = 1
            return action_matrix
        else:
            return action

    def predict(self, state, return_q_values=False):
        """ get input state and return action in array if return_q_values is True.
            use to check accuracy
            always use trained model to do it (no random)
            return action as number if return_q_values is False
        """
        # p = self.shared.predict_actor(state)
        # p = self.shared.actor_model.predict(state)
        p = self.actor.predict(state)
        if return_q_values is True:
            return p
        return self.get_action(p)

    def observe(self, state, action, reward, predicted_action, done):
        """ store in memory  """
        self.trajectory.append(state, action, reward, predicted_action, done)

        # self.__r = self.policy.CheckIfRandom()

        if done:
            self.policy_tick()
        pass

    def policy_tick(self):
        """
        Manual tick policy
        """
        self.policy.Tick()

    def __replay(self):
        # replay
        obs, actions, old_preds, rewards = self.trajectory.PopBatch(self.memory_size)

        pred_values = self.critic.model.predict(obs)
        advantages = rewards - pred_values
        actor_loss = self.actor.model.fit(
            [obs, advantages, old_preds],
            [actions],
            batch_size=BATCH_SIZE, shuffle=True, epochs=EPOCHS, verbose=False)
        critic_loss = self.critic.model.fit(
            [obs],
            [rewards],
            batch_size=BATCH_SIZE, shuffle=True, epochs=EPOCHS, verbose=False)
        self.writer.add_scalar('Actor loss', actor_loss.history['loss'][-1], self.gradient_steps)
        self.writer.add_scalar('Critic loss', critic_loss.history['loss'][-1], self.gradient_steps)
        self.gradient_steps += 1

    def replay(self):
        """ replay stored memory"""
        if self.trajectory.HasBatch(self.memory_size) is True:
            self.__replay()
            return
        pass

    pass    # Agent


########################################################
# Trainer
########################################################


class Trainer:
    def __init__(self, game, agent, writer):
        self.agent = agent
        self.game = game
        self.writer = writer

        self.episode = 0
        self.game.reset()
        pass

    def run_once(self, do_train, do_render):
        self.game.reset()
        rewards = []
        while not self.game.Done():
            state = self.game.state(0)

            predicted_action = self.agent.predict(state, return_q_values=True)
            action = self.agent.get_action(predicted_action)
            action_matrix = self.agent.get_action_matrix(action)

            self.game.order(0, action)
            self.game.pulse()

            reward = self.game.agent_reward(0)
            done = self.game.Done()
            rewards.append(reward)

            if do_render:
                self.game.render()

            if do_train:
                self.agent.observe(state, action_matrix, reward, predicted_action, done)
            pass
        else:
            if do_train:
                self.writer.add_scalar('Episode reward', np.array(rewards).sum(), self.episode)
                self.agent.replay()
            else:
                self.writer.add_scalar('Val episode reward', np.array(rewards).sum(), self.episode)
                print('Episode #', self.episode, 'finished with reward', round(np.array(rewards).sum()))
                self.agent.policy_tick()
            self.episode += 1
        pass

    def run(self, episodes, episodes_start_watch):
        while self.episode < episodes:

            if self.agent.policy.CheckIfRandom():
                do_train = True
                if self.episode > episodes_start_watch:
                    do_render = True
                else:
                    do_render = False
            else:
                do_train = False
                do_render = True

            self.run_once(do_train, do_render)
        pass

    pass    # Trainer

########################################################
# __main__
########################################################


def get_name(env, is_continuous):
    name = 'AllRuns/'
    if is_continuous is True:
        name += 'continous/'
    else:
        name += 'discrete/'
    name += env
    return name

# use config to save run id
class Config:
    def __init__(self, fname):
        self.fname = fname
        if os.path.exists(fname):
            with open(fname, 'r') as json_file:
                json_dict = json.load(json_file)
            self.initialize(**json_dict)
        else:
            self.initialize()
        pass

    def initialize(self, **kwargs):
        self.run = kwargs.get('run', 0)
        pass

    def save(self):
        json_str = self.toJSON()
        with open(self.fname, 'w') as outfile:
            #    json.dump(json_dict, outfile)
            outfile.write(json_str)

    @property
    def __dict__(self):
        return {
            'run': self.run
        }

    def toJSON(self):
        return json.dumps(self, default=lambda o: o.__dict__, sort_keys=True, separators=(',', ':'), indent=4)
        # return json.dumps(self, default=lambda o: o.__dict__, sort_keys=True, separators=(',', ':'))

    def initializeFromJSON(self, s: str):
        json_dict = json.loads(s)
        self.initialize(**json_dict)
        pass


def train():

    game = GymEnvironment(ENV)

    config = Config("config.json")
    config.run += 1
    config.save()
    writer = SummaryWriter(get_name(ENV + "__" + str(config.run), CONTINUOUS))

    if CONTINUOUS is True:
        print(game.get_action_space(), 'action_space', game.get_state_size(), 'state size')
        state_size = game.get_state_size()
        action_space, space_min, space_max = game.get_action_space()
        # action_size = action_space[0]
        action_size = action_space[0] * 2
        print('action_size=', action_size, 'state_size=', state_size)

        critic_model = CriticModel(state_size, action_size, batch_size=BATCH_SIZE,
                                   neurons=NEURONS,
                                   hidden_layers=NUM_LAYERS,
                                   learnig_rate=LR)
        actor_model = ActorModelContinuous(state_size, action_size, batch_size=BATCH_SIZE,
                                           neurons=NEURONS,
                                           hidden_layers=NUM_LAYERS,
                                           learnig_rate=LR, activation_last='tanh')
        policy = PPOPolicy(
            explorationPolicy=PPOExporationPolicy(interval=EPISODES_WATCH_STEP),
            qPolicy=None,
            noisePolicy=PPONoisePolicyContinuous(noise=NOISE))
        pass
    else:
        print(game.get_action_size(), 'action_size', game.get_state_size(), 'state size')
        state_size = game.get_state_size()
        action_size = game.get_action_size()

        critic_model = CriticModel(state_size, action_size, batch_size=BATCH_SIZE,
                                   neurons=NEURONS,
                                   hidden_layers=NUM_LAYERS,
                                   learnig_rate=LR)
        actor_model = ActorModelDiscrete(state_size, action_size, batch_size=BATCH_SIZE,
                                         neurons=NEURONS,
                                         hidden_layers=NUM_LAYERS,
                                         learnig_rate=LR, activation_last='softmax')
        policy = PPOPolicy(
            explorationPolicy=PPOExporationPolicy(interval=EPISODES_WATCH_STEP),
            qPolicy=GreedyQPolicy(),
            noisePolicy=PPONoisePolicyDiscrete(action_size))
        pass

    agent = Agent( actor = actor_model,
                   critic = critic_model,
                   policy = policy,
                   trajectory=PPOGameMemory(gamma=GAMMA),
                   memory_size=MEMORY_SIZE,
                   writer=writer
                   )

    tr = Trainer(game,agent,writer)
    tr.run(EPISODES,EPISODES_START_WATCH)
    pass


if __name__ == '__main__':
    train()
    pass
