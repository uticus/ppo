#!/usr/bin/env python

import os
import random
from copy import deepcopy

import gym
import numpy as np
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import RMSprop


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


class DenseDenseModel:
    def __init__(self, state_size : int, action_size : int, **kwargs):
        self.state_size = state_size
        self.action_size = action_size
        self.batch_size = kwargs.get('batch_size', 1)
        self.history_callback = kwargs.get('history_callback', [])

        self.batch_states = []
        self.batch_actions = []

        self.optimizer = kwargs.get('optimizer', RMSprop(lr=0.00025))
        self.loss = kwargs.get('loss', 'mse')
        self.metrics = kwargs.get('metrics', ['accuracy'])

        self.input_shape = (self.state_size,)
        self.neurons = kwargs.get('neurons', self.state_size*self.batch_size * 8)
        self.kernel_initializer = kwargs.get('kernel_initializer', 'glorot_uniform')
        self.activation = kwargs.get('activation', 'relu')
        self.activation_last = kwargs.get('activation_last', 'linear')

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
        ''' create and return model '''
        return Sequential([
            Dense(self.neurons, input_shape = self.input_shape, 
                kernel_initializer=self.kernel_initializer,
                activation=self.activation ),
            Dense(self.action_size, 
                kernel_initializer=self.kernel_initializer, 
                activation = self.activation_last)
            ])


    def compileModel(self, **kwargs):
        ''' compile the existing model with optimizer, loss and metrics '''
        self.optimizer = kwargs.get('optimizer', self.optimizer)
        self.loss = kwargs.get('loss', self.loss)
        self.metrics = kwargs.get('metrics', self.metrics)
        self.model.compile( optimizer=self.optimizer, loss=self.loss, metrics=self.metrics)
        pass


    def get_input_shapes(self):
        ''' return (x_shape,y_shape) '''
        return (self.batch_size, self.state_size,), (self.batch_size, self.action_size,)



    def fit_batch(self, states, actions, epochs=1, verbose=0):
        assert len(states) == self.batch_size
        assert len(actions) == self.batch_size
        x_train = np.array(states)
        x_train = np.reshape(x_train, self.x_shape)

        y_train = np.array(actions)
        y_train = np.reshape(y_train, self.y_shape)

        self.model.fit(x_train, y_train, batch_size=self.batch_size, epochs=epochs, verbose=verbose, 
            shuffle=False, callbacks=self.history_callback)
        pass

    def predict(self, state):
        states = [[state]]*self.batch_size
        x = np.array(states)
        x = np.reshape(x, self.x_shape)        
        #return self.model.predict(x, batch_size=self.batch_size).flatten()
        return self.model.predict(x, batch_size=self.batch_size)[0]

    def predict_batch(self, states):
        x = np.array(states)
        x = np.reshape(x, self.x_shape)
        return self.model.predict(x, batch_size=self.batch_size)


    def save_weights(self, fname):
        self.model.save_weights(fname)
        pass
    def load_weights(self, fname):
        if os.path.exists(fname):
            self.model.load_weights(fname)
            self.compileModel()
            return True
        else:
            return False        

########################################################
# Memory to store observations
# just based on a list version
########################################################

class GameMemory:
    def __init__(self, mem_size = 100000, **kwargs):
        self.mem_size = mem_size
        self.mem = list()
        pass

    def append(self, state, action, reward, state_new, done):
        self.mem.append( (state, action, reward, state_new, done) )
        if len(self.mem) > self.mem_size and self.mem_size > 0:
            self.mem.pop(0)
        pass

    def clear(self):
        self.mem.clear()


    def random_samples(self, batch_size):
        return random.sample(self.mem, min(batch_size, len(self.mem)) )


########################################################
# Random agent to make a random actions
########################################################

class RandomAgent:
    def __init__(self, action_size, **kwargs):
        
        self.action_size = action_size
        self.possible_actions = [i for i in range(action_size)]

        self.memory = kwargs.get('memory', None)

        self.is_random = True
        pass


    def act(self, state):
        ''' get input state and return action as number. 
            use it to predict the action during training and in the game
        '''
        return random.choice(self.possible_actions)

    def predict(self, state):
        ''' get input state and return action in array. 
            use to check accuracy
            always use trained model to do it (no random)
        '''
        action = self.act(state)
        action = [0]*self.action_size
        action[act] = 1
        return action


    def observe(self, state, action, reward, state_new, done ):
        ''' store memory '''
        if not self.memory == None:
            self.memory.append( state, action, reward, state_new, done )
        pass

    def replay(self):
        ''' replay stored memory'''
        pass

    def load_weights(self, fname):
        return False
    def save_weights(self, fname):
        return False


########################################################
# Simple Deep Q-learning Network Agent
########################################################

class DQNAgent:
    """ Deep Q-learning network agent with experinced replay.

    Agent gets a model and defines methods to be called by games

    To implement agent, we have to implement the following methods:

    - `observe`
    - `replay`

    - `act`
    - `predict`
    - `predict_batch`

    """

    def __init__(self, model, state_size : int, action_size : int, random_agent = None, **kwargs):
        self.state_size = state_size
        self.action_size = action_size

        self.model = model

        self.is_random = False

        # memory to store observations
        self.memory = kwargs.get('memory', GameMemory())

        # parameter to calculate discounted reward in Q(s,a) -> r_t + gamma * max_a Q(s_t+1, a)
        self.gamma = kwargs.get('gamma', 0.99)

        # use epsilon-greedy policy to check if we use random action or trained model prediction 
        # exponential decay is used by default to slowly decrease epsilon
        #self.policy = kwargs.get('policy', EpsExpDecayPolicy())
        self.epsilon = kwargs.get('epsilon', 1.0)
        self.epsilon_max = kwargs.get('epsilon_max', self.epsilon)
        self.epsilon_min = kwargs.get('epsilon_min', 0.01)
        self.epsilon_decay = kwargs.get('epsilon_decay', 0.999)
        self.tick = 0


        # agent to generate random actions. No random actions at all if random_agent is None
        self.random_agent = random_agent


        pass


    def act(self, state):
        ''' get input state and return action as number. 
            use it to predict the action during training and in the game
        '''
        # use epsilon-greedy policy to check if we use random action or trained model prediction 
        # epsilon-greedy policy can be replaced by other policy during init
        if (np.random.rand() <= self.epsilon) and not self.random_agent == None:
            return self.random_agent.act(state)
        # predict action 
        return self.predict(state, False)

    def predict(self, state, return_q_values=False):
        ''' get input state and return action in array if return_q_values is True. 
            use to check accuracy
            always use trained model to do it (no random)
            return action as number if return_q_values is False
        '''
        # predict action 
        action = self.model.predict(state)
        if return_q_values:
            return action
        else:
            # use policy to select an action from predicted values
            # by default it is np.argmax(action)
            #return self.policy.SelectAction(action)
            return np.argmax(action)

    def predict_batch(self, states, return_q_values=False):
        actions = self.model.predict_batch(states)
        if return_q_values:
            return actions
        #acts = [self.policy.SelectAction(action) for action in actions]
        acts = [np.argmax(action) for action in actions]
        return acts

    def observe(self, state, action, reward, state_new, done ):
        ''' store in memory  '''
        if done:
            # create zero state if done
            state_new = np.zeros(self.state_size)
        # just store observation in a memory
        self.memory.append( state, action, reward, state_new, done )

        # update epsilon-greedy or any other policy to decay parameters from time
        #self.policy.Tick()
        self.tick += 1
        self.epsilon = self.epsilon_min + \
            (self.epsilon_max - self.epsilon_min) * \
                np.exp( (self.epsilon_decay-1) * self.tick )

        pass

    def replay(self):
        ''' replay stored memory
        '''

        # get random batch from stored observations
        batch = self.memory.random_samples(self.model.batch_size)
        # replay single batch
        self.replay_batch(batch)
        pass

    def replay_batch(self, batch):
        batch_size = self.model.batch_size
        batch_len = len(batch)

        # break into different arrays
        # model expects batches with batch_size. zero last elements but ignore later
        states = np.zeros((batch_size, self.state_size))
        states_new = np.zeros((batch_size, self.state_size))
        for i in range( batch_len):
            states[i] = batch[i][0]
            states_new[i] = batch[i][3]

        # Q(s,a)
        q_values = self.model.predict_batch(states) 

        # Q(s_t+1,a)
        q_values_next = self.model.predict_batch(states_new)

        # model expects batches with batch_size. zero last elements
        x = np.zeros((batch_size, self.state_size))
        y = np.zeros((batch_size, self.action_size))


        for i in range( batch_len):
            # Q(s,a) -> r_t + gamma * max_a Q(s_t+1, a)
            
            # Compute r_t + gamma * max_a Q(s_t+1, a) and update the target targets accordingly,
            # but only for the affected output units (as given by action_batch).
            # Set discounted reward to zero for all states that were terminal.
            state, action, reward, state_new, done = batch[i]

            if done:
                # final reward
                q_values[i][action] = reward
            else:
                # 
                discounted_reward = self.gamma * np.amax(q_values_next[i])
                q_values[i][action] = reward + discounted_reward

            x[i] = state
            y[i] = q_values[i]

            pass

        # train model
        self.model.fit_batch(x, y)

        pass

    def save_weights(self, fname):
        if self.model == None:
            return False
        #self.model.save_weights(fname + ".h5")
        self.model.save_weights(fname)
        return True

    def load_weights(self, fname):
        if self.model == None:
            return False
        #if not self.model.load_weights(fname + ".h5"):
        if not self.model.load_weights(fname):
            return False
        return True

########################################################
# Trainer
########################################################

class Trainer:
    def __init__(self, game, agentIds, agents, **kwargs):
        self.game  = game
        self.agents = agents
        self.agentIds = agentIds

        assert len(self.agents) == len(self.agentIds), "different array sizes"
        pass

    class TrainedElem:
        def __init__(self):
            self.state = None
            self.action = None
            self.reward = 0
            self.state_ = None
            self.done = False
        def get(self):
            return (self.state, self.action, 
                self.reward, self.state_, self.done)


    def train(self, **kwargs):
        self.render = kwargs.get('render', True)
        random_agent = kwargs.get('random_agent', None)

        self.game.reset()
        agents_num = len(self.agents)
        score = np.array( [0]*agents_num )

        elems = [self.TrainedElem()]*agents_num

        while not self.game.Done():
            if self.render:
                self.game.render()

            # act agents
            for agentIdx in range(agents_num):
                agentId = self.agentIds[agentIdx]

                if not random_agent == None:
                    agent = random_agent
                else:
                    agent = self.agents[agentIdx]

                state = self.game.state(agentId)

                action = agent.act(state)

                self.game.order(agentId, action)

                elems[agentIdx].state = state
                elems[agentIdx].action = action


            # all orders are given - pulse the game
            self.game.pulse()

            # observe results
            for agentIdx in range( agents_num):
                agentId = self.agentIds[agentIdx]
                agent = self.agents[agentIdx]
                state = self.game.state(agentId)

                state_ = self.game.state(agentId)
                reward = self.game.agent_reward(agentId)

                elems[agentIdx].state_ = state_
                elems[agentIdx].reward = reward

                elems[agentIdx].done = self.game.Done()

                agent.observe( *elems[agentIdx].get() )
                agent.replay()

                score[agentIdx] += reward

        else:
            # end of game
            pass
        return score




    def play(self, **kwargs):
        self.render = kwargs.get('render', True)

        self.game.reset()
        agents_num = len(self.agents)
        score = np.array( [0]*agents_num )

        while not self.game.Done():
            if self.render:
                self.game.render()

            # act agents
            for agentIdx in range(agents_num):
                agentId = self.agentIds[agentIdx]

                agent = self.agents[agentIdx]

                state = self.game.state(agentId)

                action = agent.predict(state)

                self.game.order(agentId, action)

            # all orders are given - pulse the game
            self.game.pulse()

            # observe results
            for agentIdx in range( agents_num):
                agentId = self.agentIds[agentIdx]
                agent = self.agents[agentIdx]
                state = self.game.state(agentId)

                reward = self.game.agent_reward(agentId)

                score[agentIdx] += reward

        else:
            # end of game
            pass
        return score


########################################################
# __main__
########################################################


WEIGHTS_FILE_NAME='ea01cpv0.h5'


def train(reached_max_threshold = 10,load_weights=False):
    print("dafna.tests")

    print(" ")
    print(__file__)
    print(" ")
    
    print("================ train =======================")


    game = GymEnvironment('CartPole-v0')
    state_size = game.get_state_size()
    action_size = game.get_action_size()

    model = DenseDenseModel(state_size, action_size, batch_size=64)
    agent = DQNAgent(model, 
                state_size, action_size, 
                RandomAgent(action_size),
                gamma=0.99)
    trainer = Trainer(game, [0], [agent])
    if load_weights:
        agent.load_weights(WEIGHTS_FILE_NAME)

    max_score = 200
    reached_max_counter = 0

    gameId = 0
    try:
        while True:
            gameId += 1
            score = trainer.train(render=False)
            #print("[" + str(gameId) + "]  score : " + str(score) + ", eps=" + str(agent.policy.epsilon) )
            print("[" + str(gameId) + "]  score : " + str(score) + ", eps=" + str(agent.epsilon) )
            if max_score == score[0]:
                reached_max_counter += 1
                if reached_max_counter == reached_max_threshold:
                    print("Model trained" )
                    break
            else:
                reached_max_counter = 0
    except KeyboardInterrupt:
        print("KeyboardInterrupt. train batch passed" )
        pass
    agent.save_weights(WEIGHTS_FILE_NAME)

    print("================ train complete ==============")
    pass

def play(render=True):
    print("dafna.tests")

    print(" ")
    print(__file__)
    print(" ")

    print("================ play  =======================")
    
    game = GymEnvironment('CartPole-v1')
    state_size = game.get_state_size()
    action_size = game.get_action_size()

    model = DenseDenseModel(state_size, action_size, batch_size=64)
    agent = DQNAgent(model, 
                state_size, action_size, 
                #RandomAgent(action_size), #disable for play mode
                gamma=0.99)
    trainer = Trainer(game, [0], [agent])
    agent.load_weights(WEIGHTS_FILE_NAME)


    try:
        gameId = 0
        while True:
            gameId += 1
            score = trainer.play(render=render)
            print("[" + str(gameId) + "]  score : " + str(score)  )
    except KeyboardInterrupt:
        print("KeyboardInterrupt. exiting" )
        pass

    print("================ play complete  ==============")
    pass







if __name__ == "__main__":
    
    train()
    play()

    print("passed")
    input("Enter to exit")

