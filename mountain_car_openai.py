import numpy as np
from collections import deque#deque is defined collections.
import random
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import Adam
import gym

#env.observation_space.shape[0] gives the vector dimension of the state space
#env.action_space.n gives the number of possible actions


class DQN_agent:
	def __init__(self, env):
		self.env = env
		self.memory = deque(maxlen=20000) #length of the replay memory is set here
		self.learning_rate = .01 #learning rate used in the optimizer
		self.lr_decay = .95 #learning rate decay
		self.gamma = .95 #discount factor
		self.epsilon = 1.0 #exploration parameter
		self.epsilon_min = .001 #least exploration parameter, once fully attenuated
		self.decay_rate = .9 #rate at which epsilon (exploration parameter) is attenuated
		self.model = self.create_model() #create a NN for Q-learning
		self.target_model = self.create_model() #create another NN to be maintained as target
		self.batch_size = 32 #batch size of the sample from replay memory
		self.tau = .125


	def add_to_memory(self,cur_state, action, reward, next_state, done): #this function can be called to add to SARS to memory
		self.memory.append([cur_state, action, reward, next_state, done])


	def create_model(self): #Merely a function that creates a NN model
		self.model = Sequential()
		self.model.add(Dense(512, activation='relu', input_dim = self.env.observation_space.shape[0], kernel_initializer='normal'))
		self.model.add(Dropout(.3))
		self.model.add(Dense(512, activation='relu', kernel_initializer='normal'))
		self.model.add(Dense(self.env.action_space.n))
		self.model.compile(loss='mean_squared_error', optimizer=Adam(lr=self.learning_rate, decay=self.lr_decay))
		return self.model
	
	def act(self, cur_state): #act in an epsilon greedy manner
		self.epsilon = self.epsilon * self.decay_rate
		self.epsilon = max(self.epsilon, self.epsilon_min)
		if (np.random.uniform(0,1) < self.epsilon): 
			return (np.random.choice(range(self.env.action_space.n))) #pick a random action
		return np.argmax(self.model.predict(cur_state)[0]) #pick in a Q-greedy manner
		
		
	def train_using_replay(self): #train using replay memory
		if (len(self.memory) < self.batch_size):
			return
		samples = random.sample(self.memory, self.batch_size)
#np.random.choice() needs a single dimenional list as input. If anything else, then better use random.sample(list, size)
#For this, first import random.
		for sample in samples:
			state, action, reward, nxt_state, done = sample
			target = self.target_model.predict(state)
			if (done):
				target[0][action] = reward
			else: 
				target[0][action] = reward + self.gamma * max(self.target_model.predict(nxt_state)[0])
			self.model.fit(state, target, epochs=1, verbose=0)
	def train_target(self):
		weights = self.model.get_weights()
		target_weights = self.target_model.get_weights()
		for i in range(len(weights)):
			target_weights[i] = self.tau * weights[i] + target_weights[i] * (1 - self.tau)
		self.target_model.set_weights(target_weights)
	def save_model(self, file_name):
		self.target_model.save(file_name) #Save target model and weights in file with file_name.
#If one wishes to load the model from file_name into MODEL, then one can use MODEL = load_model(file_name).

if (__name__ == '__main__'):
	env = gym.make('MountainCar-v0')
	agent = DQN_agent(env)

	epochs_no = 1000 #Number of epochs, i.e., number of trials.
	T = 500 #Number of steps for termination within each epoch.
	
	for e_no in range(epochs_no):
		cur_state = env.reset().reshape(1, 2)
		won = False
		for t_idx in range(T):
			action = agent.act(cur_state)
			nxt_state, reward, done, _ = env.step(action)
			nxt_state = nxt_state.reshape(1,2)
			agent.add_to_memory(cur_state, action, reward, nxt_state, done)
			agent.train_using_replay()
			#if t_idx%10 == 0: #Uncomment if the target only updates once every 10 steps.
			agent.train_target()
			cur_state = nxt_state
			if (done and t_idx < 199):
				won = True
				print ('Success in step-{} of  trial-{}'.format(t_idx, e_no))
				agent.save_model('saved_model.h5py')
				break
		if won: 
			break
		else :
			print ('Failed to win in trial-{}'.format(e_no))

#Use model.save_weights('file_name.h5py') to merely save the weights and not the model itself.
#Use model.load_weights('file_name.h5py') to load weights into model. Once model has been fully defined and compiled.
#Saving weights and loading them is especially useful in transfer learning.
