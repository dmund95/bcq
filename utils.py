import numpy as np
import matplotlib.pyplot as plt
# Code based on: 
# https://github.com/openai/baselines/blob/master/baselines/deepq/replay_buffer.py

# Simple replay buffer
class ReplayBuffer(object):
	def __init__(self):
		self.storage = []

	# Expects tuples of (state, next_state, action, reward, done)
	def add(self, data):
		self.storage.append(data)

	def sample(self, batch_size):
		ind = np.random.randint(0, len(self.storage), size=batch_size)
		state, next_state, action, reward, done = [], [], [], [], []

		for i in ind: 
			s, s2, a, r, d = self.storage[i]
			state.append(np.array(s, copy=False))
			next_state.append(np.array(s2, copy=False))
			action.append(np.array(a, copy=False))
			reward.append(np.array(r, copy=False))
			done.append(np.array(d, copy=False))

		return (np.array(state), 
			np.array(next_state), 
			np.array(action), 
			np.array(reward).reshape(-1, 1), 
			np.array(done).reshape(-1, 1))

	def save(self, filename):
		np.save("./buffers/"+filename+".npy", self.storage)

	def load(self, filename):
		self.storage = np.load("./buffers/"+filename+".npy", allow_pickle=True)


# Runs policy for X episodes and returns average reward
def evaluate_policy(env, policy, eval_episodes=10):
	rewards = np.zeros(eval_episodes)
	for ii in range(eval_episodes):
		episode_reward = 0.
		obs = env.reset()
		done = False
		while not done:
			action = policy.select_action(np.array(obs))
			obs, reward, done, _ = env.step(action)
			episode_reward += reward
		rewards[ii] = episode_reward

	print ("---------------------------------------")
	print ("Evaluation over %d episodes: %f" % (eval_episodes, np.mean(rewards)))
	print ("---------------------------------------")
	return rewards


def plotter(bcq_results_file, behavioral_results_file, figName, MYDIR='./results'):
	bcq_npz = np.load(MYDIR + '/' + bcq_results_file)
	behavioral_npz = np.load(MYDIR + '/' + behavioral_results_file)

	bcq_rewards = np.asarray(bcq_npz['rewards'])
	bcq_epochs = np.asarray(bcq_npz['epochs'])
	bcq_means = np.mean(bcq_rewards, axis=1)
	bcq_std = np.std(bcq_rewards, axis=1)

	behavioral_rewards = np.asarray(behavioral_npz['rewards'])
	behavioral_epochs = np.asarray(behavioral_npz['epochs'])
	behavioral_means = np.mean(behavioral_rewards, axis=1)
	behavioral_std = np.std(behavioral_rewards, axis=1)

	plt.figure()
	plt.errorbar(bcq_epochs, bcq_means, bcq_std, linestyle='None', marker='^')
	plt.errorbar(behavioral_epochs, behavioral_means, behavioral_std, linestyle='None', marker='s')
	plt.gca().legend(('bcq', 'behavioral'))
	plt.xlabel('Timesteps')
	plt.ylabel('Average Rewards')
	plt.savefig(MYDIR + '/' + figName + '.png')

