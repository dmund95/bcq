import gym
import numpy as np
import torch
import argparse
import os

from utils import *
import DDPG
import BCQ


if __name__ == "__main__":
	
	parser = argparse.ArgumentParser()
	parser.add_argument("--test_name", default="Original")				# Specific Name for AlgoRun
	parser.add_argument("--env_name", default="Pendulum-v0")			# OpenAI gym environment name
	parser.add_argument("--seed", default=0, type=int)					# Sets Gym, PyTorch and Numpy seeds
	parser.add_argument("--buffer_type", default="Robust")				# Prepends name to filename.
	parser.add_argument("--eval_freq", default=5e3, type=float)			# How often (time steps) we evaluate
	parser.add_argument("--max_timesteps", default=1e6, type=float)		# Max time steps to run environment for
	args = parser.parse_args()

	file_name = "%s_BCQ_%s_%s" % (args.test_name, args.env_name, str(args.seed))
	buffer_name = "%s_%s_%s_%s" % (args.test_name, args.buffer_type, args.env_name, str(args.seed))
	print ("---------------------------------------")
	print ("Settings: " + file_name)
	print ("---------------------------------------")

	if not os.path.exists("./results"):
		os.makedirs("./results")

	env = gym.make(args.env_name)

	env.seed(args.seed)
	torch.manual_seed(args.seed)
	np.random.seed(args.seed)
	
	state_dim = env.observation_space.shape[0]
	action_dim = env.action_space.shape[0] 
	max_action = float(env.action_space.high[0])

	# Initialize policy
	policy = BCQ.BCQ(state_dim, action_dim, max_action)

	if os.path.exists("./results/%s_actor.pth" % (file_name)):
		policy.load(file_name, './results')

	# Load buffer
	replay_buffer = ReplayBuffer()
	replay_buffer.load(buffer_name)
	
	evaluations = []
	epochs = []

	episode_num = 0
	done = True 

	training_iters = 0
	while training_iters < args.max_timesteps: 
		pol_vals = policy.train(replay_buffer, iterations=int(args.eval_freq))

		evaluations.append(evaluate_policy(env, policy))
		epochs.append(training_iters)

		np.savez("./results/" + file_name, rewards=evaluations, epochs=epochs)

		policy.save(file_name, './results')

		training_iters += args.eval_freq
		print ("Training iterations: " + str(training_iters))