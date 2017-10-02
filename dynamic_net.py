from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from IPython import display

from torchvision import datasets, transforms
from torch.autograd import Variable
import torch.utils.data as data_utils

from copy import deepcopy


import mnist


class DynamicNet(nn.Module):
	def __init__(self, input_size, output_size, public_hidden_size):
		super(DynamicNet, self).__init__()

		# dynamic create private network for the given network should be choosen at the time.
		# add new network when apply new task
		# when
		self.input_size = input_size
		self.output_size = output_size

		self.public_net = nn.Linear(input_size, public_hidden_size)
		# self.private_net = SubNet(input_size, output_size, private_hidden_size)
		self.out_layer = nn.Linear(public_hidden_size, output_size)

	def forward(self, x, task_num):

		h_private = h_private_list[num]
		h_public = F.relu(self.public_net(x))
		out = self.out_layer([h1_private, h_public])
		
	def add_private_net(self, private_hidden_size):
		new_private_net = SubNet(input_size, output_size, private_hidden_size)
		self.h_private_list.append(new_private_net)
		

class SubNet(nn.Module):
	def __init__(self, input_size, output_size, hidden_size):
		super(Net, self).__init__()
		self.fc1 = nn.Linear(input_size, hidden_size)
		self.fc2 = nn.Linear(input_size, hidden_size)

	def forward(self, x):
		h1 = F.relu(self.fc1(x))
		out = self.fc2(h1)


def main():

	print("This is a dynamic net file")

	# predifine parameters
	input_size = 784
	public_hidden_size = 64
	
	extra_hidden_size = 8
	output_size = 10
	num_epochs = 1
	batch_size = 64
	test_batch_size = 1000
	learning_rate = 0.1

	# fisher
	sample_batch_size = 1
	num_samples = 300


	dynamic_net_model = DynamicNet(input_size, output_size, public_hidden_size)

	# register a new private net
	# choose private net and train with public net
	#choose private net and test with public net
	

if __name__ == "__main__":

	main()