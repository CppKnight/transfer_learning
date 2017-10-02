

from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from scipy import misc
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from IPython import display

from torchvision import datasets, transforms
from torch.autograd import Variable
import torch.utils.data as data_utils

from copy import deepcopy


class Net(nn.Module):
	def __init__(self, input_size, output_size, hidden_size):
		super(Net, self).__init__()

		self.input_size = input_size
		self.hidden_size = hidden_size
		self.output_size = output_size
		self.fc1 = nn.Linear(input_size, hidden_size)
		self.fc2 = nn.Linear(hidden_size, output_size)

	def forward(self, x):
		h1 = F.relu(self.fc1(x))
		out = self.fc2(h1)
		return out

	def set_loss(self, learning_rate):
		self.criterion = nn.CrossEntropyLoss()
		self.optimizer = torch.optim.SGD(self.parameters(), lr=learning_rate)

	def set_ewc_loss(self):

		self.optimizer = torch.optim.SGD(self.parameters(), lr=learning_rate)

	def get_ewc_loss(self, output, target, lam):

		self.var_list = [self.fc1.weight, self.fc1.bias, self.fc2.weight, self.fc2.bias]
		self.ewc_loss = self.criterion(output, target)

		for v in range(len(self.var_list)):
			self.ewc_loss += (lam/2) * torch.sum(torch.mul(self.F_accum_var[v], torch.mul(self.var_list[v] - self.star_vars[v], self.var_list[v] - self.star_vars[v])))
		return self.ewc_loss

	def star(self):
		self.star_vars = deepcopy(self.var_list)

	def restore(self):
		if hasattr(self, "star_vars"):
			for v in range(len(self.var_list)):
				self.var_list[v].assign(self.star_vars[v])

	def compute_fisher(self, sample_loader):
		self.F_accum = []
		self.var_list = [self.fc1.weight, self.fc1.bias, self.fc2.weight, self.fc2.bias]

		for v in range(len(self.var_list)):
			self.F_accum.append(np.zeros(self.var_list[v].size()))

		# wait = input("PRESS ENTER TO CONTINUE.")

		softmax = nn.Softmax()

		self.eval()

		F_prev = deepcopy(self.F_accum)
		mean_diffs = np.zeros(0)

		display = False
		num_samples = 1000

		for n, (data, target) in enumerate(sample_loader):

			if n == num_samples:
				break

			data = Variable(data.view(-1, self.input_size), requires_grad=True)
			target = Variable(target)

			self.optimizer.zero_grad()
			output = self(data)

			probs = softmax(output)

			# index_list[class_index[0][0]] = 1


			# print("class_index", class_index)
			# print("")

			prob_log = torch.log(probs)
			# print("log", prob_log)
			
			# loss = model.criterion(output, target)
			# loss.backward()

			class_index = torch.multinomial(probs, 1).data.cpu().numpy()
			class_index = class_index[0][0]
			class_index = np.asscalar(class_index)
			dL_dy = torch.zeros(1, self.output_size)
			dL_dy.index_fill_(1, torch.LongTensor([class_index]), 1)
			
			prob_log.backward(dL_dy, retain_variables=True)

			# self.var_list = [self.fc1.weight, self.fc1.bias, self.fc2.weight, self.fc2.bias]
			ders = []
			for var in self.var_list:
				ders.append(var.grad)
			
			for v in range(len(self.var_list)):
				self.F_accum[v] += np.square(ders[v].data.cpu().numpy())


			if display:
				disp_freq = 10
				if n % disp_freq == 0 and n > 0:
				# recording mean diffs of F
					F_diff = 0
					for v in range(len(self.F_accum)):
						F_diff += np.sum(np.absolute(self.F_accum[v]/(n+1) - F_prev[v]))
					mean_diff = np.mean(F_diff)
					mean_diffs = np.append(mean_diffs, mean_diff)
					for v in range(len(self.F_accum)):
						F_prev[v] = self.F_accum[v]/(n+1)
					plt.plot(range(disp_freq+1, n+2, disp_freq), mean_diffs)

		if display:
			plt.xlabel("Number of samples")
			plt.ylabel("Mean absolute Fisher difference")
			plt.show()

		if display:
			print(self.F_accum)
			for v in range(len(self.var_list)):
				self.F_accum[v] /= num_samples

			self.F_accum_var = []
			for v in range(len(self.var_list)):
				self.F_accum_var.append(Variable(torch.from_numpy(self.F_accum[v]).float()))

			# show fisher info from in order for the fc1 layer to fc2 layer
			fisher_weight_l1 = self.F_accum[0]
			fisher_bias_l1 = self.F_accum[1]
			fisher_weight_l2 = self.F_accum[2]
			fisher_bias_l2 = self.F_accum[3]


			# # dropout
			# percentage = 0.3
			# for accum in self.F_accum:
			# 	sort_index = np.argsort(accum, axis=None)
			# 	tupple_index = np.unravel_index(sort_index, accum.shape)
			# 	dropout_num = np.floor(percentage * accum.size)
			# 	for v in range(0, dropout_num):
			# 		dropout_index = [tuple_index[0][-v], tuple_index[1][-v]]

			# show the parameters
			plt.figure()
			plt.plot(fisher_weight_l1.flatten())
			# plt.plot(np.sort(fisher_weight_l1, axis=None))
			plt.xlabel("weight 1 id")
			plt.ylabel("fisher info")

			# plt.figure()
			# plt.plot(fisher_bias_l1.flatten())
			# # plt.plot(np.sort(fisher_bias_l1, axis=None))
			# plt.xlabel("bias 1 id")
			# plt.ylabel("fisher info")

			# plt.figure()
			# plt.plot(np.sort(fisher_weight_l2, axis=None))
			# plt.xlabel("weight 2 id")
			# plt.ylabel("fisher info")

			# plt.figure()
			# plt.plot(np.sort(fisher_bias_l2, axis=None))
			# plt.xlabel("bias 2 id")
			# plt.ylabel("fisher info")
			plt.show()
					

		# wait = input("PRESS ENTER TO CONINUE")

def train(epoch, model, train_loader, args):
	model.train()
	for batch_idx, (data, target) in enumerate(train_loader):
		
		# print("type of data ", data.size())
		# print("type of data ", data.view(-1, input_size).size())
		data = Variable(data.view(-1, model.input_size))
		# print("type of data ", type(data))
		target = Variable(target)

		model.optimizer.zero_grad()
		output = model(data)
		# print(output.size())
		# input("a")

		loss = model.criterion(output, target)
		loss.backward()
		model.optimizer.step()

		if batch_idx % args.log_interval == 0:
			print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
				epoch, batch_idx * len(data), len(train_loader.dataset),
				100. * batch_idx / len(train_loader), loss.data[0]))

		# wait = input("PRESS ENTER")

def ewc_train(epoch, model, train_loader, args, lam):
	model.train()
	for batch_idx, (data, target) in enumerate(train_loader):
		data = Variable(data.view(-1, model.input_size))
		target = Variable(target)

		model.optimizer.zero_grad()
		output = model(data)

		loss = model.get_ewc_loss(output, target, lam)
		loss_sgd = model.criterion(output, target)

		# print("loss", loss)
		# print("loss sgd", loss_sgd)

		loss.backward()
		model.optimizer.step()

		if batch_idx % args.log_interval == 0:
			print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
				epoch, batch_idx * len(data), len(train_loader.dataset),
				100. * batch_idx / len(train_loader), loss.data[0]))

def test(model, test_loader, args):

	model.eval()
	test_loss = 0
	correct = 0
	num_test = 10

	wrong_image = []
	correct_label = []
	predict_label = []
	for batch_idx, (data, target) in enumerate(test_loader):
		if batch_idx == num_test:
			break

		print("test batch", batch_idx)
		data = Variable(data.view(-1, model.input_size), volatile=True)
		target = Variable(target)
		output = model(data)

		test_loss += model.criterion(output, target).data[0]
		pred = output.data.max(1)[1]
		correct += pred.eq(target.data).cpu().sum()

		if (pred.eq(target.data).cpu().sum() != 1):
			wrong_image.append(data)
			correct_label.append(target.data.numpy())
			predict_label.append(pred.numpy())

	test_loss /= num_test * args.test_batch_size
	print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
		test_loss, correct, num_test * args.test_batch_size,
		100. * correct / num_test /args.test_batch_size))


	# correct_label = np.asarray(correct_label).flatten()
	# predict_label = np.asarray(predict_label).flatten()

	# statistic = np.zeros([10,10])
	# for i in range(correct_label.size):
	# 	statistic[correct_label[i]][predict_label[i]] += 1

	# plt.figure()
	# label = []
	# for i in range(10):
	# 	label_i, = plt.plot(statistic[i]/np.sum(statistic[i]), label=i)
	# 	label.append(label_i)
	# plt.legend(label)
	# plt.show()



def args_define():
	input_size = 784
	hidden_size = 64
	extra_hidden_size = 8
	output_size = 10
	num_epochs = 1
	batch_size = 64
	test_batch_size = 1000
	learning_rate = 0.1

	# fisher
	sample_batch_size = 1
	num_samples = 300

	# Training settings
	parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
	parser.add_argument('--batch-size', type=int, default=batch_size, metavar='N',
	                    help='input batch size for training (default: 64)')
	parser.add_argument('--test-batch-size', type=int, default=test_batch_size, metavar='N',
	                    help='input batch size for testing (default: 1000)')
	parser.add_argument('--epochs', type=int, default=num_epochs, metavar='N',
	                    help='number of epochs to train (default: 10)')
	parser.add_argument('--lr', type=float, default=learning_rate, metavar='LR',
	                    help='learning rate (default: 0.01)')
	parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
	                    help='SGD momentum (default: 0.5)')
	parser.add_argument('--no-cuda', action='store_true', default=False,
	                    help='disables CUDA training')
	parser.add_argument('--seed', type=int, default=1, metavar='S',
	                    help='random seed (default: 1)')
	parser.add_argument('--log-interval', type=int, default=10, metavar='N',
	                    help='how many batches to wait before logging training status')
	parser.add_argument('--sample-batch-size', type=int, default=sample_batch_size, metavar='N',
	                    help='input batch size for computing fisher')
	args = parser.parse_args()
	args.cuda = not args.no_cuda and torch.cuda.is_available()

	torch.manual_seed(args.seed)
	if args.cuda:
	    torch.cuda.manual_seed(args.seed)

	return parser.parse_args()


def main():
	input_size = 784
	hidden_size = 64
	extra_hidden_size = 8
	output_size = 10
	num_epochs = 1
	batch_size = 64
	test_batch_size = 1000
	learning_rate = 0.1

	# fisher
	sample_batch_size = 1
	num_samples = 300

	args = args_define()
	args.cuda = not args.no_cuda and torch.cuda.is_available()

	torch.manual_seed(args.seed)
	if args.cuda:
	    torch.cuda.manual_seed(args.seed)

	kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

	def create_dataset(dataset):
		tensor_feature = []
		tensor_targets = []
		perm_idx = torch.randperm(input_size)
		for batch_idx, (data, target) in enumerate(dataset):
			# if batch_idx > 10000:
			# 	break
			data = data.view(input_size)
			# print(data)

			data = data[perm_idx]
			data = data.view(1, 28, 28)

			# print(data)
			tensor_feature.append(data)
			tensor_targets.append(target)

			# wait = input("PRESS ENTER")
		temp = []
		for i in tensor_feature:
			temp.append(i.numpy())

		# for i in temp:
		# 	print(i)
		# 	print(i.shape)

		tensor_feature = torch.stack([torch.Tensor(i) for i in temp])
		tensor_targets = torch.LongTensor(tensor_targets)
		
		my_dataset = data_utils.TensorDataset(tensor_feature, tensor_targets)

		return my_dataset

	train_dataset = datasets.MNIST('../data', download=True,
						transform=transforms.Compose([
	                       transforms.ToTensor(),
	                       # transforms.Normalize((0.1307,), (0.3081,))
	                   ]))

	# train_dataset_2 = create_dataset(train_dataset)
	# train_dataset_3 = create_dataset(train_dataset)

	# first dataset
	train_loader_1 = torch.utils.data.DataLoader(
		train_dataset,
		batch_size=args.batch_size, shuffle=True, **kwargs)

	test_loader_1 = torch.utils.data.DataLoader(
		train_dataset,
		batch_size=args.test_batch_size, shuffle=True, **kwargs)
	sample_loader_1 = torch.utils.data.DataLoader(
		train_dataset,
		batch_size=args.sample_batch_size, shuffle=True, **kwargs)

	# second dataset
	train_dataset_2 = create_dataset(train_dataset)
	train_loader_2 = torch.utils.data.DataLoader(
		train_dataset_2,
		batch_size=args.batch_size, shuffle=True, **kwargs)
	test_loader_2 = torch.utils.data.DataLoader(
		train_dataset_2,
		batch_size=args.test_batch_size, shuffle=True, **kwargs)
	sample_loader_2 = torch.utils.data.DataLoader(
		train_dataset_2,
		batch_size=args.sample_batch_size, shuffle=True, **kwargs)

	model = Net(input_size, output_size, hidden_size)
	model.set_loss(learning_rate)

	train_loader = train_loader_1
	test_loader = test_loader_1
	sample_loader = sample_loader_1

	# model.compute_fisher(sample_loader)

	for epoch in range(num_epochs):
		train(epoch, model, train_loader, args)
		# var_list = [model.fc1.weight, model.fc1.bias, model.fc2.weight, model.fc2.bias]
		# for v in var_list:
		# 	v.index_fill_(0, torch.LongTensor([0]), 1234)
		# print(var_list)
		test(model, test_loader, args)

	# model.compute_fisher(sample_loader)
	# model.star()

	train_loader = train_loader_2
	test_loader = test_loader_2
	sample_loader = sample_loader_2

	# model_sgd = deepcopy(model)


	# for epoch in range(num_epochs):

	# 	ewc_train(epoch, model, train_loader_2, args, lam=15)
	# 	ewc_train(epoch, model_sgd, train_loader_2, args, lam=0)

	# model.star()

	# plt.show()

	test(model, test_loader, args)
	# test(model_sgd, test_loader_1, args)

if __name__ == "__main__":
	main()
