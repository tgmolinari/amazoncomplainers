import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


# Our baseline Review CNN, inspired by Zhang and LeCun (2015?)
# Torch dtype needed in case of (shudders) CPU use of the net

class ReviewsCNN(nn.Module):
	def __init__(self, dtype, embed_dims = 300):
		super(ReviewsCNN, self).__init__()
		
		# window size of 3 words
		# activation functions will be called in the forward pass
		self.conv1 = nn.Conv1d(3, 1, 1, stride = 1)
		self.conv2 = nn.Conv1d(3, 1, 1, stride = 1)
		self.lin1 = nn.Linear(55,25)
		self.lin2 = nn.Linear(25,1)

		self.type(dtype)

	def forward(self, embeds):
		x = self.conv1(embeds)
		x = F.leaky_relu(x)
		x = self.conv2(x)
		x = F.leaky_relu(x)
		x = self.lin1(x)
		x = F.leaky_relu(x)
		x = self.lin2(x)
		x = F.sigmoid(x)

		return x*5
				