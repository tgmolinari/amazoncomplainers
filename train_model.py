# Training script for reviews convnet 
import argparse
import pickle

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from torch.autograd import Variable
from models.baseline_cnn import ReviewsCNN
from util.dataset import ReviewsDataset


#rev_ids, embedding_dict, db_ptr
def train(model, args):
	train_loader = data.DataLoader(
		ReviewsDataset(args.db, args.table, args.embed_dict),
		batch_size = args.batch_size, shuffle = True,
		num_workers = 8, pin_memory = True)
	
	learning_rate = 0.001
	optimizer = optim.Adam([{'params':model.conv1.parameters()}, {'params':model.conv2.parameters()},
		{'params':model.lin1.parameters()}, {'params':model.lin2.parameters()}],
		lr = learning_rate)
	loss = nn.MSELoss()

	batch_ctr = 0

	for e in range(args.epochs):
		for i, score, rev_emb in enumerate(train_loader):
			score_var = Variable(score).type(args.dtype)
			pred_score = model.forward(Variable(rev_emb).type(args.dtype))

			optimzer.zero_grad()
			batch_loss = loss(pred_score, score_var)
			batch_loss.backward()
			optimzer.step()
			batch_ctr += 1

			if batch_ctr % 10000 == 0 and batch_ctr > 0:
				pickle.dump(model.state_dict(), open(args.save_name + '.p', 'wb'))



parser = argparse.ArgumentParser(description='trainer for embedding reviews')
parser.add_argument('--batch-size', type=int, default=32,
    help='batch size (default: 32)')
parser.add_argument('--db', type=str, default = 'reviews.db',
    help='db that houses the review text')
parser.add_argument('--table', type=str, default = 'REVIEWS',
    help='table in db we need')
parser.add_argument('--embed-dict', type=str, default = 'e_dict.p',
    help='dictionary that houses the embeddings')
parser.add_argument('--save-name', type=str, default='rev_cnn',
    help='file name to save model params')
parser.add_argument('--load-name', type=str,
    help='file name to load model params dict')
parser.add_argument('--gpu', action="store_true",
    help='attempt to use gpu')
parser.add_argument('--epochs', type=int, default=9999999,
    help='number of epochs to train, defaults to run indefinitely')

if __name__ == '__main__':
	args = parser.parse_args()
	args.dtype = torch.cuda.FloatTensor if torch.cuda.is_available() and args.gpu else torch.FloatTensor

	model = ReviewsCNN(args.dtype)
	if args.load_name is not None:
		model.load_state_dict(pickle.load(open(args.load_name + '.p', 'rb')))

	train(model, args)