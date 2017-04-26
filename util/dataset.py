import sqlite3

import numpy 
import torch
import torch.utils.data as data

# class to enable the dataloader
# interfaces with a sqlite db that holds the review text and scores
class ReviewsDataset(data.Dataset):
	def __init__(self, db_ptr, table_name, embeddings_dict):
		self.embeddings = embeddings_dict
		self.db = sqlite3.connect(db_ptr)
		self.table_name = table_name
		rev_ids = []
		cur = self.db.execute("SELECT ID FROM " + self.table_name)
		for k in cur:
			rev_ids.append(k[0])
		self.reviews = rev_ids

	def __getitem__(self, index):
		rev_id = self.reviews[index]
		cur = self.db.execute("SELECT RATING, REVIEW FROM " + self.table_name + " WHERE ID = "  + str(rev_id))
		score = cur[0]
		rev_text = cur[1]
		
		rev_embed = numpy.zeros(1,500,300)
		review = rev_text.split(" ")

		for i in range(500):
			rev_embed[0,i,:] = self.embeddings[review[i]]
		
		return score, rev_embed

	def __len__(self):
		return(len(self.reviews))
