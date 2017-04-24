import sqlite3

import numpy as np
import torch
import torch.utils.data as data

# class to enable the dataloader
class ReviewsDataset(data.Dataset):
	def __init__(self, db_ptr, embeddings_dict):
		self.embeddings = embeddings_dict
		self.db = sqlite3.connect(db_ptr)
		
		rev_ids = ""
		self.reviews = rev_ids

	# TODO: Just grab all the IDs in the DB during init instead of requiring file	
	# TODO: Get db schema and test out returning score and review text

	def __getitem__(self, index):
		rev_id = self.reviews[index]
		score, rev_text = self.db.execute("SELECT Score, Text FROM "+self.db_name+" WHERE ID = " +str(rev_id))
		rev_embed = np.zeros(1,500,300)
		# if the data is not preprocessed in the sqlite db, could change
		review = [token for token in rev_text if token in self.embeddings.keys()]
		rev_len = len(review)
		if rev_len > 500:
			rev_len = 500
		for i in range(rev_len):
			rev_embed[1,i,:] = self.embeddings[review[i]]
		return score, rev_embed

	def __len__(self):
		return(len(self.reviews))
