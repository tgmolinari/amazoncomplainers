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
		cur = self.db.execute("SELECT REV_ID FROM " + self.table_name)
		for k in cur:
			rev_ids.append(k[0])
		self.reviews = rev_ids

	# TODO: Get db schema and test out returning score and review text
	def __getitem__(self, index):
		rev_id = self.reviews[index]
		cur = self.db.execute("SELECT Score, Text FROM " + self.table_name + " WHERE REV_ID = "  + str(rev_id))
		score = cur[0]
		rev_text = cur[1]
		rev_embed = numpy.zeros(1,500,300)
		# Assumes the text is only mildly preprocessed, could change depending on final DB
		review = [token for token in rev_text if token in self.embeddings.keys()]
		rev_len = len(review)

		if rev_len > 500:
			rev_len = 500
		for i in range(rev_len):
			rev_embed[0,i,:] = self.embeddings[review[i]]
		
		return score, rev_embed

	def __len__(self):
		return(len(self.reviews))
