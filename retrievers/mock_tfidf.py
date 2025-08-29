
from typing import List, Dict
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import os


class TfidfBookRetriever:
	def __init__(self, csv_path: str = "data/books.csv"):
		if not os.path.exists(csv_path):
			raise FileNotFoundError(f"Nu găsesc {csv_path}.")
		self.df = pd.read_csv(csv_path)
		self.df["__text__"] = (
			self.df["title"].fillna("") + " "
			+ self.df["author"].fillna("") + " "
			+ self.df["genre"].fillna("") + " "
			+ self.df["tags"].fillna("").str.replace(";", " ") + " "
			+ self.df["description"].fillna("")
		).str.lower()
		self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_df=0.95)
		self.mat = self.vectorizer.fit_transform(self.df["__text__"])

	def query(self, q: str, k: int = 10) -> List[Dict]:
		q_vec = self.vectorizer.transform([q.lower()])
		sims = linear_kernel(q_vec, self.mat).flatten()
		top_idx = sims.argsort()[::-1][:k]
		out = []
		for i in top_idx:
			row = self.df.iloc[i]
			out.append({
				"score": float(sims[i]),
				"meta": {
					"title": row.get("title", ""),
					"author": row.get("author", ""),
					"genre": row.get("genre", ""),
					"tags": row.get("tags", ""),
					"info_url": row.get("info_url", ""),
				},
				"doc": (
					f"Title: {row.get('title','')}\n"
					f"Author: {row.get('author','')}\n"
					f"Genre: {row.get('genre','')}\n"
					f"Description: {row.get('description','')}\n"
					f"Tags: {row.get('tags','')}"
				),
			})
		return out