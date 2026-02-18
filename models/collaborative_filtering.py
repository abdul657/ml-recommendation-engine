import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


class UserBasedCF:
    def __init__(self, k=20):
        """
        k: number of similar users to consider
        """
        self.k = k
        self.user_similarity = None
        self.user_item_matrix = None
        self.user_ids = None
        self.item_ids = None
    
    def fit(self, train_data):
        """Train the model"""
        print("Training User-Based CF...")
        
        # Create user-item matrix
        self.user_item_matrix = train_data.pivot(
            index='user_id',
            columns='item_id',
            values='rating'
        ).fillna(0)
        
        self.user_ids = self.user_item_matrix.index
        self.item_ids = self.user_item_matrix.columns
        
        # Compute user similarity
        self.user_similarity = cosine_similarity(self.user_item_matrix)
        
        print(f"Trained on {len(self.user_ids)} users and {len(self.item_ids)} items")
    
    def predict(self, user_id, item_id):
        """Predict rating for user-item pair"""
        if user_id not in self.user_ids or item_id not in self.item_ids:
            return 3.0  # fallback
        
        user_idx = self.user_ids.get_loc(user_id)
        item_idx = self.item_ids.get_loc(item_id)
        
        # Get k most similar users
        similar_users = np.argsort(self.user_similarity[user_idx])[::-1][1:self.k+1]
        
        # Get their ratings for this item
        ratings = self.user_item_matrix.values[similar_users, item_idx]
        similarities = self.user_similarity[user_idx][similar_users]
        
        # Filter out users who haven't rated this item
        mask = ratings > 0
        if not mask.any():
            return 3.0
        
        # Weighted average
        prediction = np.sum(ratings[mask] * similarities[mask]) / np.sum(similarities[mask])
        
        return prediction
    
    def recommend(self, user_id, n=10):
        """Recommend top n items for user"""
        if user_id not in self.user_ids:
            return []
        
        user_idx = self.user_ids.get_loc(user_id)
        user_ratings = self.user_item_matrix.iloc[user_idx]
        
        # Items user hasn't rated
        unrated_items = user_ratings[user_ratings == 0].index
        
        predictions = []
        for item_id in unrated_items:
            pred = self.predict(user_id, item_id)
            predictions.append((item_id, pred))
        
        predictions.sort(key=lambda x: x[1], reverse=True)
        
        return predictions[:n]


class ItemBasedCF:
    def __init__(self, k=20):
        """
        k: number of similar items to consider
        """
        self.k = k
        self.item_similarity = None
        self.user_item_matrix = None
        self.user_ids = None
        self.item_ids = None
    
    def fit(self, train_data):
        """Train the model"""
        print("Training Item-Based CF...")
        
        # Create user-item matrix
        self.user_item_matrix = train_data.pivot(
            index='user_id',
            columns='item_id',
            values='rating'
        ).fillna(0)
        
        self.user_ids = self.user_item_matrix.index
        self.item_ids = self.user_item_matrix.columns
        
        # Compute item similarity
        self.item_similarity = cosine_similarity(self.user_item_matrix.T)
        
        print(f"Trained on {len(self.user_ids)} users and {len(self.item_ids)} items")
    
    def predict(self, user_id, item_id):
        """Predict rating for user-item pair"""
        if user_id not in self.user_ids or item_id not in self.item_ids:
            return 3.0
        
        user_idx = self.user_ids.get_loc(user_id)
        item_idx = self.item_ids.get_loc(item_id)
        
        # User's rated items
        user_ratings = self.user_item_matrix.iloc[user_idx]
        rated_items_mask = user_ratings > 0
        
        if not rated_items_mask.any():
            return 3.0
        
        # Similarities for this item
        similar_items_scores = self.item_similarity[item_idx]
        
        # Filter to only items user has rated
        rated_similarities = similar_items_scores[rated_items_mask]
        rated_ratings = user_ratings[rated_items_mask]
        
        # Top k similar items
        top_k_indices = np.argsort(rated_similarities)[::-1][:self.k]
        
        # FIX: rated_similarities is a NumPy array → use [] not .iloc
        top_similarities = rated_similarities[top_k_indices]
        
        # rated_ratings IS a pandas Series → iloc is correct
        top_ratings = rated_ratings.iloc[top_k_indices]
        
        # Avoid divide-by-zero
        if np.sum(top_similarities) == 0:
            return 3.0
        
        prediction = np.sum(top_ratings * top_similarities) / np.sum(top_similarities)
        
        return prediction
    
    def recommend(self, user_id, n=10):
        """Recommend top n items for user"""
        if user_id not in self.user_ids:
            return []
        
        user_idx = self.user_ids.get_loc(user_id)
        user_ratings = self.user_item_matrix.iloc[user_idx]
        
        unrated_items = user_ratings[user_ratings == 0].index
        
        predictions = []
        for item_id in unrated_items:
            pred = self.predict(user_id, item_id)
            predictions.append((item_id, pred))
        
        predictions.sort(key=lambda x: x[1], reverse=True)
        
        return predictions[:n]

