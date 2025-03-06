import pandas as pd
import numpy as np
from pct.tree.heuristic.Heuristic import Heuristic5
from pct.tree.heuristic.NumericHeuristic import NumericHeuristic5
from pct.tree.splitter.splitter import Splitter
from pct.tree.tree import Tree
from pct.tree.ftest.ftest import FTest



from sklearn.preprocessing import LabelEncoder

# Load the u.data dataset
u_data = pd.read_csv('ml-100k/u.data', sep='\t', header=None, names=['user_id', 'item_id', 'rating', 'timestamp'])

# Perform label encoding on user_id and item_id
user_encoder = LabelEncoder()
item_encoder = LabelEncoder()

u_data['user_id'] = user_encoder.fit_transform(u_data['user_id'])
u_data['item_id'] = item_encoder.fit_transform(u_data['item_id'])

# Create the rating matrix
n_users = u_data['user_id'].nunique()
n_items = u_data['item_id'].nunique()

# Define the threshold
THRESHOLD = 3  # Ratings above this are "Lovers", below are "Haters"

# Create a user-item matrix with users as rows and items as columns, and fill missing values with 0
rating_matrix = u_data.pivot(index='user_id', columns='item_id', values='rating')
rating_matrix.index = u_data['user_id'].unique()  # Ensure user_id alignment
rating_matrix.columns = u_data['item_id'].unique()  # Ensure item_id alignment


# Apply threshold classification to the rating matrix (after thresholding)
rating_matrix_thresholded = np.where(rating_matrix > THRESHOLD, 1, np.where(rating_matrix > 0, -1, 0))

# Convert numpy array to pandas DataFrame for easier inspection
rating_matrix_thresholded_df = pd.DataFrame(rating_matrix_thresholded)


# Create rI and rU indexes for training set (R)
# rI: item to user ratings (dictionary of items with lists of user ratings)
rI = {}
for _, row in u_data.iterrows():
    item_id = row['item_id']
    user_id = row['user_id']
    rating = row['rating']
    if item_id not in rI:
        rI[item_id] = []
    rI[item_id].append((user_id, rating))

# rU: user to item ratings (dictionary of users with lists of item ratings)
rU = {}
for _, row in u_data.iterrows():
    user_id = row['user_id']
    item_id = row['item_id']
    rating = row['rating']
    if user_id not in rU:
        rU[user_id] = []
    rU[user_id].append((item_id, rating))



tree = Tree(min_instances=6)
tree.fit(rating_matrix, u_data['rating'], target_weights=None, rI=rI, rU=rU)


