import pandas as pd
import scipy.sparse as sp
import numpy as np
# from skopt import BayesSearchCV
# from sklearn.model_selection import GridSearchCV

def threshold_interactions_df(df, row_name, col_name, row_min, col_min):
    """
    Desc:
        Using this function, we can consider user and items with more specified number of interactions.
        Then we know than we are using an interaction matrix without user and item cold-start problem
    ------
    Input:
        df: the dataframe which contains the interactions (df)
        row_name: column name referring to user (str)
        col_nam: column name referring to item (str)
        row_min: the treshold for number of interactions of users (int)
        col_min: the treshold for number of interactions of items (int)
    ------
    Output:
        dataframe in which users and items have more that thresholds interactions (df)
    """
    n_rows = df[row_name].unique().shape[0]
    n_cols = df[col_name].unique().shape[0]
    sparsity = float(df.shape[0]) / float(n_rows*n_cols) * 100
    print('Starting interactions info')
    print('Number of rows: {}'.format(n_rows))
    print('Number of cols: {}'.format(n_cols))
    print('Sparsity: {:4.3f}%'.format(sparsity))

    done = False
    while not done:
        starting_shape = df.shape[0]
        col_counts = df.groupby(row_name)[col_name].count()
        df = df[~df[row_name].isin(col_counts[col_counts < col_min].index.tolist())]
        row_counts = df.groupby(col_name)[row_name].count()
        df = df[~df[col_name].isin(row_counts[row_counts < row_min].index.tolist())]
        ending_shape = df.shape[0]
        if starting_shape == ending_shape:
            done = True

    n_rows = df[row_name].unique().shape[0]
    n_cols = df[col_name].unique().shape[0]
    sparsity = float(df.shape[0]) / float(n_rows*n_cols) * 100
    print('Ending interactions info')
    print('Number of rows: {}'.format(n_rows))
    print('Number of columns: {}'.format(n_cols))
    print('Sparsity: {:4.3f}%'.format(sparsity))
    return df

def threshold_interactions_df_plus(
    df,
    user_col='user_id',
    item_col='item_id',
    artist_col='artist_id',
    genre_col='genre_ids',  
    min_items_per_user=100,
    min_artists_per_user=20,
    min_genres_per_user=10,
    min_users_per_item=100,
    verbose=True
):
    """
    Filters a user-item interaction dataframe by enforcing minimum thresholds on:
    1. Number of items per user
    2. Number of users per item
    3. Number of distinct artists per user
    4. Number of distinct genres per user

    Parameters
    ----------
    df : pd.DataFrame
        The interaction dataframe containing user_id, item_id, artist_id, and genre_id(s).
    user_col : str
        Column name for user ID.
    item_col : str
        Column name for item ID.
    artist_col : str
        Column name for artist ID.
    genre_col : str
        Column name for genre ID (should be a list or allow explode).
    min_items_per_user : int
        Minimum number of items a user must have interacted with.
    min_artists_per_user : int
        Minimum number of unique artists a user must have interacted with.
    min_genres_per_user : int
        Minimum number of unique genres a user must have interacted with.
    min_users_per_item : int
        Minimum number of users that must have interacted with an item.
    verbose : bool
        Whether to print filtering and sparsity information.

    Returns
    -------
    pd.DataFrame
        The filtered interaction dataframe.
    """

    df_filtered = df.copy()

    # Initial sparsity
    n_users = df_filtered[user_col].nunique()
    n_items = df_filtered[item_col].nunique()
    sparsity = 100 * df_filtered.shape[0] / (n_users * n_items)
    if verbose:
        print(f"Initial: users = {n_users}, items = {n_items}, sparsity = {sparsity:.4f}%")

    # ---------- Step 1: Filter by items per user and users per item ----------
    done = False
    while not done:
        starting_shape = df_filtered.shape[0]

        user_counts = df_filtered.groupby(user_col)[item_col].count()
        df_filtered = df_filtered[df_filtered[user_col].isin(user_counts[user_counts >= min_items_per_user].index)]

        item_counts = df_filtered.groupby(item_col)[user_col].count()
        df_filtered = df_filtered[df_filtered[item_col].isin(item_counts[item_counts >= min_users_per_item].index)]

        if df_filtered.shape[0] == starting_shape:
            done = True

    n_users = df_filtered[user_col].nunique()
    n_items = df_filtered[item_col].nunique()
    sparsity = 100 * df_filtered.shape[0] / (n_users * n_items)
    if verbose:
        print(f"After item filtering: users = {n_users}, items = {n_items}, sparsity = {sparsity:.4f}%")

    # ---------- Step 2: Filter by number of unique artists per user ----------
    df_artist_valid = df_filtered[df_filtered[artist_col] != 0]
    artist_per_user = df_artist_valid.groupby(user_col)[artist_col].nunique()

    # ---------- Step 3: Filter by number of unique genres per user ----------
    df_genre_exploded = df_filtered.explode(genre_col)
    df_genre_exploded = df_genre_exploded[df_genre_exploded[genre_col] != 0]
    genre_per_user = df_genre_exploded.groupby(user_col)[genre_col].nunique()

    # ---------- Step 4: User intersection ----------
    valid_users = set(artist_per_user[artist_per_user > min_artists_per_user].index) & \
                  set(genre_per_user[genre_per_user > min_genres_per_user].index)

    final_df = df_filtered[df_filtered[user_col].isin(valid_users)]

    n_users = final_df[user_col].nunique()
    n_items = final_df[item_col].nunique()
    sparsity = 100 * final_df.shape[0] / (n_users * n_items)
    if verbose:
        n_artists = final_df[artist_col].nunique()
        n_genres = final_df.explode(genre_col)[genre_col].nunique()
        print(f"Final: users = {n_users}, items = {n_items}, sparsity = {sparsity:.4f}%")
        print(f"unique artists = {n_artists}, unique genres = {n_genres}")
       

    return final_df

def train_test_split(interactions, split_count, fraction=None):
    """
    Desc:
        Using this function, we split avaialble data to train and test set.
    ------
    Input:
        interactions : interaction between users and streams (scipy.sparse matrix)
        split_count : number of interactions per user to move from training to test set (int)
        fraction : fraction of users to split their interactions train/test. If None, then all users (float)
    ------
    Output:
        train_set (scipy.sparse matrix)
        test_set (scipy.sparse matrix)
        user_index
    """
    train = interactions.copy().tocoo()
    test = sp.lil_matrix(train.shape)

    if fraction:
        try:
            user_index = np.random.choice(
                np.where(np.bincount(train.row) >= split_count * 2)[0],
                replace=False,
                size=np.int64(np.floor(fraction * train.shape[0]))
            ).tolist()
        except:
            print(('Not enough users with > {} '
                  'interactions for fraction of {}')\
                  .format(2*split_count, fraction))
            raise
    else:
        user_index = range(train.shape[0])

    train = train.tolil()

    for user in user_index:
        test_interactions = np.random.choice(interactions.getrow(user).indices,
                                        size=split_count,
                                        replace=False)
        train[user, test_interactions] = 0.
        test[user, test_interactions] = interactions[user, test_interactions]

    assert(train.multiply(test).nnz == 0)
    return train.tocsr(), test.tocsr(), user_index

def get_df_matrix_mappings(df, row_name, col_name):
    """
    Desc:
        Using this function, we are able to get mappings between original indexes and new (reset) indexes
    ------
    Input:
        df: the dataframe which contains the interactions (df)
        row_name: column name referring to user_id (str)
        col_nam: column name referring to stream_slug (str)
    ------
    Output:
        rid_to_idx: a dictionary contains mapping between real row ids and new indexes (dict)
        idx_to_rid: a dictionary contains mapping between new indexes and real row ids (dict)
        cid_to_idx: a dictionary contains mapping between real column ids and new indexes (dict)
        idx_to_cid: a dictionary contains mapping between new indexes and real column ids (dict)
    """

    rid_to_idx = {}
    idx_to_rid = {}
    for (idx, rid) in enumerate(df[row_name].unique().tolist()):
        rid_to_idx[rid] = idx
        idx_to_rid[idx] = rid

    cid_to_idx = {}
    idx_to_cid = {}
    for (idx, cid) in enumerate(df[col_name].unique().tolist()):
        cid_to_idx[cid] = idx
        idx_to_cid[idx] = cid

    return rid_to_idx, idx_to_rid, cid_to_idx, idx_to_cid


def df_to_matrix(df, row_name, col_name,value):
    """
    Desc:
        Using this function, we transfrom the interaction matrix to scipy.sparse matrix
    ------
    Input:
        df: the dataframe which contains the interactions (df)
        row_name: column name referring to user_id (str)
        col_nam: column name referring to stream_slug (str)
        value: the value of interaction between row and column
    ------
    Output:
        interactions: Sparse matrix contains user and streams interactions (sparse csr)
        rid_to_idx: a dictionary contains mapping between real row ids and new indexes (dict)
        idx_to_rid: a dictionary contains mapping between new indexes and real row ids (dict)
        cid_to_idx: a dictionary contains mapping between real column ids and new indexes (dict)
        idx_to_cid: a dictionary contains mapping between new indexes and real column ids (dict)
    """


    rid_to_idx, idx_to_rid,cid_to_idx, idx_to_cid = get_df_matrix_mappings(df,row_name,col_name)

    def map_ids(row, mapper):
        return mapper[row]

    I = df[row_name].apply(map_ids, args=[rid_to_idx]).to_numpy()
    J = df[col_name].apply(map_ids, args=[cid_to_idx]).to_numpy()
    V = df[value]
    interactions = sp.coo_matrix((V, (I, J)), dtype=np.float64)
    interactions = interactions.tocsr()
    return interactions, rid_to_idx, idx_to_rid, cid_to_idx, idx_to_cid

def matrix_to_df(x,r,c):
    d = []
    cx = x.tocoo()
    for i,j,v in zip(cx.row, cx.col, cx.data):
        d.append({'user_id':r[i],'item_id':c[j],'rating':v})
    return pd.DataFrame.from_dict(d)

def matrix_to_df_2(x,r,c):
    d = []
    cx = x.tocoo()
    for i,j,v in zip(cx.row, cx.col, cx.data):
        d.append({'user_id':r[i],'item_id':c[j],'rating':v})

    for i,j in enumerate(np.asarray(cx.sum(0))[0]):
        if j ==0:
            d.append({'user_id':r[0],'item_id':c[i],'rating':0})
    return pd.DataFrame.from_dict(d)

def matrix_to_full_df(sparse_matrix, idx_to_rid, idx_to_cid):
    """
    Convert sparse matrix to full DataFrame with original user_id / item_id
    """
    dense_array = sparse_matrix.toarray()
    user_ids = [idx_to_rid[i] for i in range(dense_array.shape[0])]
    item_ids = [idx_to_cid[i] for i in range(dense_array.shape[1])]
    return pd.DataFrame(dense_array, index=user_ids, columns=item_ids)


def set_intersection(a,b):
    return list(set(a).intersection(set(b)))

def get_0_and_p_index(data):
    num_users,num_items=data.shape
    user_nonzero = []
    user_zero = []
    for i in range(data[:,0].shape[0]):
       p_idxes = data[i,:].nonzero()[1]
       j_idx = np.where(data.A[i]==0)[0]
       user_nonzero.append(p_idxes)
       user_zero.append(j_idx)
    return user_nonzero,user_zero

def set_diff(a,b):
    return list(set(a)-set(b))

def matrix_completion(train_set,trained_model,model_name):
    if model_name == "SLIM" or model_name == "EASE":
        matrix = np.matmul(train_set.A,trained_model.W_sparse.A)
    cx = train_set.tocoo()
    for i,j,v in zip(cx.row, cx.col, cx.data):
        matrix[i,j]=v
    return matrix

def Bayes_tune(X,Y,model,params,niter=10,c_v=5,jobs=-1,score='neg_mean_squared_error'):

    reg_bay = BayesSearchCV(estimator=model,
                    search_spaces=params,
                    n_iter=niter,
                    cv=c_v,
                    n_jobs=jobs,
                    scoring=score,
                    verbose=True)
    reg_bay.fit(X, Y)
    return reg_bay, reg_bay.best_params_

def Grid_tune(X,Y,model,params,c_v=5,jobs=-1,score='neg_mean_squared_error'):
    reg_bay = GridSearchCV(estimator=model,
                    param_grid=params,
                    cv=c_v,
                    n_jobs=jobs,
                    scoring=score,
                    verbose=True)
    reg_bay.fit(X, Y)
    return reg_bay, reg_bay.best_params_
