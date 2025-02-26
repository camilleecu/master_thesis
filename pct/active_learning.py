import numpy as np

def active_learning_iteration(learner, x_train, y_train, X_matrix, iteration=10, k=20):
    """
    Implements active learning by iteratively querying users about the most uncertain items.

    :param learner: The current PCT model.
    :param x_train: User-item interaction matrix (training set).
    :param y_train: Target ratings.
    :param X_matrix: Matrix of hidden ratings to be revealed during active learning.
    :param iteration: Number of active learning iterations.
    :param k: Number of items to query per user.
    :return: Updated x_train after active learning.
    """
    print("Starting Active Learning...")

    for i in range(iteration):
        print(f"\nActive Learning Iteration {i + 1}")

        # Predict missing ratings for all users
        predictions = {}
        for user in range(x_train.shape[0]):  # Loop through all users
            unrated_items = np.where(X_matrix[user, :] > 0)[0]  # Items not yet rated in training
            if len(unrated_items) > 0:
                scores = learner.predict(user, unrated_items)  # Predict missing ratings
                predictions[user] = scores

        # Select the most uncertain items (high variance)
        for user, scores in predictions.items():
            sorted_items = sorted(scores.items(), key=lambda x: abs(x[1] - 50))  # Items close to 50 are uncertain
            top_k_items = [item[0] for item in sorted_items[:k]]  # Select Top-K uncertain items

            # Simulate feedback by revealing actual ratings
            for item in top_k_items:
                x_train[user, item] = X_matrix[user, item]  # Move rating from hidden X to train
                X_matrix[user, item] = 0  # Mark as revealed

        print("Retraining PCT model with updated data...")
        learner.fit(x_train, y_train)  # Re-train the model

    print("Active Learning Complete!")
    return x_train  # Return the updated training set
