import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from pct.tree.heuristic.Heuristic import Heuristic5
from pct.tree.heuristic.NumericHeuristic_pair import NumericHeuristic5


# Global item_type_map is expected to be defined externally in the notebook
item_type_map = {}

def get_item_type(item_id):
    return item_type_map.get(item_id, 'unknown')
    
class Splitter:
    def __init__(
        self,
        min_instances,
        numerical_attributes,
        categorical_attributes,
        strategy,  # Strategy for selecting pairs
        # target_weights  # Mostly used for HMC
    ):
        """Constructs this splitter object with the given parameters."""
        self.criterion = "Squared Error"
        self.worst_performance = -1
        print("✅Initializing Splitter...")

        self.min_instances = min_instances
        self.numerical_attributes = numerical_attributes
        self.categorical_attributes = categorical_attributes
        self.strategy = strategy  # Strategy for selecting pairs
        # self.target_weights = target_weights

    def find_split_items(self, x, y, return_ranked=True):
        """Finds the most informative item to split users based on squared error reduction."""
        errors = {}
        best_item = None
        lowest_error = np.inf  # We want to minimize squared error

        for item_id in x.columns:


            item_ratings = x[item_id].values.reshape(-1, 1) # This line extracts the ratings for a specific item from a DataFrame and reshapes them into a column vector.
            if np.count_nonzero(~np.isnan(item_ratings)) < self.min_instances:
                continue

            heuristic = NumericHeuristic5(
                self.criterion, self.min_instances, #If it does not have enough ratings, it is skipped and not considered for splitting at this node.
                x, y
            )
            total_error = heuristic.squared_error_total(item_id)
            errors[item_id] = total_error
            # print(f"🔍 Item {item_id} has total squared error: {total_error:.3f}")
            

            if total_error < lowest_error:
                best_item = item_id
                lowest_error = total_error

        if return_ranked:
            ranked_items = sorted(errors.items(), key=lambda x: x[1])  # list of (item_id, error)
            # print(f"🔍 Ranked items by squared error: {ranked_items}")
            return ranked_items # best_item, lowest_error,
        else:
            return best_item, lowest_error if best_item is not None else (-np.inf)

    def select_pair(self, items_ranked, x_df, strategy=1, top_k=20):
        """
        Selects a pair of items (itemA, itemB) from the ranked list for pairwise comparison.

        Args:
            items_ranked (list): List of item IDs ranked by heuristic score (best to worst).
            x_df (DataFrame): User-item rating matrix, values can be 0 for unknown, 0.1 for hater and 1 for lover.
            strategy (int): Selection strategy:
                            1 - Top 2 items of the same type,
                            2 - Most similar (cosine),
                            3 - Least similar (cosine).
            top_k (int): Only effective for strategy 2 and 3, limits number of candidate items.

        Returns:
            tuple: (itemA, itemB), both are item IDs.
        """
        if not items_ranked:
            return None, None

        print(f"[DEBUG] select_pair called with strategy={strategy}")

        itemA, errorA = items_ranked[0] # itemA = items_ranked[0]
        typeA = get_item_type(itemA)
        print(f"\n[DEBUG] Selected itemA = {itemA} with error = {errorA:.3f}, type = {typeA}")
     
        # Build item vectors (binary: 1 if rated > 0, else 0)
        item_vectors = {
            item_id: (x_df[item_id] > 0).astype(int).values
            for item_id, _ in items_ranked[:top_k]
            if get_item_type(item_id) == typeA
        }

        same_type_candidates = [i for i in item_vectors if i != itemA]

        if strategy == 1:
            if same_type_candidates:
                itemB = same_type_candidates[0]
            else:
                raise ValueError("No same-type item found for strategy 1.")

        elif strategy in (2, 3):
            if itemA not in item_vectors:
                raise ValueError(f"No vector found for item {itemA}")
            vecA = item_vectors[itemA].reshape(1, -1)
            print(f"\n[DEBUG] itemA = {itemA}, vecA = {vecA.flatten()}")

            similarities = []
            for item in same_type_candidates:
                vecB = item_vectors[item].reshape(1, -1)
                sim = cosine_similarity(vecA, vecB)[0, 0]
                similarities.append((item, sim))
                print(f"  Compared to itemB = {item}, vecB = {vecB.flatten()}, similarity = {sim:.3f}")

            if not similarities:
                raise ValueError("No valid similarity candidates found.")

            itemB = (
                max(similarities, key=lambda x: x[1])[0] if strategy == 2
                else min(similarities, key=lambda x: x[1])[0]
            )
        

        else:
            raise ValueError(f"Unsupported strategy: {strategy}")

        return itemA, itemB
