import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from pct.tree.heuristic.Heuristic import Heuristic5
from pct.tree.heuristic.semibi_NumericHeuristic import NumericHeuristic5

# Global item_type_map is expected to be defined externally in the notebook
item_type_map = {}
    
class Splitter:
    def __init__(
        self,
        min_instances,
        numerical_attributes,
        categorical_attributes,
        strategy,
        item_type_map,  # Map of item IDs to their types
        # target_weights  # Mostly used for HMC
    ):
        """Constructs this splitter object with the given parameters."""
        self.criterion = "Squared Error"
        self.worst_performance = -1
        print("Initializing Splitter...")

        self.min_instances = min_instances
        self.numerical_attributes = numerical_attributes
        self.categorical_attributes = categorical_attributes
        # self.target_weights = target_weights
        self.strategy = strategy
        self.item_type_map = item_type_map 

    def get_item_type(item_id):
        return item_type_map.get(item_id, 'unknown')

    def find_split_items(self, x, y, return_ranked=True):
        errors = {}
        best_item = None
        lowest_error = np.inf

        for item_id in x.columns:
            item_ratings = x[item_id].values.reshape(-1, 1)
            if np.count_nonzero(~np.isnan(item_ratings)) < self.min_instances:
                continue

            heuristic = NumericHeuristic5(self.criterion, self.min_instances, x, y)
            total_error = heuristic.squared_error_total(item_id)
            errors[item_id] = total_error

            if total_error < lowest_error:
                best_item = item_id
                lowest_error = total_error

        if return_ranked:
            ranked_items = sorted(errors.items(), key=lambda x: x[1])
            return ranked_items
        else:
            return best_item, lowest_error if best_item is not None else (-np.inf)

    def select_pair(self, items_ranked, x_df, strategy=None, top_k=20):
        if strategy is None:
            strategy = self.strategy  # use splitter default

        if not items_ranked:
            return None, None

        itemA, errorA = items_ranked[0]
        typeA = self.get_item_type(itemA)

        item_vectors = {
            item_id: (x_df[item_id] > 0).astype(int).values
            for item_id, _ in items_ranked[:top_k]
            if self.get_item_type(item_id) == typeA
        }

        same_type_candidates = [i for i in item_vectors if i != itemA]

        # Fallback to strategy 1 if candidates missing
        if strategy == 1 or (strategy in (2, 3) and not same_type_candidates):
            if same_type_candidates:
                itemB = same_type_candidates[0]
            else:
                print(f"[⚠️] No same-type candidates for itemA={itemA}, returning None.")
                return None, None

        elif strategy in (2, 3):
            if itemA not in item_vectors:
                print(f"[⚠️] No vector for itemA={itemA}, fallback to strategy 1.")
                return self.select_pair(items_ranked, x_df, strategy=1, top_k=top_k)

            vecA = item_vectors[itemA].reshape(1, -1)
            similarities = []
            for item in same_type_candidates:
                vecB = item_vectors[item].reshape(1, -1)
                sim = cosine_similarity(vecA, vecB)[0, 0]
                similarities.append((item, sim))

            if not similarities:
                print(f"[⚠️] No valid similarity candidates for itemA={itemA}, fallback to strategy 1.")
                return self.select_pair(items_ranked, x_df, strategy=1, top_k=top_k)

            itemB = max(similarities, key=lambda x: x[1])[0] if strategy == 2 else min(similarities, key=lambda x: x[1])[0]
        else:
            raise ValueError(f"Unsupported strategy: {strategy}")

        return itemA, itemB
