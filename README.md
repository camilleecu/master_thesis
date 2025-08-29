This repository contains the source code to reproduce results and figures from the paper "Pairwise and Attribute-Aware Decision Tree-Based Preference Elicitation for Cold-Start Recommendation", submitted to the MuRS workshop at RecSys 2025.



Data File Description
filtered_df.csv
Description: This file contains the filtered interaction data used in the experiments.

Columns:
user_id: Unique identifier for each user.
item_id: Unique identifier for each item (e.g., song or record).
rating: The explicit or implicit rating given by the user.
item_type: Type/category of item (e.g., song, album).
artist_id: Unique identifier for the artist of the item.
genre_ids: One or more genre categories associated with the item.

Usage:
This file serves as the main data source for training and evaluating recommendation models in the code and notebooks provided.
