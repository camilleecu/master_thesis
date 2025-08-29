This repository contains the source code to reproduce results and figures from the paper "Pairwise and Attribute-Aware Decision Tree-Based Preference Elicitation for Cold-Start Recommendation", submitted to the MuRS workshop at RecSys 2025.



## Data File Descriptions

### filtered_df.csv
**Description:**  
This file contains the filtered interaction data used in the experiments.

**Columns:**  
- **user_id:** Unique identifier for each user.  
- **item_id:** Unique identifier for each item (e.g., song or record).  
- **rating:** The explicit or implicit rating given by the user.  
- **item_type:** Type/category of item (e.g., song, album).  
- **artist_id:** Unique identifier for the artist of the item.  
- **genre_ids:** One or more genre categories associated with the item.  

**Usage:**  
This file serves as the main data source for training and evaluating recommendation models in the notebooks Yahoo_Pairwise_elicitation.ipynb and Yahoo_four_elicitation.ipynb.

---

### filtered_semi_binary.csv
**Description:**  
This file contains user-item interaction data where the `rating` column has been converted to a binary format.

**Columns:**  
- **user_id:** Unique identifier for each user.  
- **item_id:** Unique identifier for each item.  
- **rating:** Binary rating (e.g., 0/1) indicating negative or positive feedback.  
- **item_type:** Type/category of item.  
- **artist_id:** Unique identifier for the item’s artist.  
- **genre_ids:** Genre categories associated with each item.  

**Usage:**  
This file serves as the main data source for training and evaluating recommendation models in the notebooks Yahoo_single_elicitation.ipynb.

### Yahoo Data Processing

Details on how the Yahoo data is preprocessed and filtered can be found in the notebook:  
**Yahoo_data_processor.ipynb**

This notebook documents the steps for cleaning, filtering, and preparing the Yahoo dataset before it is used in experiments and model training.


### Figure Reproducibility

- Plots of **pairwise_tree_1** and **pairwise_tree_2** can be found in the notebook:  
  `Yahoo_Pairwise_elicitation.ipynb`

- Plots of **single_item_tree_hybrid** and **Max performance** are available in the notebook:  
  `Yahoo_single_elicitation.ipynb`

Each notebook contains code and instructions to reproduce the corresponding figures used in the paper.

