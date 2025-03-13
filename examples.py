import numpy as np
import pandas as pd

# Contents (also serves as glossary)
# PCT    = Predictive Clustering Tree for standard regression and classification
# HMC    = Hierarchical Multilabel Classification
# PBCT   = Predictive Bi-Clustering Tree
# RF     = Random Forest (of PCT, PBCT, ...)

#   ____   ____ _____ 
#  |  _ \ / ___|_   _|
#  | |_) | |     | |  
#  |  __/| |___  | |  
#  |_|    \____| |_|  
#
from pct.tree.indiantree import Tree

# ==========
# Regression
# 1. Loading the data
x_regr = pd.read_csv("data/regression/hard_test1.csv")
targets = [v for i,v in enumerate(x_regr.columns) if "target" in v.lower()]
y_regr = pd.DataFrame(x_regr[targets])
x_regr = x_regr.drop(targets,axis=1)

# 2. Training the tree
tree = Tree(min_instances=5, ftest=1)
tree.fit(x_regr, y_regr)

# 3. Other fun stuff
tree.predict(x_regr)
#tree.draw_tree('tree')
tree.decision_path(x_regr)

# ==============
# Classification
# 1. Loading the data
x_clas = pd.read_csv("data/classification/test_both_missing1.csv")
classes = [v for i,v in enumerate(x_clas.columns) if "class" in v]
y_clas = pd.DataFrame(x_clas[classes])
x_clas = x_clas.drop(classes,axis=1)

# 2. Training the tree
tree = Tree(min_instances=5, ftest=1)
tree.fit(x_clas, y_clas)

# 3. Other fun stuff
tree.predict(x_clas, single_label=True)
#tree.draw_tree('tree')
tree.decision_path(x_clas)

#   _   _ __  __  ____ 
#  | | | |  \/  |/ ___|
#  | |_| | |\/| | |    
#  |  _  | |  | | |___ 
#  |_| |_|_|  |_|\____|
#
from pct.tree.indiantree import Tree
from pct.tree.utils import HMC_parser
from pct.evaluate.evaluate import Evaluate

# ================
# Tree hierarchies
# 1. Loading the data
path = 'data/classification_hmc/toy' # Options: toy, eisen_FUN
parser = HMC_parser("tree")
x, y = parser.parse(path+'.hierarchy.txt', path+'.train.csv')
class_weights = parser.get_class_weights(initial_weight=0.75)

# 2. Training the tree
tree = Tree(min_instances=5, ftest=1)
tree.fit(x, y, target_weights=class_weights)

# 3. Evaluating the results
e = Evaluate()
y_pred = tree.predict(x)
AUROC, AUPRC, AUPRC_w, pooled = e.CLUS_multiclass_classification_measures(y_pred, y)

# ===============
# DAG hierarchies
# 1. Loading the data
path = 'data/classification_hmc/toy_DAG' # Options: toy_DAG, eisen_GO
parser = HMC_parser("DAG")
x, y = parser.parse(path+'.hierarchy.txt', path+'.train.csv')
class_weights = parser.get_class_weights(initial_weight=0.75, aggregation_func=np.mean)

# 2. Training the tree
tree = Tree(min_instances=5, ftest=1)
tree.fit(x, y, target_weights=class_weights)

# 3. Evaluating the results
e = Evaluate()
y_pred = tree.predict(x)
topLevel = np.all(y.values == 1, axis=0) # Remove illegal classes
y      = y.iloc[:,~topLevel]
y_pred = y_pred[:,~topLevel]
AUROC, AUPRC, AUPRC_w, pooled = e.CLUS_multiclass_classification_measures(y_pred, y)


#   ____  ____   ____ _____ 
#  |  _ \| __ ) / ___|_   _|
#  | |_) |  _ \| |     | |  
#  |  __/| |_) | |___  | |  
#  |_|   |____/ \____| |_|  
#
from pct.tree.bi_tree import BiClusteringTree
from pct.tree.utils import HMC_parser

# ================
# 1. Loading the data
# Parse HMC data as before
path = 'data/classification_hmc/toy' # Options: toy, eisen_FUN
parser = HMC_parser("tree")
x_pbct, y_pbct = parser.parse(path+'.hierarchy.txt', path+'.train.csv')
class_weights_pbct = parser.get_class_weights(initial_weight=0.75)
# Construct vertical features
label_features = [label for label in y_pbct.columns if "/" not in label] # top level
vert_features = [label.split('/')[0] for label in y_pbct.columns]
vert_features = pd.DataFrame(vert_features, index=y_pbct.columns, columns=["Subtree"])

# 2. Training the tree
BiClusteringTree.CLUS = False
tree = BiClusteringTree(min_instances=5, ftest=0.01)
tree.fit(x_pbct, y_pbct, vert_features, target_weights=class_weights_pbct)


#   ____  _____ 
#  |  _ \|  ___|
#  | |_) | |_   
#  |  _ <|  _|  
#  |_| \_\_|    
#
from pct.forest.forest import RandomForest
from pct.forest.bi_forest import RandomBiClusteringForest
from pct.tree.utils import HMC_parser

# =========================
# Forest of regression PCTs
# 1. Loading the data -- same as in PCT/regression
x_regr = pd.read_csv("data/regression/hard_test1.csv")
targets = [v for i,v in enumerate(x_regr.columns) if "target" in v.lower()]
y_regr = pd.DataFrame(x_regr[targets])
x_regr = x_regr.drop(targets,axis=1)

# 2. Training the forest
# For classification and HMC, apply exactly the same changes to this as regular PCT
# (i.e. use the `single_label` and `target_weights` arguments)
forest = RandomForest(min_instances=5, ftest=1, num_trees=3, random_state=1)
forest.fit(x_regr, y_regr, num_sub_instances=-1, num_sub_features=-1, n_jobs=-1)

# 3. Other fun stuff
forest.predict(x_regr)
forest.feature_importances_
forest.oob_error_
forest.itb_error_
forest.decision_path(x_regr)

# ===============
# Forest of PBCTs
# 1. Loading the data -- same as in PBCT
# Parse HMC data
path = 'data/classification_hmc/toy' # Options: toy, eisen_FUN
parser = HMC_parser("tree")
x_pbct, y_pbct = parser.parse(path+'.hierarchy.txt', path+'.train.csv')
class_weights_pbct = parser.get_class_weights(initial_weight=0.75)
# Construct vertical features
label_features = [label for label in y_pbct.columns if "/" not in label] # top level
vert_features = [label.split('/')[0] for label in y_pbct.columns]
vert_features = pd.DataFrame(vert_features, index=y_pbct.columns, columns=["Subtree"])

# 2. Training the forest
forest = RandomBiClusteringForest(num_trees=3, random_state=1)
forest.fit(
    x_pbct, y_pbct, vert_features, 
    target_weights=class_weights_pbct,
    num_sub_instances  =-1,
    num_sub_features   =-1,
    num_sub_V_features =-1,
    n_jobs=-1
)

# 3. Other fun stuff
forest.predict(x_pbct)
forest.feature_importances_
forest.oob_error_
forest.itb_error_
forest.decision_path(x_pbct)
