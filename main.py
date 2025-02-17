#!/usr/bin/env python3

import sys
import os
import time
from pathlib import Path
# Data handling
import numpy as np
import pandas as pd
# Input-output
from pct.parser.parser import Parser
from pct.tree.utils import HMC_parser
# Learners
from pct.tree.tree import Tree
from pct.tree.bi_tree import BiClusteringTree
from pct.forest.forest import RandomForest
from pct.forest.bi_forest import RandomBiClusteringForest
# Utilities
import pct.tree.utils as utils
from pct.evaluate.evaluate import Evaluate
import datetime
from active_learning import active_learning_iteration


def main():
    # ======================
    # Parsing input settings
    # ======================
    #SettingsFile = "PCT/test.s" #sys.argv[1]
    SettingsFile = sys.argv[1]
    
    SettingsFile = Path(os.getcwd(), SettingsFile)
    if not SettingsFile.exists():
        raise FileNotFoundError(f"Could not find '{SettingsFile}'.")
    parser = Parser(SettingsFile)

    # General
    Verbose    = parser.settings.sections["General"].attributes["Verbose"].value
    RandomSeed = parser.settings.sections["General"].attributes["RandomSeed"].value
    # Data
    File      = parser.settings.sections["Data"].attributes["File"].value
    TestSet   = parser.settings.sections["Data"].attributes["TestSet"].value
    Hierarchy = parser.settings.sections["Data"].attributes["Hierarchy"].value
    # Attributes
    Target = parser.settings.sections["Attributes"].attributes["Target"].value
    # Output
    WritePredTrain = parser.settings.sections["Output"].attributes["WritePredictionsTrain"].value
    WritePredTest  = parser.settings.sections["Output"].attributes["WritePredictionsTest"].value
    # Model
    MinimalWeight = parser.settings.sections["Model"].attributes["MinimalWeight"].value
    # Tree
    FTest = parser.settings.sections["Tree"].attributes["FTest"].value
    PBCT  = parser.settings.sections["Tree"].attributes["PBCT"].value
    # Hierarchical
    HierType   = parser.settings.sections["Hierarchical"].attributes["Type"].value
    HierWType  = parser.settings.sections["Hierarchical"].attributes["WType"].value
    HierWParam = parser.settings.sections["Hierarchical"].attributes["WParam"].value
    ClassificationThreshold = parser.settings.sections["Hierarchical"].attributes["ClassificationThreshold"].value
    # Forest
    NumberOfTrees            = parser.settings.sections["Forest"].attributes["NumberOfTrees"].value
    BagSize                  = parser.settings.sections["Forest"].attributes["BagSize"].value
    NumberOfFeatures         = parser.settings.sections["Forest"].attributes["NumberOfFeatures"].value
    NumberOfVerticalFeatures = parser.settings.sections["Forest"].attributes["NumberOfVerticalFeatures"].value
    FeatureRanking           = parser.settings.sections["Forest"].attributes["FeatureRanking"].value
    OOBEstimate              = parser.settings.sections["Forest"].attributes["OOBEstimate"].value

    # ===============================
    # Import data (and class weights)
    # ===============================
    if Verbose > 0: print("Reading data")
    target_weights = None
    if HierType is None:
        x_train, y_train = read_data(File, Target)
        if TestSet != "":
            x_test, y_test = read_data(TestSet, Target)
    else:
        x_train, y_train, target_weights = read_data_hier(File, Hierarchy, HierType, HierWType, HierWParam)
        if TestSet != "":
            x_test, y_test, _ = read_data_hier(TestSet, Hierarchy, HierType, "NoWeight", HierWParam)
    if PBCT == True:
        assert HierType is not None, "PBCT does not yet support non-HMC tasks!"
        subtree = HMC_parser.get_subtree(labelvector=y_train.columns, Hseparator="/")
        subtree = pd.DataFrame(subtree, index=y_train.columns, columns=["Subtree"])



    # =================
    # Train the learner
    # =================
    Tree.VERBOSE = Verbose
    if Verbose > 0: print("Training the learner\n")
    start = time.time()
    X_matrix = np.zeros_like(x_train)  # Placeholder for unrated items (Ensure it has actual data!)## Camille
    
    if NumberOfTrees is None:
        if PBCT == False:
            
            learner = Tree(min_instances=MinimalWeight, ftest=FTest)

            x_train = active_learning_iteration(learner, x_train, y_train, X_matrix) ## Camille

            learner.fit(x_train, y_train, target_weights=target_weights)
            # Apply Active Learning to refine training data
            
        elif PBCT == True:
            learner = BiClusteringTree(min_instances=MinimalWeight, ftest=FTest)
            learner.fit(x_train, y_train, subtree,
                target_weights=target_weights)
    else:
        if PBCT == False:
            learner = RandomForest(
                min_instances=MinimalWeight, ftest=FTest, num_trees=NumberOfTrees, random_state=RandomSeed
            )
            learner.fit(
                x_train, y_train, target_weights=target_weights, 
                num_sub_instances=BagSize, num_sub_features=NumberOfFeatures
                # n_jobs=-1
            )
        elif PBCT == True:
            learner = RandomBiClusteringForest(
                min_instances=MinimalWeight, ftest=FTest, num_trees=NumberOfTrees, random_state=RandomSeed
            )
            learner.fit(
                x_train, y_train, subtree,
                target_weights=target_weights, 
                num_sub_instances=BagSize, num_sub_features=NumberOfFeatures, num_sub_V_features=NumberOfVerticalFeatures, 
                n_jobs=-1
            )
    end = time.time()
    inductionTime = end-start
    if Verbose > 0: print(f"Induction time: {inductionTime:.3f}\n\n")

    # ===================
    # Writing predictions TODO maybe add x to the generated prediction csv as well?
    # ===================
    if WritePredTrain is True:
        predFilePath = str(os.path.splitext(SettingsFile)[0]) + ".train.pred.csv"
        y_pred = learner.predict(x_train)
        pd.DataFrame(y_pred, columns=y_train.columns).to_csv(predFilePath)
    if TestSet != "" and WritePredTest is True:
        predFilePath = str(os.path.splitext(SettingsFile)[0]) + ".test.pred.csv"
        y_pred = learner.predict(x_test)
        pd.DataFrame(y_pred, columns=y_test.columns).to_csv(predFilePath)
    
    # ==========================
    # Printing to an output file
    # ==========================
    outFilePath = str(os.path.splitext(SettingsFile)[0]) + ".out"
    with open(outFilePath, 'w') as f:
        f.write(f"Python PCT Framework -- run on '{os.path.basename(SettingsFile)}'\n")
        f.write(datetime.datetime.now().strftime('Date: %d/%m/%Y, time: %H:%M\n'))
        f.write("*"*48 + "\n")
        f.write(str(parser.settings))
        f.write("\n\n\n")
        f.write("Statistics\n")
        f.write("----------\n")
        f.write(f"F-test: significance level = {FTest}\n")
        f.write(f"Induction time: {inductionTime:.3f}\n")
        f.write(f"Model information: ")
        if NumberOfTrees is None:
            learnerIter = [learner]
        else:
            learnerIter = learner.trees
            f.write(f"({NumberOfTrees} trees)\n")
        for tree in learnerIter:
            if NumberOfTrees is not None:
                f.write("\t")
            f.write(f"Nodes = {tree.size['node_count']+tree.size['leaf_count']}") 
            f.write(f" (Leaves: {tree.size['leaf_count']})")
            if PBCT == True:
                f.write(f" --  Vertical nodes: {tree.size['vert_node_count']}")
            f.write("\n")
        f.write("\n")
        f.write("Training error\n")
        f.write("--------------\n")
        if Verbose > 0: print("Computing training error")
        calc_and_write_error(f, learner, x_train, y_train, HierType, ClassificationThreshold, NumberOfTrees)
        f.write("\n")
        if TestSet != "":
            f.write("Testing error\n")
            f.write("--------------\n")
            if Verbose > 0: print("Computing testing error")
            calc_and_write_error(f, learner, x_test, y_test, HierType, ClassificationThreshold, NumberOfTrees)
            f.write("\n")

def read_data(FileName, Target):
    """Reads data from the given filename."""
    Target = [Target[i] - 1 for i in range(len(Target))] # Adjust for zero-indexing
    data = pd.read_csv(FileName)
    NonTarget = list(set(range(len(data.columns)))-set(Target))
    y = data.iloc[:,Target]
    x = data.iloc[:,NonTarget]
    return x, y

def read_data_hier(FileName, Hierarchy, HierType, WType, WParam, PBCT=False):
    """Reads and parsers hierarchical data (and computes class weights)."""
    parser = HMC_parser(HierType)
    x, y = parser.parse(Hierarchy, FileName)
    if WType == "NoWeight":
        class_weights = np.ones(len(y.columns))
    elif HierType.upper() == "TREE":
        class_weights = parser.get_class_weights(initial_weight=WParam)
    elif HierType.upper() == "DAG":
        mapping = {
            'ExpSumParentWeight':np.sum,
            'ExpAvgParentWeight':np.mean,
            'ExpMinParentWeight':np.min,
            'ExpMaxParentWeight':np.max,
        }
        class_weights = parser.get_class_weights(initial_weight=WParam, aggregation_func=mapping[WType])
    return x, y, class_weights

def calc_and_write_error(f, learner, x, y, HierType, thresholds, NumberOfTrees):
    """Writes prediction error to the given output file (handle f), depending
    on the learning task.
    """
    f.write(f"Number of examples: {len(x.index)}\n")
    if HierType is not None:
        f.write("Hierarchical error measures\n")
        e = Evaluate()
        thresholds = [thr/100 for thr in thresholds]
        AUROC, AUPRC, AUPRC_w, pooled = e.CLUS_multiclass_classification_measures(
            learner.predict(x), y, thresholds=thresholds
        )
        f.write(f"\tAverage AUROC           : {AUROC}\n")
        f.write(f"\tAverage AUPRC           : {AUPRC}\n")
        f.write(f"\tAverage AUPRC (weighted): {AUPRC_w}\n")
        f.write(f"\tPooled AUPRC            : {pooled}\n")
        # TODO also AUPRC for each class separately, as in Clus?
    elif NumberOfTrees is not None:
        f.write(f"Out-of-bag error: \n")
        errors = learner.oob_error_
        for target in errors.index:
            f.write(f"\t{target}: {errors[target]}\n")
        f.write("\n")
    else:
        task = utils.learning_task(y)
        if task == "regression":
            f.write("Mean squared error: TODO\n")
        elif task == "classification":
            f.write("Accuracy: TODO\n")

main()