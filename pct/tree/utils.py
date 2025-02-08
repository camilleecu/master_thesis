import warnings
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from collections import OrderedDict
from sklearn.preprocessing import OneHotEncoder

def learning_task(y):
    """Tries to define the learning task based on the given target matrix.

    @param y: Pandas dataframe holding the target variables.
    @return: One of several values:
        - "regression"     if all target variables are numeric
        - "classification" if all target variables are not numeric
        - "mixed"          otherwise
    """
    is_numeric = np.array([np.issubdtype(y.dtypes.iloc[i], np.number) for i in range(len(y.dtypes))])
    if all(is_numeric):
        return "regression" # create_prototypes = False
    elif all(~is_numeric):
        return "classification" # create_prototypes = True
    else:
        return "mixed"


def create_prototypes(y):
    """Converts the given categorical vector into binary prototypes using one-hot encoding."""
    y = y.astype(str)
    enc = OneHotEncoder(categories="auto")

    prototypes = pd.DataFrame(enc.fit_transform(y).todense())
    prototypes.index = list(prototypes.index)
    return prototypes

def get_target_weights(y):
    """Computes the weights for the given matrix's columns, using their variance.

    @param y: Numpy array or Pandas dataframe containing the target variables.
    @return: Numpy array of the same length as the number of columns in y.
    """
    y = np.atleast_2d(y)
    var = np.apply_along_axis(variance, 0, y)
    var /= (y.shape[0] * 2)
    weights = 1/var
    if any(weights == np.inf):
        warnings.warn("At least one of the target variables is constant, setting its weight to 0.")
        weights[weights == np.inf] = 0 # Target variables having no variance
    return weights

def variance(y):
    """Returns the variance of the given vector.

    The variance (with missing values) is defined as C{ss*(n-1)/(k-1) - n*(sv/k)**2},
    where n is the total number of instances, k is the number of non-missing instances,
    sv is the sum of the values and ss is the sum of the squared values.
    """
    k_tot = np.count_nonzero(~np.isnan(y),axis=0)  * 2 # k
    if k_tot > 1:
        n_tot = y.shape[0] * 2   # n
        sv_tot = np.nansum(y)  * 2   # sv
        ss_tot = np.nansum(y ** 2)  * 2   # ss
        var = ss_tot * (n_tot - 1) / (k_tot - 1) - n_tot * (sv_tot / k_tot)**2
    else:
        var = 1.0
    # print (var)
    return var

def is_missing(values):
    """Operator defining the condition for missing values (either numerical or categorical).

    @param values: Vector to check (elementwise) (numpy array / pandas series).
    @rtype: Boolean array of same size and type as the given values. 
    """
    if np.issubdtype(values.dtype, np.number):
        return np.isnan(values)
    else:
        return values == '?'

def is_in_left_branch(values, splitting_value):
    """Operator defining the condition for going to the left child of a given node.

    @param values: Vector to check (elementwise) (numpy / pandas).
    @param splitting_value: The test to apply to the given values.
    @type  splitting_value: Number for numerical attributes, list of strings for categorical.
    @rtype: Boolean array of same size and type as the given values.
    """
    if np.issubdtype(values.dtype, np.number):
        return values > splitting_value
    else:
        # If pandas index or df, we might use (maybe faster?)
        # values.isin(splitting_value) (won't work for numpy arrays though)
        arr = (values == splitting_value[0])
        for i in range(len(splitting_value)-1):
            arr = (arr | (values == splitting_value[i+1]))
        return arr

class HMC_parser:
    """Class containing parsing functions for hierarchical multilabel classification.

    Main usage:
    >>> parser = HMC_parser(task) # Either "tree" or "DAG"
    >>> x, y = parser.parse("hierarchy.txt", "data.csv")
    >>> class_weights = parser.get_class_weights(initial_weight=0.75)
    If `task` is "DAG", you can also pass the aggregation function that is used to
    aggregate the weights of the parent nodes (see L{get_class_weights_DAG}).

    For a nice matplotlib visualization of your graph after parsing:
    >>> parser.show_hierarchy() # (may take long / be unreadable for large graphs)

    Getting the classes (= nodes in the graph) might also prove useful:
    >>> parser.hierarchy
    """

    def __init__(self, task):
        """Initializes this parser for the given task ("tree" or "DAG")."""
        if task.upper() == "TREE":
            self.parse_hierarchy      = self.parse_hierarchy_tree
            self.show_hierarchy       = self.show_hierarchy_tree
            self.multiclass_to_binary = self.multiclass_to_binary_tree
            self.get_class_weights    = self.get_class_weights_tree
        elif task.upper() == "DAG":
            self.parse_hierarchy      = self.parse_hierarchy_DAG
            self.show_hierarchy       = self.show_hierarchy_DAG
            self.multiclass_to_binary = self.multiclass_to_binary_DAG
            self.get_class_weights    = self.get_class_weights_DAG
        self.graph = None

    def parse(self, hierarchy_file, data_file, separators=(',','/','@')):
        """
        Parses the hierarchy and data in the given files into two dataframes
        containing the descriptors and the target (as binary vectors).

        @param hierarchy_file:
            A .txt containing the <hierarchystring> from Weka's arff files,
            from the format "@ATTRIBUTE class hierarchical <hierarchystring>".
        @param data_file:
            A .csv containing the descriptors and target (as a string, last column),
            format e.g. "1/1@2" for an instance belonging to classes 1/1 and 2.
        @return:
            Pandas dataframes containing the descriptors and target binary matrix.
        """
        # TODO implement the separator functionality
        # Extract data and hierarchy
        x, y = self.extract_dataset(data_file)
        hierarchy = self.extract_hierarchy(hierarchy_file)
        
        # Parse hierarchy and convert class to binary
        self.graph = self.parse_hierarchy(hierarchy)
        y = self.multiclass_to_binary(y) # May take a while (especially for DAGs)
        return x, y

    @staticmethod
    def extract_hierarchy(filename_txt):
        """
        Extracts the hierarchy information contained in the given txt, see L{parse}.
        """
        # TODO add an "entry_sep" as argument (default ',') (or always comma?)
        hierFile = open(filename_txt)
        hierarchy = hierFile.readline().strip('\n')
        hierFile.close()
        hierarchy = hierarchy.split(sep=',')
        return hierarchy

    @staticmethod
    def extract_dataset(filename_csv):
        """
        Extracts the dataset contained in the given .csv file, see L{parse}.
        """
        data = pd.read_csv(filename_csv)

        # Extract labels
        y = pd.DataFrame(data.iloc[:,-1])
        x = pd.DataFrame(data.iloc[:,:-1])
        return x, y

    @staticmethod
    def arff_to_csv(filename_arff, filename_csv):
        """Converts the given arff to a csv, after some manual preprocessing.

        The last attribute should be "@ATTRIBUTE class string", instead of the "@ATTRIBUTE class
        hierarchical <hierarchystring>" used by CLUS. This <hierarchystring> should be extracted
        into a separate ".hierarchy.txt" file for use in L{extract_hierarchy}.
        TODO Might extract this automatically with code later
        """
        # NOTE scipy.io.arff.loadarff does not handle strings (and our target, the hierarchy, is a string)
        # TODO Missing values for numerical attributes get mapped to None by liac-arff, and then
        #      to NaN by pd.DataFrame. How are missing values for categorical attributes handled?
        #      (not an issue at the moment as everything is numeric)
        import arff # pip install liac-arff
        data = arff.load(open(filename_arff, 'r'))
        data = pd.DataFrame(data['data'], columns=data['attributes'])

        # Extract attribute names e.g. ('A1','NUMERIC) --> 'A1'
        data.columns = [data.columns[i][0] for i in range(len(data.columns))]

        # And finally convert to csv
        data.to_csv(filename_csv, index=False)


    def parse_hierarchy_tree(self, hierarchy):
        """Parses the given (tree) hierarchy into a networkx directed graph.

        @param hierarchy: List of the path to each node (excluding root),
            for example ['1', '1/1', '1/1/1', '1/1/2' '1/2', '2'].
        @return: A networkx DiGraph object.
        """
        # TODO add subclass separation symbol (default "/") as a parameter (edge_symbol)
        G = nx.DiGraph()
        G.add_node('root')
        for path in hierarchy:
            newNodeIndex = path.rfind('/') # Last occurence of '/'
            if newNodeIndex == -1:         # New top-level class (under root)
                G.add_edge('root', path)
            else:
                parent = path[:newNodeIndex] # Strips last part (e.g. '/01')
                G.add_edge(parent, path)
        return G

    def parse_hierarchy_DAG(self, hierarchy):
        """Parses the given (graph) hierarchy into a networkx directed graph.
        
        @param hierarchy: List of each edge in the graph,
            for example ['root/N1', 'N1/N2', 'N1/N3', 'root/N4' 'N4/N2'].
        @return: A networkx DiGraph object.
        """
        G = nx.DiGraph()
        hierarchy = [tuple(edge.split('/')) for edge in hierarchy]
        G.add_edges_from(hierarchy)
        return G

    @property
    def hierarchy(self):
        return list(self.graph.nodes)[1:] # Exclude the root node

    def show_hierarchy_tree(self):
        """Visualizes the hierarchy represented in this parsers graph."""
        plt.figure()
        nx.draw_planar(self.graph, with_labels=True, node_size=10, font_size=6)
        plt.show()

    def show_hierarchy_DAG(self):
        """Visualizes the hierarchy represented in this parsers graph."""
        plt.figure()
        # nx.draw_networkx(self.G, with_labels=True, node_size=10, font_size=6)
        nx.draw_networkx(self.graph)
        plt.show()

    def multiclass_to_binary_tree(self, y):
        """Converts the class vector y into a binary matrix (using this parsers graph).
        
        @param y:
            Vector with classes, where each element contains one or more nodes in the graph.
            Different elements are separated by '@', e.g. '1/1@1/2@2'. Due to the hierarchy 
            constraint, every instance should also contain the (in)direct parents of y.
        @return: Pandas dataframe containing the binary class vector for each instance.
        """
        # TODO unify this method with DAGs; in a unified function "__multiclass_to_binary"
        # we can put everything but the core (the innermost loop). However, wait until both
        # these functions are optimized.
        # TODO make this optimal for trees (replace the nx.shortest_path by just
        # using the representation, i.e. if a class is "01/01/03" then you know that
        # the parents are just "01/01" and "01")
        # TODO add separation symbol (default "@") as a parameter (multiclass_separator)
        hierarchy = list(self.graph.nodes)[1:] # All nodes except root
        labels = y.values[:,0] # Classes for each instance
        y = np.zeros((y.shape[0], len(hierarchy)))
        for i, label in enumerate(labels):
            classes = label.split('@')
            # Implement the hierarchy constraint
            for j in range(len(classes)): # Don't loop over the classes container! (we change it in the loop)
                node = classes[j]
                parents = nx.shortest_path(self.graph, source='root', target=node)
                parents = parents[1:-1] # Exclude the root and final node (is already in there)
                classes.extend(parents)
            # With list comprehension this would be:
            # classes = [nx.shortest_path(G, source='root', target=classes[j])[1:] for j in range(len(classes))]

            # TODO A lot of repetitive searches here (with index), might be mor efficient if we use
            #   some kind of datastructure like a dictionary, or let it be sorted
            #   Ensure that the search returns an error when classes[j] is not found though!
            # classes = set(classes) # Remove the duplicate entries
            indices = [hierarchy.index(classes[j]) for j in range(len(classes))]
            y[i,indices] = 1
        return pd.DataFrame(y, columns=hierarchy, dtype=int)

    def multiclass_to_binary_DAG(self, y):
        """Converts the class vector y into a binary matrix (using this parsers graph).
        
        @param y:
            Vector with classes, where each element contains one or more nodes in the graph.
            Different elements are separated by '@', e.g. '1@1/2@2'. Due to the hierarchy 
            constraint, every instance is also contained in the (in)direct parents of y.
        @return: Pandas dataframe containing the binary class vector for each instance.
        """
        hierarchy = self.hierarchy
        labels = y.values[:,0]
        y = np.zeros((y.shape[0], len(hierarchy)))
        for i, label in enumerate(labels):
            classes = label.split('@')
            # Implement the hierarchy constraint 
            # TODO for efficiency, maybe not remove those 2 then we can do without for loop
            for path in nx.all_simple_paths(self.graph, source='root', target=classes):
                classes.extend(path[1:-1]) # Exclude root and final node (are already in there)
            classes = set(classes) # Remove the duplicate entries
            indices = [hierarchy.index(node) for node in classes]
            y[i,indices] = 1
        return pd.DataFrame(y, columns=hierarchy, dtype=int)

        # # Alternative version where we calculate the nx.all_simple_paths first and put it
        # # in a dictionary (because a lot of the root-node paths will have to be calculated
        # # multiple times)
        # 
        # # Build dictionary with indexes for each node
        # indexing = dict()
        # for i, node in enumerate(hierarchy):
        #     indexing[node] = i
        # # Build dictionary telling the nodes that are between root and key
        # visitedNodes = dict()
        # for node in hierarchy:
        #     pathNodes = set()
        #     for path in nx.all_simple_paths(self.graph, source='root', target=node):
        #         pathNodes |= set(path)
        #     visitedNodes[node] = pathNodes

    def get_class_weights_tree(self, initial_weight=0.75):
        """Returns the weights for each class in the hierarchy, exponential in their depth.
        
        @param initial_weight: The w_0 to use in the formula weight = w_0 ^ depth.
            The depth starts from 0 for top-level nodes (just under the root).
        @return: A vector of class weights.
        """
        hierarchy = self.hierarchy
        depth = np.zeros(len(hierarchy), dtype=int) # NOTE: Depth is not unique in a DAG.
        for i, node in enumerate(hierarchy):
            depth[i] = node.count('/')
        return initial_weight**depth

    def get_class_weights_DAG(self, initial_weight=0.75, aggregation_func=np.mean):
        """Returns the weights for each class in the hierarchy, exponential in their depth.

        For trees, it returns the same output as `get_class_weights_tree`, but it is somewhat slower. 
        However, the small timescale makes this difference irrelevant.
        For any given node, its weight is given by ( initial_weight * aggregation_func(parent_weights) )
        
        @param initial_weight: Weight used for the top-level nodes (under the root), and used as 
            multiplier before the aggregation function for child nodes.
        @param aggregation_func:
            How to combine the weights of the parent nodes (i.e. sum, min, max, mean)
        @return: A vector of class weights.
        """
        weights = {}
        weights['root'] = 1 # Top-level nodes apparently don't get weight 1 for DAGs
        for node in self.graph.nodes - ['root']:
            self.__get_weight_recursive(node, weights, initial_weight, aggregation_func)
        return [weights[node] for node in self.hierarchy] # Return in order of hierarchy

    def __get_weight_recursive(self, node, weights, initial_weight, aggregation_func):
        """Helper function to correctly compute the weights recursively in a graph.

        @param node: The current node to get the weight of.
        @param weights: Dictionary of (node, weight) pairs which are already computed.
        @param initial_weight  : See `get_class_weights_DAG`.
        @param aggregation_func: See `get_class_weights_DAG`.
        @post: The weight of the given node and all its parent nodes are now written
            into the given weights dictionary.
        """
        if node in weights:
            return
        else:
            # First make sure the parents weights are already there
            for parent in set(self.graph.predecessors(node)) - set(weights.keys()):
                self.__get_weight_recursive(parent, weights, initial_weight, aggregation_func)

            # Then aggregate the parent weights
            parentWeights = [weights[parent] for parent in self.graph.predecessors(node)]
            weights[node] = initial_weight * aggregation_func(parentWeights)

    # def get_subtree(self):
    #     """Returns the subtree (class under root) for each label in this parsers 
    #     hierarchy. Currently only defined for tree-shaped hierarchies.
    #     """
    #     # subtrees = self.graph.successors('root') # The possible values for subtree
    #     return np.array( # As usual, we should have self.hierarchy == y.columns
    #         [nx.shortest_path(self.graph, 'root', node)[1] for node in self.hierarchy]
    #     )

    @staticmethod
    def get_subtree(labelvector, Hseparator="/"):
        """Returns the subtree (class under root) for each label in the given 
        label vector (target classes). Only for tree-shaped hierarchies.
        """
        return [label.split('/')[0] for label in labelvector]
