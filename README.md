# Predictive Clustering Trees
A package for predictive clustering trees (PCTs) in Python.
Predictive clustering trees are a variant of decision trees where a tree is regarded as a hierarchy of clusters. 
Variance reduction is used as heuristic (both for classification and regression targets), allowing them to handle also multi-target learning problems.
For some references, see https://dtai.cs.kuleuven.be/clus/publications.html.

In particular, this Python implementation of PCTs was developed specifically to design new tree-based methods within our group.
We aimed to mimic the relevant functionality in the Java implementation from the DTAI group (Clus), while still being flexible enough to allow rapid prototyping of new ideas.
The package offers the following functionality:
- Core features
    - Support for missing values (in the form of a soft decision tree: an instance with missing value for the split attribute is split amongst both children of that node, proportional to the number of training instances with non-missing values that went to each child, impacting further splits and predictions by having a reduced weight/importance for this instance)
    - Support for categorical variables (a greedy search is performed in which the "best" category is iteratively added to the other child of a potential split)
    - A stopping criterion based on the F-test (asserting a statistically significant difference in variance for a split)
    - Support for multi-class classification, multi-target regression, hierarchical multilabel classification
- Extensions and adaptations of the base tree learner
    - Random forests (ensemble of PCTs trained on a bootstrapped sample of the data with a random subset of the features at each node)
    - Biclustering (allowing also the targets to be split into 2 groups at a node, instead of only the instances)
    - ~~Semi-supervised (accounting for both the input as well as the target space in the splitting heurstic)~~ (does not work yet)

## Installation instructions
This package can be used in 2 ways:
1. By making a settings file (like `test.s` in this folder) and then running
    ```bash
    $ python main.py test.s
    # Or, alternatively:
    $ ./main.py test.s
    ```
    There is some 'smart' path management going on, so this should also work when `main.py`, `test.s` and the current working directory are all different.
  
2. Within your Python code (kind of like working with `scikit-learn`) by installing `pct` as a Python package.
    - To install in "editable" mode (so you can change things in the code as you run experiments), run the following:
      ```bash
      cd pct
      python -m pip install -e .
      ```
    - Remove the "-e" if you don't want the possibility to frequently edit the package without reinstalling.
    - To uninstall, `python -m pip uninstall pct`.
    - (installation via `python setup.py install` is not recommended, see [link](https://stackoverflow.com/questions/19048732/python-setup-py-develop-vs-install))
    - In any python file you can now do `import pct.xxx` and work with the code as usual. For example:
      ```python
      from pct.tree.tree import Tree
      tree = Tree()
      tree.fit(x, y)
      ```
    - See `examples.py` for more code examples on how to use the API.


## Folder structure
[**cluschecks**](cluschecks) is used for comparing outputs with Clus, the original framework written in Java.

- [regression](cluschecks/regression) is used for a simple multi-target regression test, which should produce exactly the same splits and heuristic values.
- [hmc](cluschecks/hmc) does all kinds of tests on hierarchical multilabel classification:
  * A toy dataset for a tree hierarchy and a graph hierarchy (`toy` and `toy_DAG`)
  * Same for more involved hierarchies (`FunCat_eisen` and `GeneOnt_eisen`, data taken from [here](https://dtai.cs.kuleuven.be/clus/hmcdatasets/)).
  * Evaluation metrics can be checked thoroughly using the settings file in `auprc`. A bash script is also given to recompile and relink Clus (for printing stuff in the Java code).
- [compare\_ssl](compare_ssl) is used to compare the `clusSSL.jar` (taken from [here](http://kt.ijs.si/jurica_levatic/)) with `Clus.jar`.

---

[**data**](data) contains some `.csv` datasets. 
The datasets in `classification` and `regression` were randomly generated with the scripts in [createDatasets](createDatasets).
The datasets in `classification_hmc` are the processed versions of the `.arff` datasets in [cluschecks](/cluschecks/hmc).


---

[**doc**](doc) contains documentation files:
* *Documenting code*.
    We try to follow [Epydoc](http://epydoc.sourceforge.net/epytext.html), a lightweight Python documentation markup language, bearing many similarities to Java documentation. 
    An overview of fields (`@param`, `@return`, ...) can be found [here](http://epydoc.sourceforge.net/fields.html).
    The generated documentation (html?) can be placed in this folder.

* *Diagrams*.
    The UML and dependency diagram found here can be made by placing an `__init__.py` file in the main folder ([PCT](./)) (and maybe temporarily removing the `parser` folder and `examples.py`) and then running:
    ```bash
    $ pyreverse -o png -p all .
    ```
    I also put some profiler information here (probably already outdated by now). See [main\_optimizing.py](main_optimizing.py) for how to generate it yourself.

---

[**old_main**](old_main) contains example main Python scripts that have been used to develop the framework. Of particular interest is e.g. `main_optimizing`, containing code that allows to profile and reliably time the induction of a tree.

---

[**pct**](pct) is the codebase. See the `doc` folder for more information on its structure.

## Todolist
- Problems to address
  - [ ] pd.NA and np.nan seem to be treated differently in prediction with categorical input variables
  - [ ] Continue work on SSL (see [main_SSL](old_main/main_SSL.py))
- Parser improvements
  - [ ] Actually use the FeatureRanking and OOBEstimate settings (in section "Forest") to print stuff to `.out`
  - [ ] Smart reading of the "Forest" attributes e.g. "BagSize = sqrt" or "BagSize = log"
  - [ ] Do checks on the new "Forest" attributes (e.g. num_sub_features should be nonzero and positive, but -1 is also a valid option, because that means they are automatically generated. Maybe change this default argument to "None" in forest.py?)
  - [ ] Add "Descriptive" attribute to "Attributes" section (so user can confirm that everything  is correct)
- Performance improvements
  - [ ] Change storage of trees (Node class --> sklearn way of storing )
  - [ ] Move core to Cython
- Add functionality/other
  - [ ] Generate Epydoc documentation
  - [ ] Add survival analysis support
