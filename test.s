[General]
Verbose = 1
RandomSeed = 1

[Data]
File = data/regression/hard_test1.csv
%File = data/classification_hmc/eisen_FUN.train.csv
TestSet = data/regression/hard_test1.csv
Hierarchy = data/classification_hmc/eisen_FUN.hierarchy.txt

[Attributes]
Target = [7,8]

[Output]
WritePredictionsTrain = false
WritePredictionsTest = NO

[Model]
%MinimalWeight = 5.0

[Tree]
FTest = 1.0
%PBCT = yes

[Hierarchical]
%Type = Tree
%WType = "ExpAvgParentWeight"
%WParam = 
%ClassificationThreshold = [0,2,4,100]

[Forest]
NumberOfTrees = 4