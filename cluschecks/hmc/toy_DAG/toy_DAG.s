[General]
Compatibility = MLJ08

[Data]
File    = /home/wuyi/Dropbox/job2020_PCT_Forest/ClusJava/toy_DAG/toy_DAG.train.arff.zip
TestSet = /home/wuyi/Dropbox/job2020_PCT_Forest/ClusJava/toy_DAG/toy_DAG.train.arff.zip

[Attributes]
ReduceMemoryNominalAttrs = yes

[Hierarchical]
Type = DAG
WType = ExpAvgParentWeight
HSeparator = /
EvalClasses = /home/wuyi/Dropbox/job2020_PCT_Forest/ClusJava/toy_DAG/evalclasses.txt
ClassificationThreshold = [0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32,34,36,38,40,42,44,46,48,50,52,54,56,58,60,62,64,66,68,70,72,74,76,78,80,82,84,86,88,90,92,94,96,98,100]

[Tree]
PruningMethod = None
ConvertToRules = No
FTest = 0.01

[Model]
MinimalWeight = 5.0

[Output]
WritePredictions = {Train, Test}

