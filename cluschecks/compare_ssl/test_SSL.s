[General]
Compatibility = MLJ08

[Data]
File    = hard_test1.arff
TestSet = hard_test1.arff

[Attributes]
ReduceMemoryNominalAttrs = yes
Descriptive = 1-6
Target = 7-8
% Values: None, 1-8
% Effect: No influence on first split, but 3 unique trees: {None,1,2,3,4,5,6}, {7} and {8}
GIS = None

[Tree]
ConvertToRules = No
FTest = 1
PruningMethod = None
% Values: N2, LOG, LINEAR, NPAIRS, TEST
% Effect: None
HeuristicComplexity = TEST
% Values: GSMDistance, HammingLoss, Jaccard, Matching, Euclidean
% Effect: None
SetDistance = Euclidean
% Values: Euclidean, Minkowski
% Effect: None
TupleDistance = Euclidean
% Values: DTW, QDM, TSC
% Effect: None
TSDistance = TSC
% Values: Ignore, EstimateFromTrainingSet, EstimateFromParentNode
% Effect: None
MissingClusteringAttrHandling = Ignore
% Values: Zero, DefaultModel, ParentNode
% Effect: None
MissingTargetAttrHandling = DefaultModel
% Values: StandardEntropy, ModifiedEntropy
% Effect: None
EntropyType = ModifiedEntropy
% Values: ?? I guess {Yes, No}? (there's no error when putting a random value)
% Effect: None (for the assumed values) (also, typo in this setting name?)
ConsiderUnlableInstancesInIGCalc = No
% Values: Binary, Euclidean, Modified, Gaussian
% Effect: None
SpatialMatrix = Gaussian
% Values: GlobalMoran, GlobalGeary, GlobalGetis, LocalMoran, LocalGeary, LocalGetis, StandardizedGetis, 
%         EquvalentI, IwithNeighbours, EquvalentIwithNeighbours, GlobalMoranDistance, GlobalGearyDistance, 
%         CI, MultiVariateMoranI, CwithNeighbours, Lee, MultiIwithNeighbours, CIwithNeighbours, 
%         LeewithNeighbours, Pearson, CIDistance, DH, EquvalentIDistance, PearsonDistance, EquvalentG, 
%         EquvalentGDistance, EquvalentPDistance
% Effect: None (yes, I've tried them all)
SpatialMeasure = EquvalentPDistance
% Values: Any float I guess (even negative)
% Effect: None
Bandwidth = 0.001
% Values: ?? I guess {Yes, No} again
% Effect: None (for the assumed values)
Longlat = No
% Values: Any float I guess (even negative) (was at 0.0 initially)
% Effect: None
NumNeightbours = -10.0
% Values: Any float I guess (was at 1.0 initially)
% Effect: None
Alpha = -10.22
% Values: Exact, Middle
% Effect: It changed the splitting value of the first split by 1!
%         But still a different attribute from the other Clus (and a whole different value)
SplitPosition = Exact

[Model]
MinimalWeight = 5.0
