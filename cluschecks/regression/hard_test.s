[General]
Compatibility = MLJ08

[Data]
File    = hard_test1.arff
TestSet = hard_test1.arff

[Attributes]
ReduceMemoryNominalAttrs = yes
Descriptive = 1-6
Target = 7-8

[Tree]
%{Default, ReducedError, Gain, GainRatio, SSPD, VarianceReduction, MEstimate, Morishita, DispersionAdt, DispersionMlt, RDispersionAdt, RDispersionMlt, GeneticDistance, SemiSupervised, VarianceReductionMissing}
Heuristic = VarianceReduction
ConvertToRules = No
FTest = 1
PruningMethod = None

[Model]
MinimalWeight = 5.0
