import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc # Area Under Curve (using trapezoid rule)
    # See Clus/clus/error/ROCAndPRCurve.java/computeArea
    # Line 140: area += 0.5*(pt[1]+prev[1])*(pt[0]-prev[0]);
    # i.e. area = (y2+y1)*(x2-x1) / 2
    # i.e. area = half of the rectangle representing a trapezoid stacked onto itself
    # i.e. Clus also uses the trapezoid rule
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import multilabel_confusion_matrix

class Evaluate:
    # TODO should this be a class?
    """Class containing all kinds of evaluation metrics."""
    def __init__(self):
        pass

    def accuracy(self, y_predicted, y_real):
        return accuracy_score(y_real, y_predicted)

    def pr_curve(self, y_predicted, y_real):
        """Returns a tuple of vectors (precision, recall, thresholds) of the same length."""
        return precision_recall_curve(y_real, y_predicted)

    """
    NOTE The following functions are from sklearn (automatic thresholding). 
    Their correctness is not guaranteed.
    """ 
    def auprc(self, y_predicted, y_real):
        """Returns the area under the precision-recall curve, using a trapezoidal rule."""
        if sum(y_real) == 0: # No positive examples, so TP+FN=0, so recall=nan
            return 1         # => the positive examples (there are none) are correctly predicted
        else:
            precision, recall, _ = self.pr_curve(y_predicted, y_real)
            return auc(recall, precision)

    def avg_auroc(self, y_predicted, y_real):
        """For a multiclass classification problem, returns the average area under the ROC curve."""
        # # Following gives an error because (I presume) some classes are not reached
        # return roc_auc_score(y_real, y_predicted, average='macro', 
        #   multi_class='ovr', labels=range(len(y_real.columns)))
        numClasses = y_real.shape[1]
        auroc = np.zeros(numClasses)
        for i in range(numClasses):
            fpr, tpr, _ = roc_curve(y_real.values[:,i], y_predicted[:,i])
            auroc[i] = auc(fpr, tpr)
        return np.nansum(auroc) / numClasses # TODO Should probably do numClassesOccuring

    def avg_auprc(self, y_predicted, y_real, weights="mean"):
        """For a multiclass classification problem, returns the average area under the PR curve.
        
        @input weights
            If "mean", weighs the classes by 1/(number of classes)
            If "freq", weighs each class by its frequency in y_real
        """
        if weights == "mean":
            weights = 1/y_real.shape[1]
        elif weights == "freq":
            weights = np.sum(y_real, axis=0)
            weights = weights/sum(weights)
        scores_per_class = average_precision_score(y_real, y_predicted, average=None)
        return np.nansum( weights * scores_per_class )

    def pooled_auprc(self, y_predicted, y_real):
        """For a multiclass classification problem, returns the pooled area under the PR curve.

        The pooled AUPRC can be seen as the area under the average PRC, and corresponds to micro-
        averaging the precision and recall. See section 5.2.2 of "Decision trees for hierarchical
        multi-label classification" by C. Vens et al.
        """
        return average_precision_score(y_real, y_predicted, average="micro")

    """
    NOTE The following functions allow for custom thresholds.
    """
    def multiclass_classification_measures(self, y_pred, y_true, thresholds=np.linspace(0, 1, 51)):
        """The naive way to implement AUROC, AUPRC, AUPRC_w and pooled AUPRC.
        
        Iterates over the given thresholds, calculates TP and FP for each, and then uses
        these to calculate the classification measures (see also the HMC paper by Celine):
            - AUROC = Area Under Receiver Operating Characteristic:
                Calculates ROC curve for each class and then averages by number of classes.
            - AUPRC = average Area Under Precision-Recall Curve:
                Calculates PR curve for each class and then averages by number of classes.
            - AUPRC_w = weighted Area Under Precision-Recall Curve:
                Calculates PR curve for each class and then averages by class frequency.
            - pooled AUPRC = Area Under average Precision-Recall Curve:
                Aggregates TP and FP count for each threshold (corresponds to micro-averaging
                the precision and recall).
        Could be further optimized, but runs fine for the moment.

        @param y_pred: Numpy array containing the prediction probabilities for each (instance, class).
        @param y_true: Pandas dataframe containing the true labels for each (instance, class).
        @param thresholds: Decision boundaries for y_pred, used to get different points on the curves.
        @return: A tuple containing (AUROC, AUPRC, weighted AUPRC, pooled AUPRC)
        """
        # Some convenience variables
        num_samples = y_true.shape[0]
        num_classes = y_true.shape[1]
        num_positive = np.sum(y_true).values      # Number of positive examples per class
        num_negative = num_samples - num_positive # ^, but negative.
        freq = num_positive / num_samples         # Class frequencies
        num_classes_occuring = sum(freq != 0)

        # Init (preallocation)
        y_pred_thresholded = y_pred.copy() # Will hold the thresholded probabilities each step
        roc_tpr = np.zeros((len(thresholds), num_classes))
        roc_fpr = np.zeros((len(thresholds), num_classes))
        avg_precis = np.zeros((len(thresholds), num_classes)) # Precision values for each class
        avg_recall = np.zeros((len(thresholds), num_classes)) # Recall values for each class
        poo_precis = np.zeros(len(thresholds)) # Pooled precision values
        poo_recall = np.zeros(len(thresholds)) # Pooled recall values

        # ------------
        # Thresholding
        thresholds = sorted(thresholds, reverse=True)
        for i, threshold in enumerate(thresholds):

            # Retrieve the confusion metrics
            tp = sum((y_pred >= threshold) & (y_true.values == 1))
            fp = sum((y_pred >= threshold) & (y_true.values == 0))
            # print("Threshold %1.2f (class 0): TP = %d, FP = %d" % (threshold, tp[0], fp[0]))
            
            # AUROC -- division could be postponed until after the loop
            roc_tpr[i,:] = tp / num_positive # = tp / (tp + fn)
            roc_fpr[i,:] = fp / num_negative # = fp / (tn + fp)

            # Average AUPRC
            avg_precis[i,:] = tp / (tp + fp)
            avg_recall[i,:] = tp / num_positive # = tp / (tp + fn)

            # Pooled AUPRC
            poo_precis[i] = sum(tp) / (sum(tp) + sum(fp))
            poo_recall[i] = sum(tp) / (sum(num_positive))

        # ---------------------
        # Calculating the AUROC
        auroc = np.zeros(num_classes)
        for j in range(num_classes):
            if freq[j] != 0:
                auroc[j] = auc(roc_fpr[:,j], roc_tpr[:,j])
        AUROC = np.sum( auroc / num_classes_occuring )

        # -----------------------------
        # Calculating the average AUPRC
        auprc = np.zeros(num_classes)
        for j in range(num_classes):
            if freq[j] != 0:
                ind = ~np.isnan(avg_precis[:,j])
                precis = avg_precis[ind,j]
                recall = avg_recall[ind,j]
                # Extend to zero recall (by setting to last non-NaN value)
                precis = np.insert(precis, 0, precis[0])
                recall = np.insert(recall, 0, 0)
                auprc[j] = auc(recall, precis)

        weights = sum(y_true.values) / np.sum(y_true.values)
        AUPRC   = np.sum( auprc / num_classes_occuring )
        AUPRC_w = np.sum( auprc * weights )

        # ----------------------------
        # Calculating the pooled AUPRC
        ind = ~np.isnan(poo_precis)
        precis = poo_precis[ind]
        recall = poo_recall[ind]
        # Extend to zero recall (by setting to last non-NaN value)
        precis = np.insert(precis, 0, precis[0])
        recall = np.insert(recall, 0, 0)
        pooled = auc(recall, precis)

        return AUROC, AUPRC, AUPRC_w, pooled

    def CLUS_multiclass_classification_measures(self, y_pred, y_true, thresholds=np.linspace(0, 1, 51)):
        """Calculates the multiclass classification measures in the same way as CLUS.

        The PR curves are built with linear interpolation: between each calculated value
        of true positives (TP), CLUS considers all integer TP values in between, and uses
        a linear interpolation to get a floating value FP at the corresponding TP's.
        Formula: interFP = (FP - prevFP)/(TP - prevTP) * (interTP - prevTP)

        There is still a lot of room for optimization. A first point that should be addressed
        is sorting the predictions before each class run. Then we can iterate over the prediction
        list until we get a value bigger than the current threshold. For the corresponding code
        in CLUS, see the while loop in clus/error/ROCAndPRCurve.java/enumerateThresholdsSelected.
        (np.unique() might be useful, returns the sorted unique values of an array)

        For debugging -- to compare stuff to Clus:
        >>> data = pd.read_csv('PCT/cluschecks/hmc/FunCat_eisen/FunCat.train.pr.csv')
        >>> node = '01' # Or 01/01, or ... (see y_pre) 
        >>> node = 'ALL' # For pooled
        >>> clusPrec = data[data['Class'] == node].iloc[:,2].values
        >>> clusRec  = data[data['Class'] == node].iloc[:,1].values
        Also useful for debugging
        >>> print(np.vstack((recall, precision)).transpose())
        """
        assert all([0 <= thr <= 1 for thr in thresholds]), "Thresholds should be in [0,1]!"

        # Some convenience variables
        num_positive = np.sum(y_true).values
        num_negative = len(y_true.index) - num_positive
        freq = num_positive / len(y_true.index) # Vector of class frequencies (as outputted by clus)
        num_samples = y_true.shape[0]
        num_classes = y_true.shape[1]
        num_classes_occuring = sum(freq != 0) # = sum( np.any(y_true.values == 1, axis=0) )

        # -----------------------------------------------------
        # Calculating the points on the precision-recall curves
        thresholds = sorted(thresholds, reverse=True)
        pooledTPFP = np.zeros((len(thresholds), 2))
        auprc = np.zeros(num_classes)
        auroc = np.zeros(num_classes)
        for col in range(num_classes):
            if freq[col] != 0: # Only consider classes occuring in y_true
                firstPoint = True
                true = y_true.values[:,col]
                pred = y_pred[:,col]
                classPrecis = []
                classRecall = []
                roc_tpr = [0]
                roc_fpr = [0]
                prevTP = 0
                prevFP = 0
                for i, threshold in enumerate(thresholds):
                    TP = sum(( true) & (pred >= threshold))
                    FP = sum((~true) & (pred >= threshold))
                    # TP = sum(  true [pred >= threshold])
                    # FP = sum((~true)[pred >= threshold])

                    pooledTPFP[i,:] += [TP,FP]

                    if (TP != prevTP or FP != prevFP):
                        # For ROC
                        roc_tpr.append( TP/num_positive[col] )
                        roc_fpr.append( FP/num_negative[col] )

                        # For PR
                        if firstPoint:
                            classPrecis.append(TP/(TP+FP))
                            classRecall.append(TP/num_positive[col])
                            firstPoint = False
                        else:
                            # Taking some extra points (prevTP instead of prevTP+1) because
                            # we otherwise don't add the second point (the one after firstPoint)
                            interTP = range(prevTP, TP+1)
                            interFP = np.interp( interTP, [prevTP,TP], [prevFP,FP] ) # Linear interpol
                            classPrecis.extend( interTP / (interTP + interFP) )
                            classRecall.extend( interTP / num_positive[col] )
                        
                        prevTP = TP
                        prevFP = FP

                # Extend to zero recall
                classPrecis.insert(0, classPrecis[0])
                classRecall.insert(0, 0)
                # print( np.vstack( (classRecall, classPrecis) ).transpose() )

                auprc[col] = auc(classRecall, classPrecis)
                auroc[col] = auc(roc_fpr, roc_tpr) # should always have (0,0) and (1,1)

        # -----------------------------
        # Calculating the AUROC
        AUROC = np.sum( auroc / num_classes_occuring )

        # -----------------------------
        # Calculating the average AUPRC
        weights = sum(y_true.values) / np.sum(y_true.values)
        AUPRC   = np.sum( auprc / num_classes_occuring )
        AUPRC_w = np.sum( auprc * weights )

        # ----------------------------
        # Calculating the pooled AUPRC
        minTP = int(min(pooledTPFP[pooledTPFP[:,0] != 0,0])) # don't count '0' for the minimum
        maxTP = int(max(pooledTPFP[:,0]))
        TP = range(minTP, maxTP+1)
        FP = np.interp(TP, pooledTPFP[:,0], pooledTPFP[:,1]) # Linear interpolation
        recall = list(TP / sum(num_positive))
        precis = list(TP / (TP + FP))
        # Extend to zero recall
        precis.insert(0, precis[0])
        recall.insert(0, 0)
        pooled = auc(recall, precis)
        return AUROC, AUPRC, AUPRC_w, pooled