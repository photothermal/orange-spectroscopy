import numpy as np
import sklearn.cross_decomposition as skl_pls


from Orange.data import Variable, ContinuousVariable
from Orange.preprocess.score import LearnerScorer
from Orange.regression import SklLearner, SklModel

# Add any pre-processing of data here
# Normalization is only needed if and when the data
# is changing overall shape or the x axis is varying for every data row/instance

pls_pps = SklLearner.preprocessors


class _FeatureScorerMixin(LearnerScorer):
    feature_type = Variable
    class_type = ContinuousVariable

    def score(self, data):
        model = self(data)
        return np.abs(model.coefficients), model.domain.attributes


# Simple PLS Regression:


class SklPLSRegressionLearner(SklLearner, _FeatureScorerMixin):
    __wraps__ = skl_pls.PLSRegression

    def __init__(self, preprocessors=None):
        super().__init__(preprocessors=preprocessors)

    def fit(self, X, Y, W=None):
        model = super().fit(X, Y, W)
        return PLSModel(model.skl_model)


class PLSRegressionLearner(SklPLSRegressionLearner):
    __wraps__ = skl_pls.PLSRegression
    preprocessors = pls_pps

    def __init__(self, n_components=2, scale=True,
                 max_iter=500, preprocessors=None):
        super().__init__(preprocessors=preprocessors)
        self.params = vars()

# PLS MODEL


class PLSModel(SklModel):

    @property
    def coefficients(self):
        return self.skl_model.coef_

    def predict(self, X):
        vals = self.skl_model.predict(X)
        if len(vals.shape) == 1:
            # Prevent IndexError for 1D array
            return vals
        elif vals.shape[1] == 1:
            return vals.ravel()
        else:
            return vals

    def __str__(self):
        return 'PLSModel {}'.format(self.skl_model)


if __name__ == '__main__':
    import Orange

    data = Orange.data.Table('housing')
    learners = [PLSRegressionLearner(n_components=2, max_iter=100)]
    res = Orange.evaluation.CrossValidation(data, learners)
    for l, ca in zip(learners, Orange.evaluation.RMSE(res)):
        print("learner: {}\nRMSE: {}\n".format(l, ca))
