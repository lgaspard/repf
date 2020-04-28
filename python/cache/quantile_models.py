from sklearn.base import BaseEstimator
from skgarden import ExtraTreesQuantileRegressor
from sklearn.ensemble import GradientBoostingRegressor


class QuantileExtraTrees(BaseEstimator):
    """
    Wrapper for a 3-predictions Extra Trees
     - lower prediction: `self.lower_quantile`-quantile regression
     - median prediction: .5-quantile regression
     - upper prediction: `self.upper_quantile`-quantile regression
    """
    
    def __init__(self, lower_quantile, upper_quantile, n_estimators=100,
                 max_features='auto', n_jobs=-1, min_samples_split=10):
        
        super(QuantileExtraTrees, self).__init__()
        
        self.n_estimators = n_estimators
        self.lower_quantile = lower_quantile
        self.upper_quantile = upper_quantile
        self.max_features = max_features
        self.min_samples_split = min_samples_split
        self.n_jobs=-1
        
        self.model = ExtraTreesQuantileRegressor(
            max_features=max_features,
            n_estimators=n_estimators,
            n_jobs=n_jobs,
            min_samples_split=min_samples_split)
        
    def fit(self, X, y):
        """
        Fits the Quantile Extra Trees on the data.
        """
        self.model.fit(X, y)
        return self
    
    def predict(self, X):
        """
        Returns the prediction for the three quantiles: self.lower_quantile,
        self.upper_quantile and 0.5
        """
        l = self.model.predict(X, quantile=self.lower_quantile)
        m = self.model.predict(X, quantile=.5)
        u = self.model.predict(X , quantile=self.upper_quantile)
        return l, m, u
    
    def get_params(self, deep=False):
        params = {
            'n_estimators': self.n_estimators,
            'lower_quantile': self.lower_quantile,
            'upper_quantile': self.upper_quantile,
            'max_features': self.max_features,
            'min_samples_split': self.min_samples_split
        }
        if deep:
            params = {**params, **self.model.get_params(deep=deep)}
        return params

    def set_params(self, **kwargs):
        if 'lower_quantile' in kwargs:
            self.lower_quantile = kwargs['lower_quantile']
        if 'upper_quantile' in kwargs:
            self.upper_quantile = kwargs['upper_quantile']
        if 'n_estimators' in kwargs:
            self.n_estimators = kwargs['n_estimators']
            self.model.set_params(n_estimators=self.n_estimators)
        if 'max_features' in kwargs:
            self.max_features = kwargs['max_features']
            self.model.set_params(max_features=self.max_features)
        if 'n_jobs' in kwargs:
            self.n_jobs = kwargs['n_jobs']
            self.model.set_params(n_jobs=self.n_jobs)
        if 'min_samples_split' in kwargs:
            self.min_samples_split = min_samples_split
            self.model.set_params(min_samples_split=self.min_samples_split)
        return self


class QuantileGradientBoosting(BaseEstimator):
    """
    Wrapper for 3 Gradient Boosting Estimators:
     - self.lower: a quantile gradient boosting for the lower quantile
     - self.upper: a quantile gradient boosting for the upper quantile
     - self.mean: a least-square gradient boosting
    """

    def __init__(self, lower_quantile, upper_quantile, n_estimators=100,
                 learning_rate=.1):

        super(QuantileGradientBoosting, self).__init__()

        self.n_estimators = n_estimators
        self.lower_quantile = lower_quantile
        self.upper_quantile = upper_quantile
        self.learning_rate = learning_rate

        self.lower = GradientBoostingRegressor(loss='quantile',
                                               alpha=lower_quantile,
                                               n_estimators=n_estimators,
                                               learning_rate=learning_rate)
        self.upper = GradientBoostingRegressor(loss='quantile',
                                               alpha=upper_quantile,
                                               n_estimators=n_estimators,
                                               learning_rate=learning_rate)
        self.mean = GradientBoostingRegressor(loss='ls', criterion='mse',
                                              n_estimators=n_estimators,
                                              learning_rate=learning_rate)

    def fit(self, X, y):
        """
        Fits the three estimators on the data.
        """
        self.lower.fit(X, y)
        self.upper.fit(X, y)
        self.mean.fit(X, y)
        return self

    def predict(self, X):
        """
        Returns the prediction of the three estimators in a 3-tuple: lower, 
        mean, and upper.
        """
        l = self.lower.predict(X)
        m = self.mean.predict(X)
        u = self.upper.predict(X)
        return l, m, u

    def get_params(self, deep=False):
        params = {
            'n_estimators': self.n_estimators,
            'lower_quantile': self.lower_quantile,
            'upper_quantile': self.upper_quantile,
            'learning_rate': self.learning_rate
        }
        if deep:
            params = {**params, **self.lower.get_params(deep=deep),
                      **self.upper.get_params(deep=deep),
                      **self.mean.get_params(deep=deep)}
        return params

    def set_params(self, **kwargs):
        if 'n_estimators' in kwargs:
            self.n_estimators = kwargs['n_estimators']
            self.lower.set_params(n_estimators=self.n_estimators)
            self.upper.set_params(n_estimators=self.n_estimators)
            self.mean.set_params(n_estimators=self.n_estimators)
        if 'lower_quantile' in kwargs:
            self.lower_quantile = kwargs['lower_quantile']
            self.lower.set_params(alpha=self.lower_quantile)
        if 'upper_quantile' in kwargs:
            self.upper_quantile = kwargs['upper_quantile']
            self.upper.set_params(alpha=self.upper_quantile)
        if 'learning_rate' in kwargs:
            self.learning_rate = kwargs['learning_rate']
            self.lower.set_params(learning_rate=self.learning_rate)
            self.upper.set_params(learning_rate=self.learning_rate)
            self.mean.set_params(learning_rate=self.learning_rate)
        return self
