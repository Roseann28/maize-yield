"""Preprocessing: capping, imputation, scaling and encoding, as separate numeric and categorical pipelines."""
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler, StandardScaler


class Winsorizer(BaseEstimator, TransformerMixin):
    """Caps each column at percentiles learned on the TRAINING data only."""
    def __init__(self, lower=1, upper=99):
        self.lower, self.upper = lower, upper
    def fit(self, X, y=None):
        X = np.asarray(X, float)
        self.lo_ = np.nanpercentile(X, self.lower, axis=0)
        self.hi_ = np.nanpercentile(X, self.upper, axis=0)
        self.n_features_in_ = X.shape[1]
        return self
    def transform(self, X):
        return np.clip(np.asarray(X, float), self.lo_, self.hi_)
    def get_feature_names_out(self, names=None):
        return np.asarray(names, dtype=object)
    def set_output(self, *, transform=None):
        return self


def choose_scalers(data, cols, cont_cols):
    """Skewed columns (|skew| > 1 after capping) get the RobustScaler, the rest the StandardScaler."""
    use = [c for c in cont_cols if c in cols]
    capped = data[use].clip(data[use].quantile(.01), data[use].quantile(.99), axis=1)
    skew = capped.skew()
    return [c for c in use if abs(skew[c]) > 1], [c for c in use if abs(skew[c]) <= 1]


def build_preprocessor(cont_robust, cont_standard, binary, categorical, scale=True):
    """Continuous: cap -> median impute (+ missing indicator) -> scale. Binary: most frequent impute (+ indicator).
    Categorical: impute 'missing' -> one hot. Imputation is always present because production data can have gaps."""
    def cont(scaler):
        steps = [("cap", Winsorizer(1, 99)), ("impute", SimpleImputer(strategy="median", add_indicator=True))]
        if scale:
            steps.append(("scale", scaler))
        return Pipeline(steps)
    parts = []
    if cont_robust:
        parts.append(("cont_robust", cont(RobustScaler()), cont_robust))
    if cont_standard:
        parts.append(("cont_standard", cont(StandardScaler()), cont_standard))
    if binary:
        parts.append(("binary", Pipeline([("impute", SimpleImputer(strategy="most_frequent", add_indicator=True))]), binary))
    if categorical:
        parts.append(("categorical", Pipeline([
            ("impute", SimpleImputer(strategy="constant", fill_value="missing")),
            ("onehot", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=100, sparse_output=False))]), categorical))
    return ColumnTransformer(parts, remainder="drop", verbose_feature_names_out=True)
