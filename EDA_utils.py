#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from functools import reduce
from sklearn.preprocessing import PolynomialFeatures, StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline, FeatureUnion
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.preprocessing import FunctionTransformer
from sklearn.compose import make_column_transformer


# In[2]:


class make_column_transformerz(TransformerMixin):
    def __init__(self, estimator, col_list):
        self.estimator=estimator
        self.col_list=col_list
        self.make_= None
    def fit(self, X, y=0):
        self.make_= make_column_transformer((self.estimator, self.col_list), remainder='drop')
        self.make_.fit(X)
        return self
    def transform(self,X):
        dummy=self.make_.transform(X)
        #cols=list(X.columns).remove(str(self.col_list))
        transformed=pd.DataFrame(dummy, columns=self.col_list)
        X.drop(self.col_list, axis=1, inplace=True)
        X=pd.concat([X,transformed], axis=1, join='inner')
        return X

class columnseparater(TransformerMixin):
    def __init__(self, cols):
        self.cols=cols
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        X_new = X[self.cols]
        return X_new

class standardscaler(TransformerMixin):
    def __init__(self):
        self.ss=None
        self.mean_=None
        self.scale_=None
    def fit(self, X, y=None):
        self.ss=StandardScaler()
        self.ss.fit(X)
        self.mean_ = pd.Series(self.ss.mean_, index=X.columns)
        self.scale_ = pd.Series(self.ss.scale_, index=X.columns)
        return self
    def transform(self, X):
        x=self.ss.transform(X)
        numeric=pd.DataFrame(x, columns=X.columns)
        return numeric

class categorical_data(TransformerMixin):
    def __init__(self):
        self.value=None
        self.categories_=None
        self.columns=[]
    def fit(self, X, y=None):
        self.value=OneHotEncoder()
        self.value.fit(X)
        self.categories_=pd.Series(self.value.categories_)
        return self
    def transform(self, X):
        x=self.value.transform(X)
        for i in range(len(self.categories_)):
            self.columns+=list(self.categories_[i]) #columns remove pannuna work aaghudhu
        cate=pd.DataFrame(x.toarray())
        return cate

class DFFeatureUnion(TransformerMixin):
    # FeatureUnion but for pandas DataFrames
    def __init__(self, transformer_list):
        self.transformer_list = transformer_list

    def fit(self, X, y=None):
        for t in self.transformer_list:
            t.fit(X)
        return self

    def transform(self, X):
        # assumes X is a DataFrame
        Xts = [t.transform(X) for t in self.transformer_list]
        Xunion = reduce(lambda X1, X2: pd.merge(X1, X2, left_index=True, right_index=True), Xts)
        return Xunion

