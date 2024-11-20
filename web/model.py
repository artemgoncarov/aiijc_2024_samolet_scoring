from sklearn.ensemble import RandomForestClassifier
from catboost import CatBoostClassifier
import pandas as pd
import joblib


class Model:
    def __init__(self):
        """Model class, which is based on 2 classification models: RandomForestClassifier and CatBoostClassifier.
        The class implements the fit, predict_proba and save_model methods.
        """
        self.best_rf_params = {
            'bootstrap': True,
            'criterion': 'gini',
            'max_depth': None,
            'max_features': 'sqrt',
            'max_leaf_nodes': None,
            'max_samples': None,
            'min_samples_leaf': 1,
            'min_samples_split': 2,
            'n_estimators': 5000,
            'n_jobs': -1,
            'verbose': 0,
        }
        self.best_cb_params = {  # лучшие параметры для кота
            'iterations': 2000,
            'l2_leaf_reg': 3,
            'random_seed': 0,
            'leaf_estimation_iterations': 10,
            'max_leaves': 64,
            'depth': 6,
            'learning_rate': 0.02292199992,
            'random_strength': 1,
            'border_count': 254,
            'verbose': 0,
            "thread_count": -1,
        }
        self.rf_model = RandomForestClassifier(**self.best_rf_params)
        self.cb_model = CatBoostClassifier(**self.best_cb_params)

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """Fit Classification algorithms to X_train and y_train.
        Parameters
        ----------
        X : array-like,
            Training vectors, where rows is the number of samples
            and columns is the number of features.
        y : array-like,
            Training vectors, where rows is the number of samples
            and columns is the number of features.
        """
        self.rf_model.fit(X, y.map(int))
        self.cb_model.fit(X, y.map(int))

    def predict_proba(self, data: pd.DataFrame, class_=1) -> pd.Series:
        """Prediction probability algorithms to data.
        Parameters
        ----------
        data : array-like,
            Testing vectors, where rows is the number of samples
            and columns is the number of features.
        class_ : int,
            Number of class you need to predict
        Returns
        -------
        predictions : Pandas DataFrame
            Returns predictions of all the models in a Pandas DataFrame.
        """
        self.preds1 = self.rf_model.predict_proba(data)[:, class_]
        self.preds2 = self.cb_model.predict_proba(data)[:, class_]
        return self.preds1 * 0.8 + self.preds2 * 0.2

    def save_model(self, path_rf: str, path_cb: str):
        """Model save algorithms.
        Parameters
        ----------
        path : str,
            String with path you need to save model.
        """
        joblib.dump(self.rf_model, path_rf)
        self.cb_model.save_model(path_cb)
    
    def load_model(self, path_rf: str, path_cb: str):
        """Model load algorithms.
        Parameters
        ----------
        path : str,
            String with path you need to save model.
        """
        self.rf_model = joblib.load(path_rf)
        self.cb_model = CatBoostClassifier()
        self.cb_model.load_model(path_cb)
        return self