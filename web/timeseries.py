from sklearn.ensemble import RandomForestClassifier
from catboost import CatBoostClassifier
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from tqdm import tqdm


total = []
y = []

class TimeSeries:
    def __init__(self):
        """A class for a Time Series Stemming model that will make model-based predictions,
        which were trained on data over different periods of time. The class implements the following methods:
        fit, predict_proba, submit, init_models, time_series_predict.
        """
        self.rf_model1 = RandomForestClassifier(5000, n_jobs=-1)
        self.rf_model2 = RandomForestClassifier(5000, n_jobs=-1)
        self.cb_model1 = CatBoostClassifier(2000, verbose=False)
        self.cb_model2 = CatBoostClassifier(2000, verbose=False)

    def fit(self, X, y):
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
        self.X = X.copy()
        self.y = y.map(int)

        self.X['score_from_cross_val'] = [0] * len(X)
        self.X1, self.X2, self.y1, self.y2 = train_test_split(self.X, self.y, test_size=0.5, shuffle=False)

        self.rf_model1.fit(self.X1, self.y1)
        self.rf_model2.fit(self.X2, self.y2)

        self.cb_model1.fit(self.X1, self.y1)
        self.cb_model2.fit(self.X2, self.y2)

    def predict_proba(self):
        """Prediction probability algorithms to data."""
        self.p1 = self.rf_model2.predict_proba(self.X1)[:, 1]
        self.p2 = self.cb_model2.predict_proba(self.X1)[:, 1]
        self.preds1 = self.p1 * 0.8 + self.p2 * 0.2

        self.X1['score_from_cross_val'] = self.preds1

        self.p3 = self.rf_model1.predict_proba(self.X2)[:, 1]
        self.p4 = self.cb_model1.predict_proba(self.X2)[:, 1]
        self.preds2 = self.p3 * 0.8 + self.p4 * 0.2

        self.X2['score_from_cross_val'] = self.preds2

        return pd.concat([self.X1, self.X2])

    def submit(self, X):
        """Prediction probability algorithms to data.
        Parameters
        ----------
        X : array-like,
            Testing vectors, where rows is the number of samples
            and columns is the number of features.
        """
        X1, X2, y1, y2 = train_test_split(X, y, test_size=0.5, shuffle=False)

        p1 = self.rf_model2.predict_proba(X1)[:, 1]
        p2 = self.cb_model2.predict_proba(X1)[:, 1]
        preds1 = p1 * 0.8 + p2 * 0.2

        X1['score_from_cross_val'] = preds1

        p3 = self.rf_model1.predict_proba(X2)[:, 1]
        p4 = self.cb_model1.predict_proba(X2)[:, 1]
        preds2 = p3 * 0.8 + p4 * 0.2

        X2['score_from_cross_val'] = preds2

        return pd.concat([X1, X2])

    def init_models(self, data):
        """Fit Classification algorithms to data.
        Parameters
        ----------
        data : array-like,
            Testing vectors, where rows is the number of samples
            and columns is the number of features.
        """
        total3 = total + ['contractor_id', 'score_from_cross_val', 'score']
        for_each = data[total3].groupby('contractor_id')
        self.models = {}
        for name, group in tqdm(for_each):
            if group.shape[0] > 20:
                mod = CatBoostClassifier(allow_const_label=True, verbose=False)
                mod.fit(group.drop(columns=['score']), group['score'])
                self.models[name] = mod
        print("DONE")

    def time_series_predict(self, data):
        """Prediction probability algorithms to data.
        Parameters
        ----------
        X : array-like,
            Testing vectors, where rows is the number of samples
            and columns is the number of features.
        """
        tt = data.groupby('contractor_id')
        fn = pd.DataFrame(columns=["c", 'score2'])
        count = 0
        # total_ts = total + ['contractor_id', 'score_from_cross_val']
        for name, group in tqdm(tt):
            if name in self.models:
                ids = group['c']
                predd = self.models[name].predict_proba(group)[:, 1] #[total_ts]
                mem = pd.DataFrame({"c": ids, 'score2': predd})
                fn = pd.concat([fn, mem])
                count += 1

        submit2 = data.merge(fn, on='c', how="left")
        submit2['score'][submit2['score2'].notnull()] = (2 * submit2['score'] + submit2['score2']) / 3
        # print(submit2.score2)
        return (submit2.drop(columns=['score2', 'c']), submit2)

    def save_model(self, path: str):
        """Model save algorithms.
        Parameters
        ----------
        path : str,
            String with path you need to save model.
        """
        joblib.dump(self.models, path)

    @classmethod
    def load_model(cls, path: str):
        """Load the models from a file and return an instance of TimeSeries."""
        instance = cls()
        instance.models = joblib.load(path)
        print(f"Models loaded from {path}")
        return instance