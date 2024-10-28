import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_predict, KFold
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score, brier_score_loss
from scipy.stats import uniform
import random

class RandomSearchWithMetrics:
    def __init__(self, pipeline, param_dist, X, y, n_iter=10, cv=5, path="", random_state=42):
        self.pipeline = pipeline
        self.param_dist = param_dist
        self.X = X
        self.y = y
        self.n_iter = n_iter
        self.cv = cv
        self.history = []
        self.path = path
        self.random_state = random_state

    def generate_random_params(self):
        params = {}
        for key, values in self.param_dist.items():
            if isinstance(values, list):
                params[key] = random.choice(values)
            elif hasattr(values, 'rvs'):
                params[key] = values.rvs()
        return params

    def fit_and_evaluate(self):
        # Устанавливаем сид для воспроизводимости
        random.seed(self.random_state)
        np.random.seed(self.random_state)

        for i in range(self.n_iter):
            # random generation of parameters
            params = self.generate_random_params()
            self.pipeline.set_params(**params)

            # cross-validation
            kf = KFold(n_splits=self.cv, shuffle=True, random_state=self.random_state)
            y_pred = cross_val_predict(self.pipeline, self.X, self.y, cv=kf, method='predict')
            y_probabilities = cross_val_predict(self.pipeline, self.X, self.y, cv=kf, method='predict_proba')[:, 1]

            # calculating metrics
            f1 = f1_score(self.y, y_pred, average='weighted')
            accuracy = accuracy_score(self.y, y_pred)
            brier_score = brier_score_loss(self.y, y_probabilities)
            roc_auc = roc_auc_score(self.y, y_probabilities)

            # dictionary with metrics
            metrics = {
                'f1': f1,
                'accuracy': accuracy,
                'brier_score': brier_score,
                'roc_auc': roc_auc
            }

            # update history
            metrics.update(params)
            self.history.append(metrics)

            self.save_results(path_to_save=self.path)

    def save_results(self, path_to_save=""):
        df_history = pd.DataFrame(self.history)
        df_history.to_csv(path_to_save, index=False)
