import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_predict, KFold
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score, brier_score_loss
import random

class RandomSearchWithMetrics:
    def __init__(self, pipeline, params, X, y, n_iter=10, cv=5, path="", random_state=42, n_repeats=5):
        self.pipeline = pipeline
        self.params = params
        self.X = X
        self.y = y
        self.n_iter = n_iter
        self.cv = cv
        self.history = []
        self.path = path
        self.random_state = random_state
        self.n_repeats = n_repeats
        self.generated_params = set()

    def generate_random_params(self):
        params = {}
        for key, values in self.params.items():
            params[key] = random.choice(values)
        return params

    def fit_and_evaluate(self):

        random.seed(self.random_state)
        np.random.seed(self.random_state)

        for i in range(self.n_iter):
            while True:
                params = self.generate_random_params()
                params_tuple = tuple(sorted(params.items())) 
                if params_tuple not in self.generated_params:
                    self.generated_params.add(params_tuple)
                    break

            self.pipeline.set_params(**params)

            # lists to accumulate metrics across rounds
            f1_scores, accuracies, brier_scores, roc_aucs = [], [], [], []

            for j in range(self.n_repeats):
                # we dont set random_state here to get different splits for each repeat, 
                # becuase if we set random_state here, we will be fitting just one configuration of data
                kf = KFold(n_splits=self.cv, shuffle=True)
                y_pred = cross_val_predict(self.pipeline, self.X, self.y, cv=kf, method='predict')
                y_probabilities = cross_val_predict(self.pipeline, self.X, self.y, cv=kf, method='predict_proba')[:, 1]

                # calculate metrics
                f1_scores.append(f1_score(self.y, y_pred, average='weighted'))
                accuracies.append(accuracy_score(self.y, y_pred))
                brier_scores.append(brier_score_loss(self.y, y_probabilities))
                roc_aucs.append(roc_auc_score(self.y, y_probabilities))

            # average metrics across repeats
            avg_metrics = {
                'f1_mean': np.mean(f1_scores),
                'accuracy_mean': np.mean(accuracies),
                'brier_score_mean': np.mean(brier_scores),
                'roc_auc_mean': np.mean(roc_aucs)
            }

            # update history with averaged metrics
            avg_metrics.update(params)
            self.history.append(avg_metrics)

            self.save_results(path_to_save=self.path)

    def save_results(self, path_to_save=""):
        df_history = pd.DataFrame(self.history)
        df_history.to_csv(path_to_save, index=False)
