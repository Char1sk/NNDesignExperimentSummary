RANDOM_STATE = 42

from self_paced_ensemble import SelfPacedEnsembleClassifier
from self_paced_ensemble.canonical_ensemble import *
from self_paced_ensemble.utils import load_covtype_dataset
from self_paced_ensemble.self_paced_ensemble.base import sort_dict_by_key

from time import time
from collections import Counter
import matplotlib.pyplot as plt

from sklearn.decomposition import KernelPCA
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score

X_train, X_test, y_train, y_test = load_covtype_dataset(subset=0.1, random_state=RANDOM_STATE)

origin_distr = sort_dict_by_key(Counter(y_train))
test_distr = sort_dict_by_key(Counter(y_test))

init_kwargs = {
    'n_estimators': 10,
    'random_state': RANDOM_STATE,
}
fit_kwargs = {
    'X': X_train,
    'y': y_train,
}

ensembles = {
    'SelfPacedEnsemble': SelfPacedEnsembleClassifier,
    'SMOTEBagging': SMOTEBaggingClassifier,
    'RUSBoost': RUSBoostClassifier,
}

fit_ensembles = {}
for ensemble_name, ensemble_class in ensembles.items():
    ensemble_clf = ensemble_class(**init_kwargs)
    print ('Training {:^20s} '.format(ensemble_name), end='')
    start_time = time()
    ensemble_clf.fit(X_train, y_train)
    fit_time = time() - start_time
    y_pred = ensemble_clf.predict_proba(X_test)[:, 1]
    score = average_precision_score(y_test, y_pred)
    print ('| AUPRC {:.3f} | Time {:.3f}s'.format(score, fit_time))