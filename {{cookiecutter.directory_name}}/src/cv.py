import pandas as pd
import numpy as np
from sklearn import model_selection
import datetime as dt
import xgboost
from sklearn import metrics

from src import config

def cv(df, cols_to_drop, xgboost_params=config.xgboost_params):
    X = df.drop(columns=['target'] + cols_to_drop)
    y = df.target

    started = dt.datetime.now()
    kf      = model_selection.RepeatedKFold(n_repeats=1, n_splits=10)
    cv_perf = {'kf train': [], 'kf test': [], 'evals_result': []}

    for i, (train_index, test_index) in enumerate(kf.split(X)):

        kf_X_train, kf_X_test = X.iloc[train_index], X.iloc[test_index]
        kf_y_train, kf_y_test = y.iloc[train_index], y.iloc[test_index]

        model = xgboost.XGBClassifier(**xgboost_params)
        _ = model.fit(kf_X_train, kf_y_train, verbose=False, eval_metric=["error"],
                      eval_set=[(kf_X_train, kf_y_train), (kf_X_test, kf_y_test)]) # early_stopping_rounds=100

        kf_train_pred = model.predict(kf_X_train)
        kf_test_pred  = model.predict(kf_X_test)

        evals_result = {k: v['error'] for k, v in zip(['train', 'test'], model.evals_result().values())}

        cv_perf['kf train'    ].append(metrics.accuracy_score(kf_y_train, kf_train_pred, normalize=True))
        cv_perf['kf test'     ].append(metrics.accuracy_score(kf_y_test , kf_test_pred , normalize=True))
        cv_perf['evals_result'].append(evals_result)

        tr_cummean = np.mean(cv_perf['kf train'])
        te_cummean = np.mean(cv_perf['kf test'])

        print(f'Iteration #{i+1:02}. Elapsed: {dt.datetime.now()-started}. Cum. Accuracy: test: {te_cummean:.2%}, train: {tr_cummean:.2%}')