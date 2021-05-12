import xgboost
from .config import xgboost_params
from sklearn.model_selection import train_test_split

def train(df, cols_to_drop, target_col='target'):
    X_train, X_test, y_train, y_test = train_test_split(df.drop(columns=target_col), df[target_col])
    
    model = xgboost.XGBClassifier(**xgboost_params)
    
    _ = model.fit(
        X_train.drop(columns=cols_to_drop), 
        y_train, 
        verbose=50, 
        eval_metric=["error"],
        eval_set=[
            (X_train.drop(columns=cols_to_drop), y_train), 
            (X_test.drop(columns=cols_to_drop), y_test)
        ])
    
    return model