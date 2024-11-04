# from torchtext.vocab import build_vocab_from_iterator
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer
import xgboost as xgb

# GLOBALS
SEED = 42
K_FOLDS = 5


def get_categorical_features():
    return [
        ]

def get_numerical_features():
    return [
        'total_distance_traveled',
        'avg_distance_traveled',
        'total_frames',
        'offense_shifts',
        'offense_motions',
        ]

class Model():
    def __init__(self, df, target_column):
        self.X = self.create_X(df)
        self.y = df[target_column].values


    def create_X(self, df):
        categorical_features = get_categorical_features()
        numerical_features = get_numerical_features()

        preprocessor = ColumnTransformer(
            transformers=[
                # ('cat', OneHotEncoder(drop='first', sparse_output=False), categorical_features),
                ('num', StandardScaler(), numerical_features)
            ]
        )
        return preprocessor.fit_transform(df)

    def predict(self, df):
        return self.model.predict_proba(self.create_X(df))[:, 1]

class LogisticRegressionModel(Model):
    def __init__(self, df, target_column):
        super().__init__(df, target_column)
        self.model = LogisticRegression(max_iter=1000, random_state=SEED).fit(self.X, self.y)

class XGBoostModel(Model):
    def __init__(self, df, target_column):
        super().__init__(df, target_column)
        self.model = xgb.XGBClassifier(max_depth=3, eval_metric='logloss', random_state=SEED).fit(self.X, self.y)

if __name__ == '__main__':
    import pickle
    import pandas as pd
    from sklearn.model_selection import KFold
    import os

    os.makedirs(f"models/lr/", exist_ok=True)
    os.makedirs(f"models/xgb/", exist_ok=True)

    kfold = KFold(n_splits=K_FOLDS, shuffle=True, random_state=SEED)

    y = 'isMan'
    df = pd.read_csv("data/2025/processed/Defense.csv")
    for fold, (train_idx, val_idx) in enumerate(kfold.split(df)):
        train = df.iloc[train_idx]
        print(f"Training LR model for fold {fold}")
        lr_model = LogisticRegressionModel(train, y)

        with open(f"models/manZone/lr/model_fold_{fold}.pkl", 'wb') as f:
            pickle.dump(lr_model, f)

    model = LogisticRegressionModel(df, y)
    with open(f"models/manZone/lr/model_full_data.pkl", 'wb') as f:
        pickle.dump(model, f)

    df = pd.read_csv("data/2025/processed/Defense.csv")
    for fold, (train_idx, val_idx) in enumerate(kfold.split(df)):
        train = df.iloc[train_idx]
        print(f"Training XGB model for fold {fold}")
        xgb_model = XGBoostModel(train, y)

        with open(f"models/manZone/xgb/model_fold_{fold}.pkl", 'wb') as f:
            pickle.dump(xgb_model, f)

    model = XGBoostModel(df, y)
    with open(f"models/manZone/xgb/model_full_data.pkl", 'wb') as f:
        pickle.dump(model, f)


    y = 'isRunPlay'
    df = pd.read_csv("data/2025/processed/Offense.csv")
    for fold, (train_idx, val_idx) in enumerate(kfold.split(df)):
        train = df.iloc[train_idx]
        print(f"Training LR model for fold {fold}")
        lr_model = LogisticRegressionModel(train, y)

        with open(f"models/runPass/lr/model_fold_{fold}.pkl", 'wb') as f:
            pickle.dump(lr_model, f)

    model = LogisticRegressionModel(df, y)
    with open(f"models/runPass/lr/model_full_data.pkl", 'wb') as f:
        pickle.dump(model, f)

    df = pd.read_csv("data/2025/processed/Offense.csv")
    for fold, (train_idx, val_idx) in enumerate(kfold.split(df)):
        train = df.iloc[train_idx]
        print(f"Training XGB model for fold {fold}")
        xgb_model = XGBoostModel(train, y)

        with open(f"models/runPass/xgb/model_fold_{fold}.pkl", 'wb') as f:
            pickle.dump(xgb_model, f)

    model = XGBoostModel(df, y)
    with open(f"models/runPass/xgb/model_full_data.pkl", 'wb') as f:
        pickle.dump(model, f)