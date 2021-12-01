# ### Imports

from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import pandas as pd
from sklearn.model_selection import train_test_split


# ### Score

def score(target, predictions):
    print(f'ACCURACY: {accuracy_score(target, predictions)}')


# ### Get Feat/ Targ

df_feat = pd.read_csv("datasets/train_features.csv", low_memory=False)
df_targ = pd.read_csv("datasets/train_target.csv")


# ### Divide Train / Val

def train_val( feat, targ ):
    return train_test_split(
        feat,
        targ,
        test_size=0.1,
        stratify=targ.llovieron_hamburguesas_al_dia_siguiente,
        random_state=1
    )


train_feat, holdout_feat, train_target, holdout_target = train_val( df_feat, df_targ )
