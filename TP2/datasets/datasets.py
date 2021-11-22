# # Tratamiento Basico del Dataset

# ### Imports

import pandas as pd
from sklearn.model_selection import train_test_split

# ## Tratamiento

df_feat = pd.read_csv("features.csv", low_memory=False)
df_targ = pd.read_csv("target.csv")

df_all = pd.merge(df_feat, df_targ, on='id', how='inner')

# ### Limpiar nulls del Target

df_all.dropna( subset=['llovieron_hamburguesas_al_dia_siguiente'], inplace=True )

# ### Separar Train y Holdout

train_feat, holdout_feat, train_target, holdout_target = train_test_split(
    df_all.drop('llovieron_hamburguesas_al_dia_siguiente', 1),
    df_all.llovieron_hamburguesas_al_dia_siguiente,
    test_size=0.1,
    stratify=df_all.llovieron_hamburguesas_al_dia_siguiente,
    random_state=1
)

train_feat.to_csv('train_features.csv')
train_target.to_csv('train_target.csv')

holdout_feat.to_csv('holdout_features.csv')
holdout_target.to_csv('holdout_target.csv')
