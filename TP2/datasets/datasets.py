# # Tratamiento Basico del Dataset

# ### Imports

import pandas as pd
from sklearn.model_selection import train_test_split

# ## Tratamiento

df_feat = pd.read_csv("features.csv", low_memory=False)
df_targ = pd.read_csv("target.csv")

df_all = pd.merge(df_feat, df_targ, on='id', how='inner')

df_all.info()

# ### Limpiar nulls del Target

df_all.dropna( subset=['llovieron_hamburguesas_al_dia_siguiente'], inplace=True )

# ### Forzar `presion_atmosferica_tarde` a numerico

# `presion_atmosferica_tarde` deberia ser de tipo numerica

df_all['presion_atmosferica_tarde'] = pd.to_numeric(
    df_all['presion_atmosferica_tarde'], errors='coerce'
)

# ### Resultado

df_all.info()

# ### Separar Train y Holdout

train_feat, holdout_feat, train_target, holdout_target = train_test_split(
    df_all.drop('llovieron_hamburguesas_al_dia_siguiente', 1),
    df_all.llovieron_hamburguesas_al_dia_siguiente,
    test_size=0.1,
    stratify=df_all.llovieron_hamburguesas_al_dia_siguiente,
    random_state=1
)
# TODO: VERIFY THIS IS CORRECT

train_feat.drop('id',1).to_csv('train_features.csv', index_label='id')
train_target.to_csv('train_target.csv', index_label='id')

holdout_feat.drop('id',1).to_csv('holdout_features.csv', index_label='id')
holdout_target.to_csv('holdout_target.csv', index_label='id')
