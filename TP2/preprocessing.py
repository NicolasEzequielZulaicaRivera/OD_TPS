# # Preprocesamiento

# ### Imports

import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso
from joblib import dump, load
from matplotlib import pyplot as plt

# ### Util

showPrints = False
showStatus = True

runTraining = False
saveTraining = True

if( showStatus ):
    print(f'[1/6] Loading Util {" "*20}', end='\r')


def join_df( feat, target ):
    return pd.merge(feat, target, on='id', how='inner')


def separate_df( joined ):
    return (
        joined.drop('llovieron_hamburguesas_al_dia_siguiente', 1),
        joined.llovieron_hamburguesas_al_dia_siguiente,
    )


# ### Dataset

df_feat = pd.read_csv("datasets/train_features.csv", low_memory=False)
df_targ = pd.read_csv("datasets/train_target.csv")

df_all = join_df(df_feat,df_targ)

if(showPrints):
    df_all.info()

# ## Tratamiento de Nulls

if( showStatus ):
    print(f'[2/6] Loading Null Preprocessing {" "*20}', end='\r')

if( runTraining ):
    meanImputer =  SimpleImputer(strategy='most_frequent')
    meanImputer.fit( df_feat )
    if( saveTraining ):
        dump(meanImputer, 'models/Preprocessing/meanImputer.sk') 
else:
    meanImputer = load('models/Preprocessing/meanImputer.sk')


def reemplazarNulls( feat, inplace=False ):
    # TODO: Use a better method
    _feat = feat
    if( not inplace ):
        _feat = feat.copy()
    _feat[:] = meanImputer.transform(feat)
    return _feat


# ### Dataset

if(showPrints):
    print( df_all.isna().sum() > 0 )
    reemplazarNulls(df_feat).isna().sum().sum() == 0

# ## Tratamiento de Categoricas

if( showStatus ):
    print(f'[3/6] Loading Categorical Preprocessing {" "*20}', end='\r')


# +
def reemplazarCategoricas_Barrio( feat ):
    if ('barrio' in feat):
        feat.drop('barrio',axis=1,inplace=True)
    return feat

def reemplazarCategoricas_Viento( feat ):
    dirViento= [
        'norte',
        'nornoreste',
        'noreste',
        'estenoreste',
        'este',
        'estesureste',
        'sureste',
        'sursureste',
        'sur',
        'sursuroeste',
        'suroeste',
        'oestesuroeste',
        'oeste',
        'oestenoroeste',
        'noroeste',
        'nornoroeste'
    ]
    valDirViento= [
        90,
        67.5,
        45,
        22.5,
        0,
        337.5,
        315,
        292.5,
        270,
        247.5,
        225,
        202.5,
        180,
        157.5,
        135,
        112.5
    ]
    def getDirVal( direction ):
        if ( not type(direction)==str ):
            return 0
        
        if ( not direction.lower() in dirViento ):
            return 0
    
        return valDirViento[ 
            dirViento.index( direction.lower() ) 
        ]
    
    feat['direccion_viento_temprano'] = feat['direccion_viento_temprano'].map(
        lambda direction: getDirVal( direction )
    )
    
    feat['direccion_viento_tarde'] = feat['direccion_viento_tarde'].map(
        lambda direction: getDirVal( direction )
    )
    
    feat['rafaga_viento_max_direccion'] = feat['rafaga_viento_max_direccion'].map(
        lambda direction: getDirVal( direction )
    )
    
    return feat

def reemplazarCategoricas_HamburguesasHoy( feat ):
    feat['llovieron_hamburguesas_hoy'] = ( feat['llovieron_hamburguesas_hoy'] == 'si' )
    return feat


# -

def reemplazarCategoricas( feat, inplace = False ):
    
    _feat = feat
    if( not inplace ):
        _feat = feat.copy()
    
    return reemplazarCategoricas_HamburguesasHoy(
        reemplazarCategoricas_Viento(
            reemplazarCategoricas_Barrio( _feat )
        )
    )


# ### Dataset

if (showPrints): 
    cat_labels = df_feat.select_dtypes(exclude=np.number).columns.tolist()
    cat_labels.remove('dia')
    
    for label in cat_labels:
        print(label)
        print( df_feat[label].unique() )
        
    cat_feat = reemplazarCategoricas(df_feat)
    
    for label in cat_labels:
        print(label)
        print( cat_feat[label].unique() if label in cat_feat else "Not Found")

# ## Tratamiento de Fechas

if( showStatus ):
    print(f'[4/6] Loading Date Preprocessing {" "*20}', end='\r')


def reemplazarFechas( feat, inplace = False, removeOriginal=True ):
    
    _feat = feat
    if( not inplace ):
        _feat = feat.copy()
        
    _feat['y'] = pd.to_numeric( _feat['dia'].map(lambda x: x.split('-')[0]), errors='coerce' )
    _feat['m'] = pd.to_numeric( _feat['dia'].map(lambda x: x.split('-')[1]), errors='coerce' )
    _feat['d'] = pd.to_numeric( _feat['dia'].map(lambda x: x.split('-')[2]), errors='coerce' )
    
    if( removeOriginal ):
        _feat.drop( 'dia', 1, inplace=True )

    return _feat


# ### Dataset

if (showPrints): 
    dat_feat = reemplazarFechas(df_feat,removeOriginal=False)
    display(dat_feat[['dia','y','m','d']])

# ## Tratamiento del Target

if( showStatus ):
    print(f'[5/6]Loading Target Preprocessing {" "*20}', end='\r')


def targetBooleano( target, inplace=False ):
    result = target.llovieron_hamburguesas_al_dia_siguiente == 'si'
    if (inplace):
        target = result
    return result


if (showPrints): 
    targetBooleano(df_targ)

# ## Regularizacion

if( showStatus ):
    print(f'[6/6] Loading Regularizaton {" "*20}', end='\r')

reg_feat = reemplazarFechas(reemplazarCategoricas(reemplazarNulls(df_feat)))

if( runTraining ):
    scaler = StandardScaler()
    scaler.fit( reemplazarFechas(reemplazarCategoricas(df_feat)) )
    if( saveTraining ):
        dump(scaler, 'models/Preprocessing/scaler.sk') 
else:
    scaler = load('models/Preprocessing/scaler.sk')

# +
if( runTraining ):
    lasso = Lasso(alpha=0.05)
    lasso.fit( 
        reg_feat,
        targetBooleano(df_targ)
    )
    if( saveTraining ):
        dump(lasso, 'models/Preprocessing/lasso.sk') 
else:
    lasso = load('models/Preprocessing/lasso.sk')
    
lasso_coef = pd.Series(lasso.coef_, index=reg_feat.columns).abs().sort_values()
# -

if (showPrints): 
    print( f' Seleccionadas: {sum(lasso_coef != 0)}  Eliminadas: {sum(lasso_coef == 0)}')
    lasso_coef.plot(kind="barh")


def regularizar( feat, inplace=False, drop=sum(lasso_coef == 0) ):
    _feat = feat
    if( not inplace ):
        _feat = feat.copy()
        
    # Normalize
    _feat[:] = scaler.transform(feat)
    
    # Drop less representative columns
    for i in range(0,drop):
        d_col = lasso_coef.axes[0][i]
        if d_col != 'id':
            _feat.drop( d_col, 1, inplace=True )  
    
    return _feat


if (showPrints):
    display(regularizar(reg_feat))

# ---

if( showStatus ):
    print(f'[###] All Done {" "*25}')
