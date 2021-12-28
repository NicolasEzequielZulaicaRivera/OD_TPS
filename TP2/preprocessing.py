# # Preprocesamiento

# ### Imports

import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso
from sklearn.feature_selection import SelectKBest, f_classif as f_classif
from joblib import dump, load
from matplotlib import pyplot as plt

# ### Util

showPrints = False # False
showStatus = True # True

runTraining = False # False
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

df_feat = pd.read_csv("datasets/train_features.csv", low_memory=False).set_index('id')
df_targ = pd.read_csv("datasets/train_target.csv")

df_all = join_df(df_feat,df_targ)

if(showPrints):
    df_all.info()

df_feat.columns

# ## Tratamiento de Nulls

if( showStatus ):
    print(f'[2/6] Loading Null Preprocessing {" "*20}', end='\r')

if( runTraining ):
    imputer =  SimpleImputer(strategy='most_frequent')
    imputer.fit( df_feat )
    if( saveTraining ):
        dump(imputer, 'models/Preprocessing/imputer.sk') 
else:
    imputer = load('models/Preprocessing/imputer.sk')


def reemplazarNulls( feat, inplace=False ):
    # TODO: Use a better method
    _feat = feat
    if( not inplace ):
        _feat = feat.copy()
    _feat[:] = imputer.transform(_feat)
    return _feat


def reemplazarNullsNum( feat, inplace=False ):
    _feat = feat
    if( not inplace ):
        _feat = feat.copy()
        
    numLabels = df_feat.select_dtypes(include=np.number).columns.tolist()
    
    for label in numLabels:
        if label not in _feat:
            continue
        # anadir feture que marca missing
        # y poner esta como el mean
        _feat["missing_"+label] = _feat[label].isna()
        _feat[label] = _feat[label].fillna( df_feat[label].mean() )
    
    return _feat


# ### Dataset

if(showPrints):
    print( df_all.isna().mean()*100 )
    print( f'\n Nulls Reemplazados: {reemplazarNulls(df_feat).isna().sum().sum() == 0}' )
    display(reemplazarNulls(df_feat))

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
    _target = target
    if( not inplace ):
        _target = target.copy()
        
    _target.llovieron_hamburguesas_al_dia_siguiente = _target.llovieron_hamburguesas_al_dia_siguiente == 'si'
    
    return _target


if (showPrints): 
    display(targetBooleano(df_targ))

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
        targetBooleano(df_targ).llovieron_hamburguesas_al_dia_siguiente
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


def stdScaleFunc( x, c1, c2=50 ):
    return x * (1-c1*c2)


def regularizar( feat, inplace=False, drop=sum(lasso_coef == 0), scaleFunc=None ):
    _feat = feat
    if( not inplace ):
        _feat = feat.copy()
        
    # Normalize
    _feat[:] = scaler.transform(_feat)
    
    # Scale
    if( type(scaleFunc) != type(None) ):
        if( scaleFunc == 'std' ):
            scaleFunc = stdScaleFunc
        for i in range(drop+1,len(lasso_coef)):
            j = lasso_coef.axes[0][i]
            _feat[ j ] = stdScaleFunc( _feat[ j ], lasso_coef[i] )
    
    # Drop less representative columns
    for i in range(0,drop):
        d_col = lasso_coef.axes[0][i]
        if d_col != 'id':
            _feat.drop( d_col, 1, inplace=True )  
    
    return _feat


if (showPrints):
    display(regularizar(reg_feat))
    display(regularizar(reg_feat, scaleFunc='std'))

# ---

if( showStatus ):
    print(f'[###] Initial Preprocessings Done {" "*25}')

# ---

# ## One Hot Encoding

if( showStatus ):
    print(f'[1/2] Loading One Hot Encoding {" "*20}', end='\r')


def reemplazarCategoricas_OHE( feat, inplace = False ):
    
    categoricas = ['barrio','direccion_viento_temprano','direccion_viento_tarde','rafaga_viento_max_direccion','llovieron_hamburguesas_hoy']
    
    _feat = feat
    if( not inplace ):
        _feat = feat.copy()
    
    for label in categoricas:
        _feat[label] = _feat[label].str.lower()
        
    _feat = pd.get_dummies(_feat,columns = categoricas)
        
    return _feat


if(showPrints):
    ohercat = reemplazarFechas(reemplazarCategoricas_OHE(df_feat))
    ohernull =reemplazarNullsNum(ohercat)
    print( f'{ohernull.isna().sum().sum()} missings' )
    display(ohernull)
    print("-------")
    ohernull.info(verbose=True)

# ### Seleccionar variables

if( runTraining ):
    ohe_varsel = SelectKBest(f_classif, k=20).fit(ohernull, df_targ.llovieron_hamburguesas_al_dia_siguiente)
    if(saveTraining):
        dump(ohe_varsel, 'models/Preprocessing/ohe_varsel.sk') 
else:
    ohe_varsel = load('models/Preprocessing/ohe_varsel.sk')

ohe_feats = pd.DataFrame(
    data={
        'feature':ohe_varsel.feature_names_in_,
        'pvalue':ohe_varsel.pvalues_,
        'scores':ohe_varsel.scores_,
    }
).sort_values(by=['scores'], ascending=False)

if(showPrints):
    display(
        ohe_feats.head(20)
    )


def keepFeat_OHE( feat, keep = 20, inplace = False ):

    res = feat[ ohe_feats.head(keep).feature.array  ]
    
    if(inplace):
        feat = res
        
    return res


if(showPrints):
    display(
        keepFeat_OHE( ohernull )
    )

# ## Hashing Trick

if( showStatus ):
    print(f'[2/2] Loading Hashing Trick {" "*20}', end='\r')


def hash_col(df, col, N):
    def hashrow(x): tmp = [((hash(x)*(i+1)) % (N+1)) for i in range(N)]; return pd.Series(tmp,index=cols)
    cols = [col + "_" + str(i) for i in range(N)]
    df[cols] = df[col].apply(hashrow)
    df.drop(col,axis=1, inplace=True)
    return df


def reemplazarCategoricas_HashTrick( feat, inplace = False ):
    
    categoricas = ['barrio','direccion_viento_temprano','direccion_viento_tarde','rafaga_viento_max_direccion']
    oheable = ['llovieron_hamburguesas_hoy']
    
    _feat = feat
    if( not inplace ):
        _feat = feat.copy()
    
    for label in oheable:
        _feat[label] = _feat[label].str.lower()
    _feat = pd.get_dummies(_feat,columns = oheable)
    
    for label in categoricas:
        _feat[label] = _feat[label].str.lower()
        hash_col(_feat,label,4)
        
    return _feat


if(showPrints or runTraining):
    # This can take 2 min
    htrcat = reemplazarFechas(reemplazarCategoricas_HashTrick(df_feat))
    htrnull = reemplazarNullsNum(htrcat)

if(showPrints):
    print( f'{htrnull.isna().sum().sum()} missings' )  
    display(htrnull)
    print("-------")
    htrnull.info(verbose=True)

# ### Normalizar

if( runTraining ):
    scaler_ht = StandardScaler()
    scaler_ht.fit( htrnull )
    if( saveTraining ):
        dump(scaler_ht, 'models/Preprocessing/scaler_ht.sk') 
else:
    scaler_ht = load('models/Preprocessing/scaler_ht.sk')


def normalizar_HashTrick( feat, inplace = False ):
    _feat = feat
    if( not inplace ):
        _feat = feat.copy()
        
    # Normalize
    _feat[:] = scaler_ht.transform(_feat)
    
    return _feat


if( showPrints ):
    htr2 = normalizar_HashTrick(htrnull)
    display(htr2)

# ---

if( showStatus ):
    print(f'[###] Aditional Preprocessings Done {" "*50}')
