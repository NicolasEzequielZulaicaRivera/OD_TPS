# ### Imports

from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, confusion_matrix, f1_score
import pandas as pd
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np

sns.set()

# ### Code

showPrints = False


# ### Score

def fmtAvg( num, total ):
    num = 100*float(num)/total
    return f'{"%.2f" % num}%'


def score(target, predictions=None, probas=None):
    
    if( type(probas) != type(None) ):
        print(f'AUC-ROC: {roc_auc_score(target, probas)}')

    if( type(predictions) != type(None) ):
        print(f'ACCURACY: {accuracy_score(target, predictions)}')
        print(f'PRESICION: {precision_score(target, predictions)}')
        print(f'RECALL: {recall_score(target, predictions)}')
        print('CONFUSION MATRIX')
        conf_matrix = confusion_matrix(target, predictions,labels=[1,0])
        hmap=sns.heatmap(
            conf_matrix,
            cmap="YlOrBr",
            annot=True,
            yticklabels=['Real True','Real False'],
            xticklabels=['Pred True','Pred False'],
        )
        tarSize = target.size
        hmap.texts[0].set_text(fmtAvg(hmap.texts[0].get_text(),tarSize) + " TP")
        hmap.texts[1].set_text(fmtAvg(hmap.texts[1].get_text(),tarSize) + " FN")
        hmap.texts[2].set_text(fmtAvg(hmap.texts[2].get_text(),tarSize) + " FP")
        hmap.texts[3].set_text(fmtAvg(hmap.texts[3].get_text(),tarSize) + " TN")


def score2(target, predictions=None, probas=None):
    
    if( type(probas) != type(None) ):
        roc_auc = roc_auc_score(target, probas)

    if( type(predictions) != type(None) ):
        acc = accuracy_score(target, predictions)
        prec = precision_score(target, predictions)
        rec = recall_score(target, predictions)
        f1 = f1_score(target, predictions)
        
    return ( roc_auc, acc, prec, rec, f1 )


true = pd.Series([True,True,True,False,False,False])
pred = pd.Series([True,True,True,False,False,True])
prob = np.array([[0.9,0.8,0.7,0.3,0.2,0.6]])

if( showPrints ):
    display(score(true,pred))

if( showPrints ):
    display(score(true,pred,prob))
    display(score2(true,pred,prob))

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
