# ### Imports

from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, confusion_matrix, f1_score, classification_report
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


def og_score(target, predictions=None, probas=None):
    
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


def og_score2(target, predictions=None, probas=None):
    
    if( type(probas) != type(None) ):
        roc_auc = roc_auc_score(target, probas)

    if( type(predictions) != type(None) ):
        acc = accuracy_score(target, predictions)
        prec = precision_score(target, predictions)
        rec = recall_score(target, predictions)
        f1 = f1_score(target, predictions)
        
    return ( roc_auc, acc, prec, rec, f1 )


def score(target, predictions=None, probas=None):
    
    if( type(probas) != type(None) ):
        print(f'AUC-ROC: {roc_auc_score(target, probas)}')

    if( type(predictions) != type(None) ):
        print(classification_report(target, predictions))
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


def score2(name, preproc, target, predictions=None, probas=None):
    
    df = pd.DataFrame(columns=["Modelo", "Preprocesamientos", "Clase", "AUC-ROC", "Accuracy", "Precision", "Recall", "F1 score", "Support"])
    
    auc_roc = None
    report = {}
    
    if( type(probas) != type(None) ):
        auc_roc = roc_auc_score(target, probas)

    if( type(predictions) != type(None) ):
        report = classification_report(target, predictions, output_dict=True)
        
    df = df.append(
        {
            "Modelo": name,
            "Preprocesamientos":  preproc,
            "Clase": "AVG",
            "AUC-ROC":  auc_roc,
            "Accuracy":  report['accuracy'],
            "Precision":  report['weighted avg']['precision'],
            "Recall":  report['weighted avg']['recall'],
            "F1 score":  report['weighted avg']['f1-score'],
            "Support": report['weighted avg']['support'],
        },
        ignore_index=True
    )
    clase = "True"
    df = df.append(
        {
            "Modelo": name,
            "Preprocesamientos":  preproc,
            "Clase": clase,
            "Precision":  report[clase]['precision'],
            "Recall":  report[clase]['recall'],
            "F1 score":  report[clase]['f1-score'],
            "Support": report[clase]['support'],
        },
        ignore_index=True
    )
    clase = "False"
    df = df.append(
        {
            "Modelo": name,
            "Preprocesamientos":  preproc,
            "Clase": clase,
            "Precision":  report[clase]['precision'],
            "Recall":  report[clase]['recall'],
            "F1 score":  report[clase]['f1-score'],
            "Support": report[clase]['support'],
        },
        ignore_index=True
    )
         
    return df


true = pd.Series([True,True,True,False,False,False])
pred = pd.Series([True,True,True,False,False,True])
prob = np.array([[0.9,0.8,0.7,0.3,0.2,0.6]])

if( showPrints ):
    display(score(true,pred))

if( showPrints ):
    display(score(true,pred,prob))
    display(score2(true,pred,prob))


def og_evalModels( df_feat=[], df_targ=[], modelos=[], predicciones=[] ):
    df = pd.DataFrame(columns=["Modelo", "Preprocesamientos", "AUC-ROC", "Accuracy", "Precision", "Recall", "F1 score"])
    
    for (name, model, preproc) in modelos:
        pred = model.predict(df_feat)
        prob = model.predict_proba(df_feat)

        roc_auc, acc, prec, rec, f1 = og_score2(df_targ, pred, prob[:,1])

        df = df.append(
            {
                "Modelo": name,
                "Preprocesamientos":  preproc,
                "AUC-ROC":  roc_auc,
                "Accuracy":  acc,
                "Precision":  prec,
                "Recall":  rec,
                "F1 score":  f1,
            },
            ignore_index=True
        )
        
    for (name, preproc, roc_auc, acc, prec, rec, f1) in predicciones:

        df = df.append(
            {
                "Modelo": name,
                "Preprocesamientos":  preproc,
                "AUC-ROC":  roc_auc,
                "Accuracy":  acc,
                "Precision":  prec,
                "Recall":  rec,
                "F1 score":  f1,
            },
            ignore_index=True
        )
    
    display(df)


def evalModels( df_feat=[], df_targ=[], modelos=[], predicciones=pd.DataFrame() ):
    
    df = pd.DataFrame(columns=["Modelo", "Preprocesamientos", "Clase", "AUC-ROC", "Accuracy", "Precision", "Recall", "F1 score", "Support"])
    
    for (name, model, preproc) in modelos:
        pred = model.predict(df_feat)
        prob = model.predict_proba(df_feat)

        newdf = score2(name, preproc,df_targ, pred, prob[:,1])

        df = df.append( newdf )
        
    df = df.append(predicciones)
    
    display(df)


# ### Get Feat/ Targ

df_feat = pd.read_csv("datasets/train_features.csv", low_memory=False).set_index('id')
df_targ = pd.read_csv("datasets/train_target.csv")


# ### Divide Train / Val

def train_val( feat, targ, targSeries = False ):
    tf, vf, tt, vt = train_test_split(
        feat,
        targ,
        test_size=0.1,
        stratify=targ.llovieron_hamburguesas_al_dia_siguiente,
        random_state=1
    )
    
    if(targSeries):
        tt = tt.llovieron_hamburguesas_al_dia_siguiente
        vt = vt.llovieron_hamburguesas_al_dia_siguiente
    
    return tf, vf, tt, vt


if(showPrints):
    train_feat, holdout_feat, train_target, holdout_target = train_val( df_feat, df_targ, True )
    display(train_feat)
    display(holdout_feat)
    display(train_target)
    display(holdout_target)
