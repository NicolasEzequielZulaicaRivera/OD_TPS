# ### Imports

from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


# ### Score

def score(target, predictions):
    print(f'ACCURACY: {accuracy_score(target, predictions)}')

