import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, KFold, StratifiedKFold

path = ''
#####Fake followers#####
df_fake_follower_media = pd.read_csv(path+'mediafake.csv',  sep=";", decimal=',')
df_fake_follower_std = pd.read_csv(path+'deviazionefake.csv',  sep=";", decimal=',')
df_fake_follower_std = df_fake_follower_std.drop(columns='Author')
df_fake_follower = pd.concat([df_fake_follower_media, df_fake_follower_std], axis=1)
df_fake_follower['Class']= 1
#####################

###########Spam BOT#########

df_spambot1_media = pd.read_csv(path+'mediasp1.csv',  sep=";", decimal=',')
df_spambot2_media = pd.read_csv(path+'mediasp2.csv',  sep=";", decimal=',')
df_spambot3_media = pd.read_csv(path+'mediasp3.csv',  sep=";", decimal=',')
df_spambot_media = df_spambot1_media
df_spambot_media = df_spambot_media.append([df_spambot2_media, df_spambot3_media])


df_spambot1_std = pd.read_csv(path+'deviazionesp1.csv',  sep=";", decimal=',')
df_spambot2_std = pd.read_csv(path+'deviazionesp2.csv',  sep=";", decimal=',')
df_spambot3_std = pd.read_csv(path+'deviazionesp3.csv',  sep=";", decimal=',')
df_spambot_std = df_spambot1_std
df_spambot_std = df_spambot_std.append([df_spambot2_std, df_spambot3_std])
df_spambot_std = df_spambot_std.drop(columns='Author')
df_spambot = pd.concat([df_spambot_media, df_spambot_std], axis=1)
df_spambot['Class']= 1

################################################

#####Tradional Spambot#####
df_trad_spampbots_media = pd.read_csv(path+'mediaTS1.csv',  sep=";", decimal=',')
df_trad_spampbots_std = pd.read_csv(path+'deviazioneTS1.csv',  sep=";", decimal=',')
df_trad_spampbots_std = df_trad_spampbots_std.drop(columns='Author')
df_trad_spampbots = pd.concat([df_trad_spampbots_media, df_trad_spampbots_std], axis=1)
df_trad_spampbots['Class']= 1
#####################


##############Genuine#####################
df_genuine_media = pd.read_csv(path+'mediagenuine.csv',  sep=";", decimal=',')
df_genuine_std = pd.read_csv(path+'deviazionegenuine.csv',  sep=";", decimal=',')
df_genuine_std = df_genuine_std.drop(columns='Author')
df_genuine = pd.concat([df_genuine_media, df_genuine_std], axis=1)
df_genuine['Class']= 0

################# Merge all BOT and genuine#########
df = df_fake_follower
df = df.append([df_spambot, df_genuine, df_trad_spampbots])
df = df.drop(columns='Author')
print(df)

X_train = df.drop(columns='Class')
y_train = df['Class']

inner_cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
outer_cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)



from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, KFold
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_std = sc.fit_transform(X_train)

param_grid = {'C':[0.001, 0.01, 0.01, 1, 10], 'gamma':[0.001, 0.01, 0.1, 1], 'kernel':['rbf']}
model = SVC()

clf = GridSearchCV(model, param_grid, cv=inner_cv, scoring='accuracy', verbose=0, n_jobs=-1)
nested_score_accuracy = cross_val_score(clf, X=X_std, y=y_train, cv=outer_cv, scoring='accuracy')
nested_score_precision = cross_val_score(clf, X=X_std, y=y_train, cv=outer_cv, scoring='precision')
nested_score_recall = cross_val_score(clf, X=X_std, y=y_train, cv=outer_cv, scoring='recall')
nested_score_f1= cross_val_score(clf, X=X_std, y=y_train, cv=outer_cv, scoring='f1')

svc_rbf_accuracy_average = nested_score_accuracy.mean()
svc_rbf_accuracy_std = nested_score_accuracy.std()
svc_rbf_precision_average = nested_score_precision.mean()
svc_rbf_precision_std= nested_score_precision.std()
svc_rbf_recall_average= nested_score_recall.mean()
svc_rbf_recall_std = nested_score_recall.std()
svc_rbf_f1_average = nested_score_f1.mean()
svc_rbf_f1_std = nested_score_f1.std()
print('SVC rbf nested cv:')
print(str(svc_rbf_accuracy_average) +' +/- ' + str(svc_rbf_accuracy_std))
print(str(svc_rbf_precision_average) +' +/- ' + str(svc_rbf_precision_std))
print(str(svc_rbf_recall_average) +' +/- ' + str(svc_rbf_recall_std))
print(str(svc_rbf_f1_average) +' +/- ' + str(svc_rbf_f1_std))
print(nested_score_f1)