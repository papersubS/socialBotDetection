import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, KFold, StratifiedKFold
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, KFold
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import precision_recall_fscore_support

path = ''
results_path = 'LOBO/'
BOTS = ['Fake followers', 'Spam BOT', 'Traditional Spam BOT']
#####Fake followers#####
df_fake_follower_media = pd.read_csv(path+'mediafake.csv',  sep=";", decimal=',')
df_fake_follower_std = pd.read_csv(path+'deviazionefake.csv',  sep=";", decimal=',')
df_fake_follower_std = df_fake_follower_std.drop(columns='Author')
df_fake_follower = pd.concat([df_fake_follower_media, df_fake_follower_std], axis=1)
df_fake_follower['Class']= 1
df_fake_follower['Type'] = 'Fake followers'
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
df_spambot['Type'] = 'Spam BOT'

################################################

#####Tradional Spambot#####
df_trad_spampbots_media = pd.read_csv(path+'mediaTS1.csv',  sep=";", decimal=',')
df_trad_spampbots_std = pd.read_csv(path+'deviazioneTS1.csv',  sep=";", decimal=',')
df_trad_spampbots_std = df_trad_spampbots_std.drop(columns='Author')
df_trad_spampbots = pd.concat([df_trad_spampbots_media, df_trad_spampbots_std], axis=1)
df_trad_spampbots['Class']= 1
df_trad_spampbots['Type'] = 'Traditional Spam BOT'
#####################


##############Genuine#####################
df_genuine_media = pd.read_csv(path+'mediagenuine.csv',  sep=";", decimal=',')
df_genuine_std = pd.read_csv(path+'deviazionegenuine.csv',  sep=";", decimal=',')
df_genuine_std = df_genuine_std.drop(columns='Author')
df_genuine = pd.concat([df_genuine_media, df_genuine_std], axis=1)
df_genuine['Class']= 0
df_genuine['Type'] = 'Human'

################# Merge all BOT and genuine#########
df = df_fake_follower
df = df.append([df_spambot, df_trad_spampbots, df_genuine])
df = df.drop(columns='Author')
print(df)

for BOT in BOTS:
    X_test = df[df['Type'] == BOT]
    y_true = X_test['Class']
    X_test = X_test.drop(columns='Class')
    X_test = X_test.drop(columns='Type')

    X_train = df[df['Type'] != BOT]
    y_train = X_train['Class']
    X_train = X_train.drop(columns='Class')
    X_train = X_train.drop(columns='Type')

    # Add 10% fake followers in training
    # if BOT == 'Fake followers':
    #     X_transferdata_df = X_test.iloc[- 310:, 0:]
    #     y_transferdata_df = y_true.iloc[- 310:]
    #     X_train = X_train.append(X_transferdata_df)
    #     y_train = y_train.append(y_transferdata_df)
    #     X_test = X_test.drop(X_transferdata_df.index)
    #     y_true = y_true.drop(y_transferdata_df.index)

    inner_cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    #outer_cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

    sc = StandardScaler()
    X_train_std = sc.fit_transform(X_train)
    X_test_std = sc.transform(X_test)

    ############ TREE ################
    param_grid = {'max_depth': [1, 2, 3, 4]}
    model = DecisionTreeClassifier(random_state=42)

    clf = GridSearchCV(model, param_grid, cv=inner_cv, scoring='f1', verbose=0, n_jobs=-1)
    clf.fit(X_train_std, y_train)
    y_pred = clf.predict(X_test_std)
    print('Tree ' + BOT + ':')
    with open(results_path + 'results.txt', 'a+') as f:
        print('Tree ' + BOT + ':', file=f)
        print(classification_report(y_true, y_pred), file=f)

    ############ RANDOM FOREST ################
    param_grid = {'n_estimators': [20, 50, 100],
                  'max_depth': [1, 2, 3, 4]}
    model = RandomForestClassifier(random_state=42)

    clf = GridSearchCV(model, param_grid, cv=inner_cv, scoring='f1', verbose=0, n_jobs=-1)
    clf.fit(X_train_std, y_train)
    y_pred = clf.predict(X_test_std)
    print('Random Forest ' + BOT + ':')
    with open(results_path + 'results.txt', 'a+') as f:
        print('Random Forest ' + BOT + ':', file=f)
        print(classification_report(y_true, y_pred), file=f)

    ############ LOGISTIC REGRESSION ################
    param_grid = {'C': [0.001, 0.01, 0.01, 1, 10], 'penalty': ['l2']}
    model = LogisticRegression(max_iter=5000, random_state=42)

    clf = GridSearchCV(model, param_grid, cv=inner_cv, scoring='f1', verbose=0, n_jobs=-1)
    clf.fit(X_train_std, y_train)
    y_pred = clf.predict(X_test_std)
    print('Logistic ' + BOT + ':')
    with open(results_path + 'results.txt', 'a+') as f:
        print('Logistic ' + BOT + ':', file=f)
        print(classification_report(y_true, y_pred), file=f)

    ############ LINEAR SVC ################
    param_grid = {'C': [0.001, 0.01, 0.01, 1, 10]}
    model = LinearSVC(max_iter=5000)

    clf = GridSearchCV(model, param_grid, cv=inner_cv, scoring='f1', verbose=0, n_jobs=-1)
    clf.fit(X_train_std, y_train)
    y_pred = clf.predict(X_test_std)
    print('SVC linear' + BOT + ':')
    with open(results_path + 'results.txt', 'a+') as f:
        print('SVC linear' + BOT + ':', file=f)
        print(classification_report(y_true, y_pred), file=f)

    ############ RBF ################
    param_grid = {'C':[0.001, 0.01, 0.01, 1, 10], 'gamma':[0.001, 0.01, 0.1, 1], 'kernel':['rbf']}
    #param_grid = {'C':[1], 'gamma':[0.01], 'kernel':['rbf']}

    model = SVC()

    clf = GridSearchCV(model, param_grid, cv=inner_cv, scoring='f1', verbose=0, n_jobs=-1)
    clf.fit(X_train_std, y_train)
    y_pred = clf.predict(X_test_std)
    print('SVC rbf' + BOT + ':')
    with open(results_path + 'results.txt', 'a+') as f:
        print('SVC rbf' + BOT + ':', file=f)
        print(classification_report(y_true, y_pred), file=f)
