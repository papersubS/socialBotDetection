import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, KFold, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, KFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, classification_report,confusion_matrix
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.feature_selection import RFECV, chi2, SelectKBest, f_classif
from sklearn.metrics import precision_recall_fscore_support
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from sklearn.svm import LinearSVC
from sklearn.svm import SVC
import sys

path = ''
#posts = ['1', '10', '20', '50', '100', 'Full']
#
posts = ['Full']
norm_columns = ['NumberOfUppercaseCharactersAvg', 'NumberOfLowercaseCharactersAvg', 'NumberOfSpecialCharactersAvg',
                'NumberOfEmoticonAvg','NumberOfNumbersAvg','NumberOfBlanksAvg','NumberOfWordsAvg',
                'NumberOfPropositionsAvg','NumberOfPunctuationCharactersAvg','NumberOfLowercaseWordsAvg',
                'NumberOfUppercaseWordsAvg','NumberOfTotalCharactersStd','NumberOfUppercaseCharactersStd',
                'NumberOfLowercaseCharactersStd','NumberOfSpecialCharactersStd','NumberOfEmoticonStd',
                'NumberOfNumbersStd','NumberOfBlanksStd','NumberOfWordsStd','NumberOfPropositionsStd',
                'NumberOfPunctuationCharactersStd','NumberOfLowercaseWordsStd','NumberOfUppercaseWordsStd']


for n_post in posts:
    #####BOT#####
    df_bot_media = pd.read_csv('Reddit/Features/'+n_post+'/mediabot.csv',  sep=";", decimal=',')
    df_bot_std = pd.read_csv('Reddit/Features/'+n_post+'/Devbot.csv',  sep=";", decimal=',')
    df_bot_std = df_bot_std.drop(columns='Author')
    df_bot = pd.concat([df_bot_media, df_bot_std], axis=1)
    df_bot['Class']= 1
    #####################

    #####TROLL#####
    df_troll_media = pd.read_csv(
        'Reddit/Features/' + n_post + '/mediatroll.csv', sep=";",
        decimal=',')
    df_troll_std = pd.read_csv(
        'Reddit/Features/' + n_post + '/Devtroll.csv', sep=";",
        decimal=',')
    df_troll_std = df_troll_std.drop(columns='Author')
    df_troll = pd.concat([df_troll_media, df_troll_std], axis=1)
    df_troll['Class'] = 0
    #####################

    #####Fake followers#####
    df_fake_follower_media = pd.read_csv(path + 'mediafake.csv', sep=";", decimal=',')
    df_fake_follower_std = pd.read_csv(path + 'deviazionefake.csv', sep=";", decimal=',')
    df_fake_follower_std = df_fake_follower_std.drop(columns='Author')
    df_fake_follower = pd.concat([df_fake_follower_media, df_fake_follower_std], axis=1)
    df_fake_follower['Class'] = 1
    #####################

    ###########Spam BOT#########

    df_spambot1_media = pd.read_csv(path + 'mediasp1.csv', sep=";", decimal=',')
    df_spambot2_media = pd.read_csv(path + 'mediasp2.csv', sep=";", decimal=',')
    df_spambot3_media = pd.read_csv(path + 'mediasp3.csv', sep=";", decimal=',')
    df_spambot_media = df_spambot1_media
    df_spambot_media = df_spambot_media.append([df_spambot2_media, df_spambot3_media])

    df_spambot1_std = pd.read_csv(path + 'deviazionesp1.csv', sep=";", decimal=',')
    df_spambot2_std = pd.read_csv(path + 'deviazionesp2.csv', sep=";", decimal=',')
    df_spambot3_std = pd.read_csv(path + 'deviazionesp3.csv', sep=";", decimal=',')
    df_spambot_std = df_spambot1_std
    df_spambot_std = df_spambot_std.append([df_spambot2_std, df_spambot3_std])
    df_spambot_std = df_spambot_std.drop(columns='Author')
    df_spambot = pd.concat([df_spambot_media, df_spambot_std], axis=1)
    df_spambot['Class'] = 1

    ################################################

    #####Tradional Spambot#####
    df_trad_spampbots_media = pd.read_csv(path + 'mediaTS1.csv', sep=";", decimal=',')
    df_trad_spampbots_std = pd.read_csv(path + 'deviazioneTS1.csv', sep=";", decimal=',')
    df_trad_spampbots_std = df_trad_spampbots_std.drop(columns='Author')
    df_trad_spampbots = pd.concat([df_trad_spampbots_media, df_trad_spampbots_std], axis=1)
    df_trad_spampbots['Class'] = 1
    #####################

    ##############Genuine#####################
    df_genuine_media = pd.read_csv(path + 'mediagenuine.csv', sep=";", decimal=',')
    df_genuine_std = pd.read_csv(path + 'deviazionegenuine.csv', sep=";", decimal=',')
    df_genuine_std = df_genuine_std.drop(columns='Author')
    df_genuine = pd.concat([df_genuine_media, df_genuine_std], axis=1)
    df_genuine['Class'] = 0

    ################# Merge all BOT and TROLL#########
    df_test = df_bot
    df_test = df_test.append([df_troll])
    df_test = df_test.drop(columns='Author')
    #df.to_excel("full_dataset.xlsx")

    ################# Merge all BOT and genuine#########
    df_train = df_fake_follower
    df_train = df_train.append([df_spambot, df_genuine, df_trad_spampbots])
    df_train = df_train.drop(columns='Author')


    X_train = df_train.drop(columns='Class')
    #X_train[norm_columns] = X_train[norm_columns].div(X_train['NumberOfTotalCharactersAvg'], axis=0)
    # X_train = X_train[norm_columns].div(X_train['NumberOfTotalCharactersAvg'], axis=0)
    X_train = X_train.drop(columns='NumberOfTotalCharactersAvg')
    y_train = df_train['Class']

    X_test = df_test.drop(columns='Class')
    #X_test[norm_columns] = X_test[norm_columns].div(X_test['NumberOfTotalCharactersAvg'], axis=0)
    # X_test = X_test[norm_columns].div(X_test['NumberOfTotalCharactersAvg'], axis=0)
    X_test = X_test.drop(columns='NumberOfTotalCharactersAvg')
    y_test = df_test['Class']

    inner_cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

    results_path = 'results/'

    ############ TREE ################
    param_grid = {'max_depth': [1, 2, 3, 4]}
    model = DecisionTreeClassifier(random_state=42)

    clf = GridSearchCV(model, param_grid, cv=inner_cv, scoring='balanced_accuracy', verbose=0, n_jobs=-1)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(clf.best_score_)
    conf_mat = confusion_matrix(y_test, y_pred)

    tree_accuracy_average = accuracy_score(y_test, y_pred)

    print('Decision Tree cross platform:')
    print(str(tree_accuracy_average))
    print(conf_mat)
    with open(results_path + 'results.txt', 'a+') as f:
        print('Window ' + n_post, file=f)
        print('Decision Tree cross platform:', file=f)
        print(str(tree_accuracy_average), file=f)
        print(precision_recall_fscore_support(y_test, y_pred, average='micro'), file=f)
        print(conf_mat, file=f)

    ############ RANDOM FOREST ################
    param_grid = {'n_estimators': [20, 50, 100],
                  'max_depth': [1, 2, 3, 4]}
    model = RandomForestClassifier(random_state=42)

    clf = GridSearchCV(model, param_grid, cv=inner_cv, scoring='balanced_accuracy', verbose=0, n_jobs=-1)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    conf_mat = confusion_matrix(y_test, y_pred)

    rf_accuracy_average = accuracy_score(y_test, y_pred)

    print('Random Forest cross platform:')
    print(str(rf_accuracy_average))
    print(conf_mat)
    with open(results_path + 'results.txt', 'a+') as f:
        print('Window ' + n_post, file=f)
        print('Random Forest cross platform:', file=f)
        print(str(rf_accuracy_average), file=f)
        print(precision_recall_fscore_support(y_test, y_pred, average='micro'), file=f)
        print(conf_mat, file=f)

    ############ LOGISTIC REGRESSION ################
    sc = StandardScaler()
    X_std = sc.fit_transform(X_train)

    param_grid = {'C':[0.001, 0.01, 0.01, 1, 10], 'penalty':['l2']}
    model = LogisticRegression(max_iter=5000, random_state=42)

    clf = GridSearchCV(model, param_grid, cv=inner_cv, scoring='balanced_accuracy', verbose=0, n_jobs=-1)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    conf_mat = confusion_matrix(y_test, y_pred)

    lr_accuracy_average = accuracy_score(y_test, y_pred)

    print('Logistic Regression cross platform:')
    print(str(lr_accuracy_average))
    print(conf_mat)
    with open(results_path + 'results.txt', 'a+') as f:
        print('Window ' + n_post, file=f)
        print('Logistic Regression cross platform:', file=f)
        print(str(lr_accuracy_average) + ' +/- ', file=f)
        print(precision_recall_fscore_support(y_test, y_pred, average='micro'), file=f)
        print(conf_mat, file=f)

    ############ LINEAR SVC ################
    sc = StandardScaler()
    X_std = sc.fit_transform(X_train)
    Xt_std = sc.fit_transform(X_test)

    param_grid = {'C':[0.001, 0.01, 0.01, 1, 10]}
    model = LinearSVC( max_iter=5000)

    clf = GridSearchCV(model, param_grid, cv=inner_cv, scoring='balanced_accuracy', verbose=0, n_jobs=-1)
    clf.fit(X_std, y_train)
    y_pred = clf.predict(Xt_std)
    conf_mat = confusion_matrix(y_test, y_pred)

    svc_linear_accuracy_average = accuracy_score(y_test, y_pred)

    print('SVC linear cross platform:')
    print(str(svc_linear_accuracy_average))
    print(print(conf_mat))
    with open(results_path + 'results.txt', 'a+') as f:
        print('Window ' + n_post, file=f)
        print('SVC linear cross platform:', file=f)
        print(str(svc_linear_accuracy_average), file=f)
        print(precision_recall_fscore_support(y_test, y_pred, average='micro'), file=f)
        print(conf_mat, file=f)

    ############ SVM RBF ################

    sc = StandardScaler()
    X_std = sc.fit_transform(X_train)
    Xt_std = sc.fit_transform(X_test)

    param_grid = {'C':[0.001, 0.01, 0.01, 1, 10], 'gamma':[0.001, 0.01, 0.1, 1], 'kernel':['rbf']}
    model = SVC()

    clf = GridSearchCV(model, param_grid, cv=inner_cv, scoring='balanced_accuracy', verbose=0, n_jobs=-1)
    clf.fit(X_std, y_train)
    y_pred = clf.predict(Xt_std)
    conf_mat = confusion_matrix(y_test, y_pred)

    svc_rbf_accuracy_average = accuracy_score(y_test, y_pred)

    print('SVC rbf cross platform:')
    print(str(svc_rbf_accuracy_average))
    print(conf_mat)
    with open(results_path + 'results.txt', 'a+') as f:
        print('Window ' + n_post, file=f)
        print('SVC rbf cross platform:', file=f)
        print(str(svc_rbf_accuracy_average), file=f)
        print(precision_recall_fscore_support(y_test, y_pred, average='micro'), file=f)
        print(conf_mat, file=f)