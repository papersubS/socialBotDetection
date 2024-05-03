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
from sklearn.metrics import precision_recall_fscore_support
import sys

tweet = ['1', '10', '20', '50', '100', 'Full']
#tweet = ['20']

for n_tweet in tweet:
    #####Fake followers#####
    df_fake_follower_media = pd.read_csv('Multiclass/Features/'+n_tweet+'/mediafake.csv',  sep=";", decimal=',')
    df_fake_follower_std = pd.read_csv('Multiclass/Features/'+n_tweet+'/devfake.csv',  sep=";", decimal=',')
    df_fake_follower_std = df_fake_follower_std.drop(columns='Author')
    df_fake_follower = pd.concat([df_fake_follower_media, df_fake_follower_std], axis=1)
    df_fake_follower['Class']= 1
    #####################
    
    ###########Spam BOT#########
    
    df_spambot1_media = pd.read_csv('Multiclass/Features/'+n_tweet+'/mediasp1.csv',  sep=";", decimal=',')
    df_spambot2_media = pd.read_csv('Multiclass/Features/'+n_tweet+'/mediasp2.csv',  sep=";", decimal=',')
    df_spambot3_media = pd.read_csv('Multiclass/Features/'+n_tweet+'/mediasp3.csv',  sep=";", decimal=',')
    df_spambot_media = df_spambot1_media
    df_spambot_media = df_spambot_media.append([df_spambot2_media, df_spambot3_media])
    df_spambot_media
    
    df_spambot1_std = pd.read_csv('Multiclass/Features/'+n_tweet+'/devsp1.csv',  sep=";", decimal=',')
    df_spambot2_std = pd.read_csv('Multiclass/Features/'+n_tweet+'/devsp2.csv',  sep=";", decimal=',')
    df_spambot3_std = pd.read_csv('Multiclass/Features/'+n_tweet+'/devsp3.csv',  sep=";", decimal=',')
    df_spambot_std = df_spambot1_std
    df_spambot_std = df_spambot_std.append([df_spambot2_std, df_spambot3_std])
    df_spambot_std = df_spambot_std.drop(columns='Author')
    df_spambot = pd.concat([df_spambot_media, df_spambot_std], axis=1)
    df_spambot['Class']= 2
    df_spambot
    ################################################
    
    #####Tradional Spambot#####
    df_trad_spampbots_media = pd.read_csv('Multiclass/Features/'+n_tweet+'/mediaTS1.csv',  sep=";", decimal=',')
    df_trad_spampbots_std = pd.read_csv('Multiclass/Features/'+n_tweet+'/devTS1.csv',  sep=";", decimal=',')
    df_trad_spampbots_std = df_trad_spampbots_std.drop(columns='Author')
    df_trad_spampbots = pd.concat([df_trad_spampbots_media, df_trad_spampbots_std], axis=1)
    df_trad_spampbots['Class']= 3
    #####################
    
    ##############Genuine#####################
    df_genuine_media = pd.read_csv('Multiclass/Features/'+n_tweet+'/mediagenuine.csv',  sep=";", decimal=',')
    df_genuine_std = pd.read_csv('Multiclass/Features/'+n_tweet+'/devgenuine.csv',  sep=";", decimal=',')
    df_genuine_std = df_genuine_std.drop(columns='Author')
    df_genuine = pd.concat([df_genuine_media, df_genuine_std], axis=1)
    df_genuine['Class']= 0
    ########################################################
    
    
    ################# Merge all BOT and genuine#########
    df = df_fake_follower
    df = df.append([df_spambot, df_genuine, df_trad_spampbots])
    df = df.drop(columns='Author')
    #df.to_excel("full_dataset.xlsx")
    
    X_train = df.drop(columns='Class')
    y_train = df['Class']
    
    inner_cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    outer_cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    results_path = 'Multiclass/results/'
    ############ TREE ################
    param_grid = {'max_depth': [1, 2, 3, 4]}
    model = DecisionTreeClassifier(random_state=42)
    
    clf = GridSearchCV(model, param_grid, cv=inner_cv, scoring='balanced_accuracy', verbose=0, n_jobs=-1)
    nested_score_accuracy = cross_val_score(clf, X=X_train, y=y_train, cv=outer_cv, scoring='accuracy')
    nested_score_precision = cross_val_score(clf, X=X_train, y=y_train, cv=outer_cv, scoring='precision_micro')
    nested_score_recall = cross_val_score(clf, X=X_train, y=y_train, cv=outer_cv, scoring='recall_micro')
    nested_score_f1_micro = cross_val_score(clf, X=X_train, y=y_train, cv=outer_cv, scoring='f1_micro')
    nested_score_f1_macro = cross_val_score(clf, X=X_train, y=y_train, cv=outer_cv, scoring='f1_macro')

    y_pred = cross_val_predict(clf, X=X_train, y=y_train, cv=outer_cv)
    conf_mat = confusion_matrix(y_train, y_pred)
    
    tree_accuracy_average = nested_score_accuracy.mean()
    tree_accuracy_std = nested_score_accuracy.std()
    print('Decision Tree nested cv:')
    print(str(tree_accuracy_average) +' +/- ' + str(tree_accuracy_std))
    print(conf_mat)
    with open(results_path + 'results.txt', 'a+') as f:
        print('Window ' + n_tweet, file=f)
        print('Decision Tree nested cv:', file=f)
        print('Accuracy: ' + str(nested_score_accuracy.mean()) + ' +/- ' + str(nested_score_accuracy.std()), file=f)
        print('Precision: ' + str(nested_score_precision.mean()) + ' +/- ' + str(nested_score_precision.std()), file=f)
        print('Recall: ' + str(nested_score_recall.mean()) + ' +/- ' + str(nested_score_recall.std()), file=f)
        print('F1_micro: ' + str(nested_score_f1_micro.mean()) + ' +/- ' + str(nested_score_f1_micro.std()), file=f)
        print('F1_macro: ' + str(nested_score_f1_macro.mean()) + ' +/- ' + str(nested_score_f1_macro.std()), file=f)
        print(conf_mat, file=f)
    
    ############ RANDOM FOREST ################
    param_grid = {'n_estimators': [20, 50, 100],
                  'max_depth': [1, 2, 3, 4]}
    model = RandomForestClassifier(random_state=42)
    
    clf = GridSearchCV(model, param_grid, cv=inner_cv, scoring='balanced_accuracy', verbose=0, n_jobs=-1)
    nested_score_accuracy = cross_val_score(clf, X=X_train, y=y_train, cv=outer_cv, scoring='accuracy')
    nested_score_precision = cross_val_score(clf, X=X_train, y=y_train, cv=outer_cv, scoring='precision_micro')
    nested_score_recall = cross_val_score(clf, X=X_train, y=y_train, cv=outer_cv, scoring='recall_micro')
    nested_score_f1_micro = cross_val_score(clf, X=X_train, y=y_train, cv=outer_cv, scoring='f1_micro')
    nested_score_f1_macro = cross_val_score(clf, X=X_train, y=y_train, cv=outer_cv, scoring='f1_macro')
    y_pred = cross_val_predict(clf, X=X_train, y=y_train, cv=outer_cv)
    conf_mat = confusion_matrix(y_train, y_pred)
    
    rf_accuracy_average = nested_score_accuracy.mean()
    rf_accuracy_std = nested_score_accuracy.std()
    print('Random Forest nested cv:')
    print(str(rf_accuracy_average) +' +/- ' + str(rf_accuracy_std))
    print(conf_mat)
    with open(results_path + 'results.txt', 'a+') as f:
        print('Window ' + n_tweet, file=f)
        print('Random Forest nested cv:', file=f)
        print('Accuracy: ' + str(nested_score_accuracy.mean()) + ' +/- ' + str(nested_score_accuracy.std()), file=f)
        print('Precision: ' + str(nested_score_precision.mean()) + ' +/- ' + str(nested_score_precision.std()), file=f)
        print('Recall: ' + str(nested_score_recall.mean()) + ' +/- ' + str(nested_score_recall.std()), file=f)
        print('F1_micro: ' + str(nested_score_f1_micro.mean()) + ' +/- ' + str(nested_score_f1_micro.std()), file=f)
        print('F1_macro: ' + str(nested_score_f1_macro.mean()) + ' +/- ' + str(nested_score_f1_macro.std()), file=f)
        print(conf_mat, file=f)
    
    ############ LOGISTIC REGRESSION ################
    sc = StandardScaler()
    X_std = sc.fit_transform(X_train)
    
    param_grid = {'C':[0.001, 0.01, 0.01, 1, 10], 'penalty':['l2']}
    model = LogisticRegression(max_iter=5000, random_state=42)
    
    
    clf = GridSearchCV(model, param_grid, cv=inner_cv, scoring='balanced_accuracy', verbose=0, n_jobs=-1)
    nested_score_accuracy = cross_val_score(clf, X=X_train, y=y_train, cv=outer_cv, scoring='accuracy')
    nested_score_precision = cross_val_score(clf, X=X_train, y=y_train, cv=outer_cv, scoring='precision_micro')
    nested_score_recall = cross_val_score(clf, X=X_train, y=y_train, cv=outer_cv, scoring='recall_micro')
    nested_score_f1_micro = cross_val_score(clf, X=X_train, y=y_train, cv=outer_cv, scoring='f1_micro')
    nested_score_f1_macro = cross_val_score(clf, X=X_train, y=y_train, cv=outer_cv, scoring='f1_macro')
    y_pred = cross_val_predict(clf, X=X_train, y=y_train, cv=outer_cv)
    conf_mat = confusion_matrix(y_train, y_pred)
    
    lr_accuracy_average = nested_score_accuracy.mean()
    lr_accuracy_std = nested_score_accuracy.std()
    print('Logistic Regression nested cv:')
    print(str(lr_accuracy_average) +' +/- ' + str(lr_accuracy_std))
    print(conf_mat)
    with open(results_path + 'results.txt', 'a+') as f:
        print('Window ' + n_tweet, file=f)
        print('Logistic Regression nested cv:', file=f)
        print('Accuracy: ' + str(nested_score_accuracy.mean()) + ' +/- ' + str(nested_score_accuracy.std()), file=f)
        print('Precision: ' + str(nested_score_precision.mean()) + ' +/- ' + str(nested_score_precision.std()), file=f)
        print('Recall: ' + str(nested_score_recall.mean()) + ' +/- ' + str(nested_score_recall.std()), file=f)
        print('F1_micro: ' + str(nested_score_f1_micro.mean()) + ' +/- ' + str(nested_score_f1_micro.std()), file=f)
        print('F1_macro: ' + str(nested_score_f1_macro.mean()) + ' +/- ' + str(nested_score_f1_macro.std()), file=f)
        print(conf_mat, file=f)
    
    ############ LINEAR SVC ################
    sc = StandardScaler()
    X_std = sc.fit_transform(X_train)
    
    param_grid = {'C':[0.001, 0.01, 0.01, 1, 10]}
    model = LinearSVC( max_iter=5000)
    
    clf = GridSearchCV(model, param_grid, cv=inner_cv, scoring='balanced_accuracy', verbose=0, n_jobs=-1)
    nested_score_accuracy = cross_val_score(clf, X=X_std, y=y_train, cv=outer_cv, scoring='accuracy')
    nested_score_precision = cross_val_score(clf, X=X_std, y=y_train, cv=outer_cv, scoring='precision_micro')
    nested_score_recall = cross_val_score(clf, X=X_std, y=y_train, cv=outer_cv, scoring='recall_micro')
    nested_score_f1_micro = cross_val_score(clf, X=X_std, y=y_train, cv=outer_cv, scoring='f1_micro')
    nested_score_f1_macro = cross_val_score(clf, X=X_std, y=y_train, cv=outer_cv, scoring='f1_macro')
    y_pred = cross_val_predict(clf, X=X_std, y=y_train, cv=outer_cv)
    conf_mat = confusion_matrix(y_train, y_pred)
    
    svc_linear_accuracy_average = nested_score_accuracy.mean()
    svc_linear_accuracy_std = nested_score_accuracy.std()
    print('SVC linear nested cv:')
    print(str(svc_linear_accuracy_average) +' +/- ' + str(svc_linear_accuracy_std))
    print(print(conf_mat))
    with open(results_path + 'results.txt', 'a+') as f:
        print('Window ' + n_tweet, file=f)
        print('SVC linear nested cv:', file=f)
        print('Accuracy: ' + str(nested_score_accuracy.mean()) + ' +/- ' + str(nested_score_accuracy.std()), file=f)
        print('Precision: ' + str(nested_score_precision.mean()) + ' +/- ' + str(nested_score_precision.std()), file=f)
        print('Recall: ' + str(nested_score_recall.mean()) + ' +/- ' + str(nested_score_recall.std()), file=f)
        print('F1_micro: ' + str(nested_score_f1_micro.mean()) + ' +/- ' + str(nested_score_f1_micro.std()), file=f)
        print('F1_macro: ' + str(nested_score_f1_macro.mean()) + ' +/- ' + str(nested_score_f1_macro.std()), file=f)
        print(conf_mat, file=f)
    
    ############ SVM RBF ################
    
    sc = StandardScaler()
    X_std = sc.fit_transform(X_train)
    
    param_grid = {'C':[0.001, 0.01, 0.01, 1, 10], 'gamma':[0.001, 0.01, 0.1, 1], 'kernel':['rbf']}
    model = SVC()
    
    clf = GridSearchCV(model, param_grid, cv=inner_cv, scoring='balanced_accuracy', verbose=0, n_jobs=-1)
    nested_score_accuracy = cross_val_score(clf, X=X_std, y=y_train, cv=outer_cv, scoring='accuracy')
    nested_score_precision = cross_val_score(clf, X=X_std, y=y_train, cv=outer_cv, scoring='precision_micro')
    nested_score_recall = cross_val_score(clf, X=X_std, y=y_train, cv=outer_cv, scoring='recall_micro')
    nested_score_f1_micro = cross_val_score(clf, X=X_std, y=y_train, cv=outer_cv, scoring='f1_micro')
    nested_score_f1_macro = cross_val_score(clf, X=X_std, y=y_train, cv=outer_cv, scoring='f1_macro')
    y_pred = cross_val_predict(clf, X=X_std, y=y_train, cv=outer_cv)
    conf_mat = confusion_matrix(y_train, y_pred)
    
    svc_rbf_accuracy_average = nested_score_accuracy.mean()
    svc_rbf_accuracy_std = nested_score_accuracy.std()
    
    print('SVC rbf nested cv:')
    print(str(svc_rbf_accuracy_average) +' +/- ' + str(svc_rbf_accuracy_std))
    print(conf_mat)
    with open(results_path + 'results.txt', 'a+') as f:
        print('Window ' + n_tweet, file=f)
        print('SVC rbf nested cv:', file=f)
        print('Accuracy: ' + str(nested_score_accuracy.mean()) + ' +/- ' + str(nested_score_accuracy.std()), file=f)
        print('Precision: ' + str(nested_score_precision.mean()) + ' +/- ' + str(nested_score_precision.std()), file=f)
        print('Recall: ' + str(nested_score_recall.mean()) + ' +/- ' + str(nested_score_recall.std()), file=f)
        print('F1_micro: ' + str(nested_score_f1_micro.mean()) + ' +/- ' + str(nested_score_f1_micro.std()), file=f)
        print('F1_macro: ' + str(nested_score_f1_macro.mean()) + ' +/- ' + str(nested_score_f1_macro.std()), file=f)
        print(conf_mat, file=f)
