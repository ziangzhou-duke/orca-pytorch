#!/usr/bin/python
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn import svm
from sklearn.metrics import roc_auc_score

# Task
task_name = 'ComParE2019_OrcaActivity'
classes   = ['noise','orca']

# Enter your team name HERE
team_name = 'baseline'

# Enter your submission number HERE
submission_index = 1

# Configuration
feature_set = 'ComParE'  # For all available options, see the dictionary feat_conf
complexities = [1e-5,1e-4,1e-3,1e-2,1e-1,1e0]  # SVM complexities (linear kernel)


# Mapping each available feature set to tuple (number of features, offset/index of first feature, separator, header option)
feat_conf = {'ComParE':      (6373, 1, ';', 'infer'),
             'BoAW-125':     ( 250, 1, ';',  None),
             'BoAW-250':     ( 500, 1, ';',  None),
             'BoAW-500':     (1000, 1, ';',  None),
             'BoAW-1000':    (2000, 1, ';',  None),
             'BoAW-2000':    (4000, 1, ';',  None),
             'auDeep-40':    (1024, 2, ',', 'infer'),
             'auDeep-50':    (1024, 2, ',', 'infer'),
             'auDeep-60':    (1024, 2, ',', 'infer'),
             'auDeep-70':    (1024, 2, ',', 'infer'),
             'auDeep-fused': (4096, 2, ',', 'infer')}

for feature_set in feat_conf:
    num_feat = feat_conf[feature_set][0]
    ind_off  = feat_conf[feature_set][1]
    sep      = feat_conf[feature_set][2]
    header   = feat_conf[feature_set][3]

    # Path of the features and labels
    features_path = '../features/'
    label_file    = '../lab/labels.csv'

    # Start
    print('\nRunning ' + task_name + ' ' + feature_set + ' baseline ... (this might take a while) \n')

    # Load features and labels
    X_train = pd.read_csv(features_path + task_name + '.' + feature_set + '.train.csv', sep=sep, header=header,
                          usecols=range(ind_off,num_feat+ind_off), dtype=np.float32).values
    X_devel = pd.read_csv(features_path + task_name + '.' + feature_set + '.devel.csv', sep=sep, header=header,
                          usecols=range(ind_off,num_feat+ind_off), dtype=np.float32).values
    X_test  = pd.read_csv(features_path + task_name + '.' + feature_set + '.test.csv',  sep=sep, header=header,
                          usecols=range(ind_off,num_feat+ind_off), dtype=np.float32).values

    df_labels = pd.read_csv(label_file)
    y_train = df_labels['label'][df_labels['file_name'].str.startswith('train')].values
    y_devel = df_labels['label'][df_labels['file_name'].str.startswith('devel')].values

    # Encode labels (necessary to obtain decision function output)
    encoder = LabelEncoder()
    encoder.fit( classes )
    y_train = encoder.transform(y_train)
    y_devel = encoder.transform(y_devel)

    # Concatenate training and development for final training
    X_traindevel = np.concatenate((X_train, X_devel))
    y_traindevel = np.concatenate((y_train, y_devel))

    # Feature normalisation
    scaler       = MinMaxScaler()
    X_train      = scaler.fit_transform(X_train)
    X_devel      = scaler.transform(X_devel)
    X_traindevel = scaler.fit_transform(X_traindevel)
    X_test       = scaler.transform(X_test)

    # Define sigmoid function to transform decision function outputs to 0->1
    def sigmoid(x):
        return 1. / (1. + np.exp(-np.clip(x, -100,100)))

    # Train SVM model with different complexities and evaluate
    auc_scores = []
    for comp in complexities:
        print('Complexity {0:.6f}'.format(comp))
        clf = svm.LinearSVC(C=comp, random_state=0)
        clf.fit(X_train, y_train)
        y_conf = sigmoid( clf.decision_function(X_devel) )
        auc_scores.append( roc_auc_score(y_devel, y_conf) )
        print('ROC-AUC on Devel {0:.3f}\n'.format(auc_scores[-1]))

    # Train SVM model on the whole training data with optimum complexity and get predictions on test data
    optimum_complexity = complexities[np.argmax(auc_scores)]
    print('\nOptimum complexity: {0:.6f}, maximum ROC-AUC on Devel {1:.3f}\n'.format(optimum_complexity, np.max(auc_scores)))

    clf = svm.LinearSVC(C=optimum_complexity, random_state=0)
    clf.fit(X_traindevel, y_traindevel)
    y_conf = sigmoid( clf.decision_function(X_test) )
    y_pred = clf.predict(X_test)
    y_pred = encoder.inverse_transform(y_pred)

    # Write out predictions to csv file (official submission format)
    pred_file_name = task_name + '.test.' + team_name + '_' + feature_set + '.csv'
    print('Writing file ' + pred_file_name + '\n')
    df = pd.DataFrame(data={'file_name': df_labels['file_name'][df_labels['file_name'].str.startswith('test')].values,
                            'prediction': y_pred.flatten(),
                            'confidence': y_conf.flatten()},
                      columns=['file_name','prediction','confidence'])
    df.to_csv('Offcial_output/' + pred_file_name, index=False)

print('Done.\n')
