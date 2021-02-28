import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from keras import backend as K
from keras import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
from keras.callbacks import EarlyStopping
from sklearn.model_selection import cross_val_score


def load_data(file):
    """This function loads the dataset and prints the first 5 rows 
       from the given file path
       Args:
           file (str): file path to be loaded

       Returns:
           Pandas DataFrame
    """
    return pd.read_csv(file, header=None)


def concatenate_data(df1, df2):
    """This function transposes both DataFrames and concatenates to combine the data needed
       Args:
           df1 (DataFrame): first dataframe
           df2 (DataFrame): second dataframe

       Returns:
           Concatenated DataFrame of both transposed DataFrames
    """
    return pd.concat([df1.T, df2.T], ignore_index=True)


def find_trials(data):
    """This function locates every index which indicates the start of a new trial
     Args:
          data (DataFrame): concatenated data

     Returns:
          List of all indices that indicate a new trial
    """
    # find every beginning point of a trial
    trial_idx = data.loc[data[73] == 1.0]
    # creates a list of all indices of the trials
    tidx_list = [x for x in trial_idx.index]
    return tidx_list


def find_markers(data):
    """This function locates every index which indicates the markers needed to create labels
       Args:
           data (DataFrame): concatenated data

       Returns:
          DataFrame of all indices that indicate a marker
    """
    # find every marker (label)
    markers_idx = data.loc[data[74] != 0.0]
    return markers_idx


def create_ic_labels(data):
    """This function creates a binary label column to append to DataFrame for classification
       (removing left/right attribute only congruent vs incongruent)
       Args:
          data (DataFrame): concatenated data

   Returns:
        Pandas Series (column) of labels for supervised classification
    """
    markers_idx = data.loc[data[74] != 0.0]
    labels = pd.Series(markers_idx[74], name='Labels').reset_index().drop(
        'index', axis=1)
    for i in labels.index:
        # if congruent label, assign 0
        if int(labels.iloc[i]) == 11 or int(labels.iloc[i]) == 22:
            labels.iloc[i] = 0
        else:
            labels.iloc[i] = 1
    return labels
# 11 = left congruent
# 22 = right congruent
# 31 = left incongruent (left denotes direction of center arrow)
# 42 = right incongruent


def create_binary_labels(data):
    """This function creates a binary label column to append to DataFrame for classification
       (removing congruent/incongruent attribute only left/right)
       Args:
          data (DataFrame): concatenated data

   Returns:
        Pandas Series (column) of labels for supervised classification
    """
    markers_idx = data.loc[data[74] != 0.0]
    labels = pd.Series(markers_idx[74], name='Labels').reset_index().drop(
        'index', axis=1)
    for i in labels.index:
        if int(labels.iloc[i]) == 11 or int(labels.iloc[i]) == 31:
            labels.iloc[i] = 0
        else:
            labels.iloc[i] = 1
    return labels


def create_multi_labels(data):
    """This function creates multiple label columns to append to DataFrame for
        multiclass classification
       Args:
          data (DataFrame): concatenated data

       Returns:
        Pandas Series (column) of labels for supervised classification
    """
    markers_idx = data.loc[data[74] != 0.0]
    labels = pd.Series(markers_idx[74], name='Labels').reset_index().drop(
        'index', axis=1)
    encoder = LabelEncoder()
    labels = encoder.fit_transform(labels.values.ravel())
    labels = pd.DataFrame(labels, columns=['Labels'])

    return labels


def separate_trials(data, trials_index):
    """This function separates the data into the different trials.
     Args:
          data (DataFrame): concatenated data
          trials_index (List): list of all indices that indicate a new trial

     Returns:
          List of each trial stored as DataFrames 
    """
    # trials list to store every trial
    trials = []

    for i in range(len(trials_index)):
        # try catch statement for end of list
        try:
            # Slices all data from first trial to second trial and removes last columns which is the first of the next trial
            trial = data.T.loc[:, trials_index[i]: trials_index[i+1]]
            trial.drop(
                trial.columns[len(trial.columns)-1], axis=1, inplace=True)
        except:
            trial = data.T.loc[:, trials_index[i]:]

        trials.append(trial)

    return trials


def process_trials(trials, window_1, window_2):
    """This function goes through each trial, resets the columns to show sample rate,
        gets data in the time window between (window_1 bound) - (window_2 bound)sample, and removes all channels from 64 on then scales 
        data per channel for each trial.
    Args:
        trials (List): list of all trials separated previously
        window_1 (Integer): first bound of window
        window_2 (Integer): second bound of window

    Returns:
        List of each processed trial stored as DataFrames 
    """
    # Go through each trial, reset the columns, we split from 100-300ms ((308th sample to 513th sample))
    scaler = MinMaxScaler()
    # Processed trials: trials which have been processed to split between 100-300ms
    pro_trials = []

    for trial in range(len(trials)):
        # Grabs each trial
        tr_df = trials[trial]
        # Resets the column numbers to allow easier slicing of samples
        tr_df.columns = range(tr_df.shape[1])
        # Slice each trial
        tr_df = tr_df.loc[:, window_1: window_2]
        # Remove all channels(rows) from 64 and up
        tr_df = tr_df.drop(tr_df.index[64:])
        # Turn trial frame around to scale across columns
        #tr_df = tr_df.T
        # Scale per column/channel
        for column in list(tr_df.columns):
            tr_df[column] = scaler.fit_transform(pd.DataFrame(tr_df[column]))
        # Flip trial frame back to output with channels on axis=0
        #tr_df = tr_df.T
        # Append new/processed trials in list
        pro_trials.append(tr_df)

    return pro_trials


def average_trials(pro_trials):
    """This function averages the data points across time/samples for every channel per trial.
    Args:
          pro_trials (List): list of all processed trials

    Returns:
          List of each averaged trial stored as DataFrames 
    """

    # Find the mean across channels (get average across sample rate for every channel in trial)
    avg_trials = []

    for split_trial in range(len(pro_trials)):
        avg_trial = pro_trials[split_trial].mean(axis=1)
        # Scale every average trial
        avg_trials.append(avg_trial)

    return avg_trials

# Concatenate the label column with the avg_trials_df


def create_ml_df(avg_trials, labels):
    """This function concatenates the average trials dataframe with labels to structure
      dataframe in format to allow machine learning classification.
    Args:
        avg_trials (List): list of all averaged trials
        labels (DataFrame): dataframe containing markers/labels

    Returns:
        DataFrame with machine learning structure 
    """

    # Once average is found, all avg_trails must become a final dataframe
    final_df = pd.DataFrame(avg_trials)

    # Concatenating the labels series as a column to the final_df
    ml_df = pd.concat([final_df, labels], axis=1)

    return ml_df

# Splits data into train and test, scales data depending on parameter


def prepare_ml_df(ml_df, scale=False):
    """This function preprocesses the machine learning dataframe by giving 
      an option of scaling the data before splitting into training and testing sets.
    Args:
        ml_df (DataFrame): DataFrame with machine learning structure
        scale (boolean): boolean to apply scaling

    Returns:
        DataFrame with machine learning structure 
    """
    # Separating the independent variables from the label

    if (scale == True):
        scaler = MinMaxScaler()
        X = ml_df.drop('Labels', axis=1)
        X = scaler.fit_transform(X)
    else:
        X = ml_df.drop('Labels', axis=1)

    # Dependent variable/labels
    y = ml_df['Labels']

    # Splitting the data to feed into the ML Classifier model
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42)

    return X_train, X_test, y_train, y_test


def train_svc(X_train, X_test, y_train, y_test):
    """This function trains an SVC classifier using grid search for hyperparameter tuning
     in order to return the accuracy and precision.

    Args:
        X_train : trained independent data
        X_test :  test independent data
        y_train : trained label/dependent data
        y_test : test label/dependent data

    Returns:
        Accuracy and precision rates for the SVC
    """
    # parameter grid
    param_grid = [{'C': [1, 10, 100, 1000], 'kernel': ['linear', 'rbf']}]

    # Initializing the SVC Classifier
    clf = SVC()

    # Initialize grid search for hyperparameter tuning
    gs_SVC = GridSearchCV(clf, param_grid, cv=5)
    gs_SVC.fit(X_train, y_train)

    # Predict using the fitted model
    y_pred = gs_SVC.predict(X_test)

    # Include cross-validation
    scores = cross_val_score(gs_SVC, X_train, y_train,
                             cv=5, scoring='accuracy')
    cv_acc = scores.mean()

    # return accuracy and precision
    accuracy = accuracy_score(y_pred, y_test)
    precision = precision_score(y_pred, y_test)

    return accuracy, precision, cv_acc


def train_svc_multi(X_train, X_test, y_train, y_test):
    """This function trains an SVC classifier using grid search for hyperparameter tuning
     in order to return the accuracy and precision.

    Args:
        X_train : trained independent data
        X_test :  test independent data
        y_train : trained label/dependent data
        y_test : test label/dependent data

    Returns:
        Accuracy and precision rates for the SVC
    """
    # parameter grid
    param_grid = [{'C': [1, 10, 100, 1000], 'kernel': ['linear', 'rbf']}]

    # Initializing the SVC Classifier
    clf = SVC()

    # Initialize grid search for hyperparameter tuning
    gs_SVC = GridSearchCV(clf, param_grid, cv=5)
    gs_SVC.fit(X_train, y_train)

    # Predict using the fitted model
    y_pred = gs_SVC.predict(X_test)

    # Include cross-validation
    scores = cross_val_score(gs_SVC, X_train, y_train,
                             cv=5, scoring='accuracy')
    cv_acc = scores.mean()

    # return accuracy and precision
    accuracy = accuracy_score(y_pred, y_test)
    precision = precision_score(y_pred, y_test, average='weighted')

    return accuracy, precision, cv_acc


def train_dtc(X_train, X_test, y_train, y_test):
    """This function trains a Decision Tree classifier using grid search for hyperparameter tuning
     in order to return the accuracy and precision.

    Args:
        X_train : trained independent data
        X_test :  test independent data
        y_train : trained label/dependent data
        y_test : test label/dependent data

    Returns:
        Accuracy and precision rates for the Decision Tree classifier
    """
    # parameter grid
    params = {'max_leaf_nodes': list(
        range(2, 100)), 'min_samples_split': [2, 3, 4]}

    # Initializing classifier
    dtc = DecisionTreeClassifier(random_state=42)

    # Initialize grid search for hyperparameter tuning
    gs_DTC = GridSearchCV(dtc, params, verbose=1, cv=5)
    gs_DTC.fit(X_train, y_train)

    # Predict using the fitted model
    y_pred = gs_DTC.predict(X_test)

    # Include cross-validation
    scores = cross_val_score(gs_DTC, X_train, y_train,
                             cv=5, scoring='accuracy')
    cv_acc = scores.mean()

    # return accuracy and precision
    accuracy = accuracy_score(y_pred, y_test)
    precision = precision_score(y_pred, y_test)

    return accuracy, precision, cv_acc


def train_dtc_multi(X_train, X_test, y_train, y_test):
    """This function trains a Decision Tree classifier using grid search for hyperparameter tuning
     in order to return the accuracy and precision.

    Args:
        X_train : trained independent data
        X_test :  test independent data
        y_train : trained label/dependent data
        y_test : test label/dependent data

    Returns:
        Accuracy and precision rates for the Decision Tree classifier
    """
    # parameter grid
    params = {'max_leaf_nodes': list(
        range(2, 100)), 'min_samples_split': [2, 3, 4]}

    # Initializing classifier
    dtc = DecisionTreeClassifier(random_state=42)

    # Initialize grid search for hyperparameter tuning
    gs_DTC = GridSearchCV(dtc, params, verbose=1, cv=5)
    gs_DTC.fit(X_train, y_train)

    # Predict using the fitted model
    y_pred = gs_DTC.predict(X_test)

    # Include cross-validation
    scores = cross_val_score(gs_DTC, X_train, y_train,
                             cv=5, scoring='accuracy')
    cv_acc = scores.mean()

    # return accuracy and precision
    accuracy = accuracy_score(y_pred, y_test)
    precision = precision_score(y_pred, y_test, average='weighted')

    return accuracy, precision, cv_acc


def train_nb(X_train, X_test, y_train, y_test):
    """This function trains a Naive Bayes classifier in order to return the accuracy and precision.

     Args:
         X_train : trained independent data
         X_test :  test independent data
         y_train : trained label/dependent data
         y_test : test label/dependent data

     Returns:
         Accuracy and precision rates for the Naive Bayes classifier
     """
    # Initialize classifier
    nb = GaussianNB()

    nb.fit(X_train, y_train)

    # Predict using the fitted model
    y_pred = nb.predict(X_test)

    # return accuracy and precision
    accuracy = accuracy_score(y_pred, y_test)
    precision = precision_score(y_pred, y_test)

    # Include cross-validation
    scores = cross_val_score(nb, X_train, y_train, cv=5, scoring='accuracy')
    cv_acc = scores.mean()

    return accuracy, precision, cv_acc


def train_nb_multi(X_train, X_test, y_train, y_test):
    """This function trains a Naive Bayes classifier in order to return the accuracy and precision.

    Args:
        X_train : trained independent data
        X_test :  test independent data
        y_train : trained label/dependent data
        y_test : test label/dependent data

    Returns:
        Accuracy and precision rates for the Naive Bayes classifier
    """
    # Initialize classifier
    nb = GaussianNB()

    nb.fit(X_train, y_train)

    # Predict using the fitted model
    y_pred = nb.predict(X_test)

    # return accuracy and precision
    accuracy = accuracy_score(y_pred, y_test)
    precision = precision_score(y_pred, y_test, average='weighted')

    scores = cross_val_score(nb, X_train, y_train, cv=5, scoring='accuracy')
    cv_acc = scores.mean()

    return accuracy, precision, cv_acc


def precision_m(y_true, y_pred):
    """This function finds the true precision of the keras classifier.
    Args:
        y_true : true numeric labels
        y_pred : predicted numeric labels 

    Returns:
        True precision rate of the keras classifier
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def train_nn(n_inputs, X_train, X_test, y_train, y_test):
    """This function uses a deep neural network for classification.

    Args:
        n_inputs : number of inputs
        X_train : trained independent data
        X_test :  test independent data
        y_train : trained label/dependent data
        y_test : test label/dependent data

    Returns:
        Accuracy and precision rates for the deep neural network classifier
    """
    classifier = Sequential()
    # First Hidden Layer
    classifier.add(Dense(4, activation='relu',
                         kernel_initializer='random_normal', input_dim=n_inputs))
    # Second Hidden Layer
    classifier.add(Dense(4, activation='relu',
                         kernel_initializer='random_normal'))
    # Output Layer
    classifier.add(Dense(1, activation='sigmoid',
                         kernel_initializer='random_normal'))

    # Compiling the neural network
    classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=[
                       'acc', precision_m])

    # Fitting the data to the training dataset
    classifier.fit(X_train, y_train, batch_size=2, epochs=700)

    _, accuracy, precision = classifier.evaluate(X_test, y_test, verbose=0)

    return accuracy, precision


def train_nn_multi(n_inputs, X_train, X_test, y_train, y_test):
    """This function uses a deep neural network for multiclass classification.

    Args:
        n_inputs : number of inputs
        X_train : trained independent data
        X_test :  test independent data
        y_train : trained label/dependent data
        y_test : test label/dependent data

    Returns:
        Accuracy and precision rates for the deep neural network classifier
    """

    model = Sequential()
    # Rectified Linear Unit Activation Function
    model.add(Dense(15, input_dim=n_inputs, activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(15, activation='relu'))
    model.add(Dropout(0.4))
    # Softmax for multi-class classification
    model.add(Dense(4, activation='softmax'))

    # Changes to labels to categorical for multi
    y_train = np_utils.to_categorical(y_train)
    y_test = np_utils.to_categorical(y_test)

    # Compile model here
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam', metrics=['acc', precision_m])

    callback = EarlyStopping(monitor='val_loss', patience=50)
    model.fit(X_train, y_train, validation_data=(X_test, y_test),
              batch_size=2, epochs=500, callbacks=[callback])

    _, accuracy, precision = model.evaluate(X_test, y_test, verbose=0)

    return accuracy, precision


def res_df(df, column, participant):
    '''
    Add precision/accuracy for every participant to the whole results
    Args:
        df: the dataframe of all results we have had
        column: the dataframe of result we want to add
        participant: participant number
    returns:
        all precision/accuracy results
    '''

    data = pd.DataFrame({f"Participant {participant}": column})
    df[f"Participant {participant}"] = data[f"Participant {participant}"].values
    return df
