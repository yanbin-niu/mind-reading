import os
from mind_reading_package import mind_reading as mr
import pandas as pd

# list all folders' name
participants = os.listdir('path')

# remove the 'cha' folder we don't need
participants = participants.remove('cha')

# create the initial dataframe
df = pd.DataFrame(index=['SVC', 'DTC', 'NB', 'NN'])

for participant in participants:
    # iterate all the folders

    for file in os.listdir(participant):
        # iterate all files in every folder, find out the one end with 'Cong.csv' and 'Incong.csv' as input data

        if file.endswith('Cong.csv'):
            file1 = f"{participant}/{file}"
        if file.endswith('Incong.csv'):
            file2 = f"{participant}/{file}"

    # load in cong and incong data for them
    df1 = mr.load_data(file1)
    df2 = mr.load_data(file2)

    # concatenate such data
    data = mr.concatenate_data(df1, df2)

    # find trials to later separate
    trials_index = mr.find_trials(data)

    # separate trials
    trials = mr.separate_trials(data, trials_index)

    # create the label column
    labels = mr.create_ic_labels(data)

    # Go through each trial, reset the columns, we split from 100-300ms ((308th sample to 513th sample))
    pro_trials = mr.process_trials(trials, 250, 550)

    # Find the mean across channels
    avg_trials = mr.average_trials(pro_trials)

    # concatenates the average trials dataframe with labels
    ml_df = mr.create_ml_df(avg_trials, labels)

    # train models
    X_train, X_test, y_train, y_test = mr.prepare_ml_df(ml_df)

    acc_svc, precision_svc = mr.train_svc(X_train, X_test, y_train, y_test)

    acc_dtc, precision_dtc = mr.train_dtc(X_train, X_test, y_train, y_test)

    acc_nb, precision_nb = mr.train_nb(X_train, X_test, y_train, y_test)

    acc_nn, precision_nn = mr.train_nn(64, X_train, X_test, y_train, y_test)

    # add every participant's accuracy together
    acc_list = [f"{acc_svc:.2f}", f"{acc_dtc:.2f}",
                f"{acc_nb:.2f}", f"{acc_nn:.2f}"]

    df = mr.res_df(df, acc_list, participant)

# generate result .csv file
df.to_csv('cong_incong_accuracy.csv')
