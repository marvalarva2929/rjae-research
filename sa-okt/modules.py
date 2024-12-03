# modules needed
# 1: embedding function
# 2: multihead attention function
# 3: normalize function
# 4: read data from file function

import pandas as pd
import torch
from sklearn.datasets import train_test_split

max_len = 50
test_size = 0.2
seed = 0


def read_okt_data(path):
    dataset = pd.read_pickle(path)

    dataset['Score'] = dataset['Score_x']
    dataset = dataset.drop(columns=['Score_x', 'Score_y'])

    dataset['prompt-embedding'] = dataset['prompt-embedding'].apply(
            lambda x: torch.tensor(x))

    dataset.drop_duplicates(
            subset=['SubjectID', 'ProblemID'], keep='first').reset_index(
                    drop=True)

    prev_subject_id = 0
    subjectid_appedix = []
    timesteps = []
    for i in range(len(dataset)):
        if prev_subject_id != dataset.iloc[i].SubjectID:
            # when encountering a new student ID
            prev_subject_id = dataset.iloc[i].SubjectID
            accumulated = 0
            id_appendix = 1
        else:
            accumulated += 1
            if accumulated >= max_len:
                id_appendix += 1
                accumulated = 0
        timesteps.append(accumulated)
        subjectid_appedix.append(id_appendix)
    dataset['timestep'] = timesteps
    dataset['SubjectID_appendix'] = subjectid_appedix
    dataset['SubjectID'] = [dataset.iloc[i].SubjectID +
                            '_{}'.format(dataset.iloc[i].SubjectID_appendix)
                            for i in range(len(dataset))]

    students = dataset['SubjectID'].unique()
    train_students, test_students = train_test_split(
            students, test_size=test_size, random_state=seed)

    valid_students, train_students = train_test_split(
            train_students, test_size=0.5, random_state=seed)

    return train_students, valid_students, test_students, dataset


# return data in the form of sequences of
# (question + response + correct_api)
# each sequence is a specific session
def read_api_data(path):
    dataset = pd.read_pickle(path)
    return dataset


def make_dataloader(
        dataset_full, students,
        collate_fn, configs, n_workers=0, do_lstm_dataset=True, train=True):

    shuffle = True if train else False

    # make these sets into pytorch dataset format (list of dicts)
    lstm_dataset = make_pytorch_dataset(
            dataset_full, students, configs)

    data_loader = torch.utils.data.DataLoader(
        lstm_dataset, collate_fn=collate_fn, shuffle=shuffle,
        batch_size=configs.batch_size, num_workers=n_workers)

    return data_loader


def make_pytorch_dataset(
        dataset_full, students, configs, do_lstm_dataset=True):
    '''
    convert the pandas dataframe into dataset format that pytorch dataloader
    takes the resulting format is a list of dictionaries
    '''
    lstm_dataset = []
    for student in students:
        subset = dataset_full[dataset_full.SubjectID == student]
        lstm_dataset.append({
            'SubjectID': student,
            'ProblemID_seq': subset.ProblemID.tolist(),
            'Score': subset.Score.tolist(),
            'prompt-embedding': subset['prompt-embedding'].tolist(),
            'input': subset.input.tolist(),
        })
    del dataset_full
    return lstm_dataset

