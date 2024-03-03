### THIS FILE CONTAINS COMMON FUNCTIONS, CLASSSES

import tqdm
import time
import random 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
from torch import nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from scipy.io import wavfile as wav

from sklearn import preprocessing
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix


loss_fn = nn.BCELoss()

class MLP(nn.Module):

    def __init__(self, no_features, no_hidden, no_labels):
        super().__init__()
        self.mlp_stack = nn.Sequential(
            # Define the layers of the network

            # Define the Input Layer --> 1st Hidden Layer
            nn.Linear(no_features, no_hidden),
            # Input layer to 1st hidden layer with 'no_features number' of input features and 'no_hidden number' of neurons
            nn.ReLU(),  # ReLU activation function
            nn.Dropout(p=0.2),  # Applying dropout with probability 0.2

            # Define the 1st Hidden Layer --> 2nd Hidden Layer
            nn.Linear(no_hidden, no_hidden),
            # 1st hidden layer to 2nd hidden layer with 'no_features number' of input features and 'no_hidden number' of neurons
            nn.ReLU(),  # ReLU activation function
            nn.Dropout(p=0.2),  # Applying dropout with probability 0.2

            # Define the 2nd Hidden Layer --> 3rd Hidden Layer
            nn.Linear(no_hidden, no_hidden),
            # 2nd hidden layer to 3rd hidden layer with 'no_features number' of input features and 'no_hidden number' of neurons
            nn.ReLU(),  # ReLU activation function
            nn.Dropout(p=0.2),  # Applying dropout with probability 0.2

            # Define the 3rd Hidden Layer --> Output Layer
            nn.Linear(no_hidden, no_labels),
            # 3rd hidden layer to output layer with 'no_hidden number' of neurons and 'no_labels number' of output neurons
            nn.Sigmoid()  # Sigmoid activation function for binary classification
        )

    def forward(self, x):
        # Pass the input tensor through the layers
        return self.mlp_stack(x)

class CustomDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X,
                              dtype=torch.float)  # Convert the input features to PyTorch tensors with the appropriate float datatype
        self.y = torch.tensor(y, dtype=torch.float).unsqueeze(
            1)  # Convert the outputs to PyTorch tensors with the appropriate float datatype convert from 1D to 2D tensor

    def __len__(self):
        return len(self.y)  # Return the number of samples in the dataset

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]  # Return the input features and the outputs for the given index


def intialise_loaders(X_train_scaled, y_train, X_test_scaled, y_test, batch_size):
    # Create a custom dataset for the training and test sets
    train_dataset = CustomDataset(X_train_scaled, y_train)
    test_dataset = CustomDataset(X_test_scaled, y_test)

    # Create a DataLoader for the training and test sets
    # Shuffle the datasets to ensure that the model does not overfit to the order of the samples
    # Set the batch size to the specified batch size
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    return train_dataloader, test_dataloader

def split_dataset(df, columns_to_drop, test_size, random_state):
    label_encoder = preprocessing.LabelEncoder()

    df['label'] = label_encoder.fit_transform(df['label'])

    df_train, df_test = train_test_split(df, test_size=test_size, random_state=random_state)

    df_train2 = df_train.drop(columns_to_drop,axis=1)
    y_train2 = df_train['label'].to_numpy()

    df_test2 = df_test.drop(columns_to_drop,axis=1)
    y_test2 = df_test['label'].to_numpy() 

    return df_train2, y_train2, df_test2, y_test2

def preprocess_dataset(df_train, df_test):

    standard_scaler = preprocessing.StandardScaler()
    df_train_scaled = standard_scaler.fit_transform(df_train)

    df_test_scaled = standard_scaler.transform(df_test)

    return df_train_scaled, df_test_scaled

def set_seed(seed = 0):
    '''
    set random seed
    '''
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


# early stopping obtained from tutorial
class EarlyStopper:
    def __init__(self, patience=3, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False
