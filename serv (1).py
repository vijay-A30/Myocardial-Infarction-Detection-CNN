import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import itertools
#%matplotlib inline
np.random.seed(0) # so our results will be copyable

from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, recall_score, precision_score, confusion_matrix, f1_score, classification_report
from sklearn.metrics import classification_report

import seaborn as sns
import tensorflow
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, Conv1D, MaxPool1D, Input
from keras.optimizers import Adam, RMSprop
from keras.losses import SparseCategoricalCrossentropy

import mlxtend 

import sys
import socket
import pickle

import hashlib
import zlib

def model_creation(Shape):
    model = Sequential([
    Input(shape=(Shape, 1)),
    Conv1D(128, 11, activation='relu', padding='Same'),
    MaxPool1D(pool_size=3, strides=2, padding='same'),
    Conv1D(64, 3, activation='relu', padding='Same'),
    MaxPool1D(pool_size=3, strides=2, padding='same'),
    Conv1D(64, 3, activation='relu', padding='Same'),
    MaxPool1D(pool_size=3, strides=2, padding='same'),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(10, activation='linear')])
    return model


# Initialize server socket
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind(('localhost', 8888))
server_socket.listen(5)

print("Server is listening...\n")

# Initialize model

while True:
    # Accept client connection
    print("Listening state..\n")
    client_socket, client_address = server_socket.accept()

    print(f"Connection from {client_address} has been established!\n")

    
   

    # Load updated model parameters
    compressed_data = b''
    while True:
        chunk = client_socket.recv(1024)
        if not chunk:
            break
        compressed_data += chunk
    
    # Decompress the data
    decompressed_data = zlib.decompress(compressed_data)
    
    # Unpickle the data and extract checksum
    checksum, pickled_data = pickle.loads(decompressed_data)
    
    # Verify data integrity
    if hashlib.sha256(pickled_data).digest() != checksum:
        raise ValueError("Data integrity check failed")
    
    print("Received model parameters from the client.\n")
    
    data=pickle.loads(pickled_data)
    #print()
    print(type(data))
    print(data.keys())

server_socket.close()

