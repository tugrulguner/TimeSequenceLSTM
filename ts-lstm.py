import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
import seaborn as sns


train = pd.read_csv('/kaggle/input/liverpool-ion-switching/train.csv')
labelt = train['open_channels'].values
traindata = train['signal'].values


def split_sequence(sequence, label, n_steps):
    X, y = list(), list()
    for i in range(len(sequence)):
        end_ix = i + n_steps
        if end_ix > len(sequence)-1:
            break
        seq_x, seq_y = sequence[i:end_ix], label[i]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)
    
def split_sequence_s(sequence, n_steps):
    X = list()
    for i in range(len(sequence)):
        end_ix = i + n_steps
        if end_ix > len(sequence)-1:
            break
        seq_x = sequence[i:end_ix]
        X.append(seq_x)
    return np.array(X)
    
n_step = 20
X, y = split_sequence(traindata[0:3000000], labelt[0:3000000], n_step)
X = X.reshape(len(X), n_step, 1)
#y = y/max(y)
(X.shape, y.shape)


model = tf.keras.Sequential([
    tf.keras.layers.LSTM(50, activation='relu', return_sequences=True),
    tf.keras.layers.LSTM(50, activation='relu'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(11),   
])

model.compile(optimizer='adam',
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])

model.fit(X, y, batch_size = 30, epochs=2)

test = pd.read_csv('/kaggle/input/liverpool-ion-switching/test.csv')
test.drop(['time'], axis = 1, inplace = True)
test = test.values
test = split_sequence_s(test, n_step)
test = test.reshape(len(test), n_step, 1)
test.shape

prediction = model.predict(test, verbose=0)

predictionpd = pd.DataFrame(predictionpd, columns=['prediction'])
predictionpd['prediction'].unique()
