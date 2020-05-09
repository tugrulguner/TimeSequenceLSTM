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


def seqcreator(seq, label, seqlength):
    inpx, inpy = list(), list()
    for i in range(len(seq)):
        steprange = i + seqlength
        if steprange > len(seq)-1:
            break
        inp_seq_x, inp_seq_y = seq[i:steprange], label[i]
        inpx.append(inp_seq_x)
        inpy.append(inp_seq_y)
    return np.array(inpx), np.array(inpy)
    
def seqcreatorsingle(seq, seqlength):
    inpx = list()
    for i in range(len(seq)):
        steprange = i + seqlength
        if steprange > len(seq)-1:
            break
        inp_seq_x = seq[i:steprange]
        inpx.append(inp_seq_x)
    return np.array(inpx)
    
seqlength = 20
X, y = seqcreator(traindata[0:3000000], labelt[0:3000000], seqlength)
X = X.reshape(len(X), seqlength, 1)
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
test = seqcreatorsingle(test, seqlength)
test = test.reshape(len(test), seqlength, 1)
test.shape

prediction = model.predict(test, verbose=0)

predictionpd = pd.DataFrame(predictionpd, columns=['prediction'])
predictionpd['prediction'].unique()
