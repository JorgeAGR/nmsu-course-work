# -*- coding: utf-8 -*-
"""
Created on Wed Sep 25 21:53:34 2019

@author: jorge
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from tensorflow.losses import huber_loss
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv('train_images.csv', header=None)
df_labels = pd.read_csv('train_labels.csv')
df_test = pd.read_csv('test_images.csv', header=None)
df_test_labels = pd.read_csv('test_labels.csv')

volcano_ind = np.where(df_labels['Volcano?'].values == 1)[0]
not_ind = np.where(df_labels['Volcano?'].values == 0)[0]

train_pix = df.values / 255
train_img = train_pix.reshape(len(df), 110, 110, 1)[:,2:,2:]
test_pix = df_test.values / 255
test_img = test_pix.reshape(len(df_test), 110, 110, 1)[:,2:,2:]
train_y = df_labels.values[:,0]
test_y = df_test_labels.values[:,0]

compression_size = 128

input_img = Input(shape=(110*110,))

encoding = Dense(compression_size, activation='relu')(input_img)
encoded_input = Input(shape=(compression_size,))
decoding = Dense(110*110, activation='sigmoid')(encoding)

autoencoder = Model(input_img, decoding)
encoder = Model(input_img, encoding)

decoded_output = autoencoder.layers[-1]
decoder = Model(encoded_input, decoded_output(encoded_input))

autoencoder.compile('adam', loss=huber_loss)

autoencoder.fit(train_pix, train_pix, epochs=50,
                batch_size=128, shuffle=True,
                validation_data=(test_pix, test_pix))

'''
input_img = Input(shape=(108, 108, 1))  # adapt this if using `channels_first` image data format

x = Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x)

encoder = Dense(compression_size, activation='relu')(x)

# at this point the representation is (4, 4, 8) i.e. 128-dimensional

x = Conv2D(8, (3, 3), activation='relu', padding='same')(encoder)
x = UpSampling2D((2, 2))(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
x = Conv2D(16, (3, 3), activation='relu')(x)
x = UpSampling2D((2, 2))(x)
decoder = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

autoencoder = Model(input_img, decoder)

autoencoder.compile('adam', loss='binary_crossentropy')

autoencoder.fit(train_img, train_img, epochs=50,
                batch_size=128, shuffle=True,
                validation_data=(test_img, test_img))
'''
reconstructed = autoencoder.predict(train_pix)

train_encoded = encoder.predict(train_pix)
test_encoded = encoder.predict(test_pix)

fig, ax = plt.subplots()
ax.imshow(reconstructed[150].reshape(110, 110))
fig2, ax2 = plt.subplots()
ax2.imshow(train_pix[150].reshape(110, 110))

rf = RandomForestClassifier(bootstrap=True)
rf.fit(train_encoded, df_labels.values[:,0])
print('train score:', rf.score(train_encoded, train_y))
print('test score:', rf.score(test_encoded, test_y))

images = train_pix.reshape(len(df), 110, 110)
