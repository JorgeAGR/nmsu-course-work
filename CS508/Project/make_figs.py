# -*- coding: utf-8 -*-
"""
Created on Wed Sep 25 21:53:34 2019

@author: jorge
"""

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from itertools import product

golden_ratio = (np.sqrt(5) + 1) / 2
width = 10
height = 10

mpl.rcParams['figure.figsize'] = (width, height)
mpl.rcParams['font.size'] = 20
mpl.rcParams['figure.titlesize'] = 'small'
mpl.rcParams['legend.fontsize'] = 'small'
mpl.rcParams['xtick.major.size'] = 12
mpl.rcParams['xtick.minor.size'] = 8
mpl.rcParams['xtick.labelsize'] = 18
mpl.rcParams['ytick.major.size'] = 12
mpl.rcParams['ytick.minor.size'] = 8
mpl.rcParams['ytick.labelsize'] = 18

new_shape = 80

def translate(row):
    '''
    translate the nxn image by some amount so that if a 
    volcano is present it will be off-center
    '''

    new = row.reshape((110,110))
    left = np.random.randint(0, 110 + 1 - new_shape)

    up = np.random.randint(0, 110 + 1 - new_shape)

    new = new[left:left + new_shape,:]
    new = new[:,up:up+new_shape]

    done = new.reshape((1,new_shape**2))
    return done

def rotate(row, rot_num, shape=(110,110)):
    '''
    rotate the nxn image so that the algorithms can learn
    to detect volcanos from different viewing angles
    
    rot_num : number of 90 deg rotations to perform
    '''
    # reshape row to nxn image
    new = row.reshape(shape)
    # rotate some number of times
    rotated = np.rot90(new, k=rot_num)
    # reshape back to 1xn
    done = rotated.reshape((1,shape[0]*shape[1]))
    return done

def trim(row):
    '''
    trims off 5 pixels from each side of the image
    '''
    new = row.reshape((110,110))
    for i in range((110-new_shape)//2):
        new = np.delete(new,0,1)
    for j in range((110-new_shape)//2):
        new = np.delete(new,-1,1)
    for k in range((110-new_shape)//2):
        new = np.delete(new,0,0)
    for l in range((110-new_shape)//2):
        new = np.delete(new,-1,0)
    done = new.reshape((1,new_shape**2))
    return done

def augment_data(x_data, y_data, n_trans=4, n_rots=4):
  aug_x, aug_y = [], []
  for i in range(x_data.shape[0]):
    for angle in np.random.choice(4, n_rots, replace=False):
      rotation = rotate(x_data[i], angle)
      for j in range(n_trans-1):
        translation = translate(rotation)
        aug_x.append(translation)
        aug_y.append(y_data)
      centered = trim(rotation)
      aug_x.append(centered)
      aug_y.append(y_data)
  aug_x = np.asarray(aug_x).reshape(x_data.shape[0]*n_trans*n_rots, new_shape**2)
  aug_y = np.asarray(aug_y)
  return aug_x, aug_y

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

fig, ax = plt.subplots(ncols = 2)
ax[0].axis('off')
ax[1].axis('off')
ax[0].imshow(train_pix[9].reshape(110, 110))
ax[1].imshow(train_pix[1].reshape(110, 110))
ax[0].set_title('Volcano')
ax[1].set_title('No Volcano')
fig.subplots_adjust(right=0.97, left=0.03, bottom=0.03, top=0.97,
                    wspace=0.1, hspace=0.1)
fig.savefig('figs/eg_img.png', dpi=250)

img_chosen = train_pix[9].reshape(1,train_pix[9].shape[0])
img_aug, y_aug = augment_data(img_chosen, train_y[9])

fig = plt.figure()
ax = [plt.subplot2grid((4,4), loc, colspan=1, rowspan = 1, fig=fig) for loc in product(np.arange(4),np.arange(4))]
for a in ax:
    a.axis('off')
for i in range(16):
    img_dat = img_aug[i]
    ax[i].imshow(img_dat.reshape(new_shape,new_shape))
fig.subplots_adjust(right=0.97, left=0.03, bottom=0.03, top=0.97,
                    wspace=0.1, hspace=0.1)
fig.savefig('figs/eg_aug.png', dpi=250)