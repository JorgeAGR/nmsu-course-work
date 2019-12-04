#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  9 15:28:01 2018

@author: jorgeagr
"""

import pandas as pd
import numpy as np

state_turnout = pd.read_csv('state_turnout.csv', index_col = 'STATE')

state = 'nm'

def ageGroup(agep):
    if 34 >= agep >= 18:
        return 1
    elif 64 >= agep >= 35:
        return 2
    elif agep >= 65:
        return 3
    else:
        return 0

def race(fhisp, rac1p):
    if fhisp == 1:
        return 5
    else:
        switch = {1: 1,
                  2: 2,
                  3: 4,
                  4: 4,
                  5: 4,
                  6: 3,
                  7: 3,
                  8: 6,
                  9: 6,}
        return switch.get(rac1p, 0)

def region(st):
    if st in (9, 23, 25, 33, 34, 36, 42, 44, 50): # North East
        return 1
    if st in (17, 18, 19, 20, 26, 27, 29, 31, 38, 39, 46, 55): # North Center
        return 2
    if st in (1, 5, 10, 11, 12, 13, 21, 22, 24, 28, 37, 40, 45, 47, 48, 51, 54): # South
        return 3
    if st in (2, 4, 6, 8, 15, 16, 30, 32, 35, 41, 49, 53, 56): # West
        return 4
    else:
        return 0

def percentile(pincp):
    if pincp <= 0:
        return 1
    elif 20472 >= pincp > 0:
        return 1
    elif 34739 >= pincp > 20472:
        return 2
    elif 85641 >= pincp > 34739:
        return 3
    elif 214462 >= pincp > 85641:
        return 4
    elif pincp > 214462:
        return 5
    else:
        return 0

def occupation(indp):
    if 290 >= indp >= 170:
        return 5
    elif 490 >= indp >= 370:
        return 4
    elif 690 >= indp >= 570:
        return 4
    elif indp == 770:
        return 4
    elif 3990 >= indp >= 1070:
        return 3
    elif 4590 >= indp >= 4070:
        return 2
    elif 5790 >= indp >= 4670:
        return 2
    elif 6390 >= indp >= 6070:
        return 3
    elif 6780 >= indp >= 6470:
        return 1
    elif 7190 >= indp >= 6870:
        return 1
    elif 7790 >= indp >= 7290:
        return 1
    elif 7890 >= indp >= 7870:
        return 1
    elif 8290 >= indp >= 7970:
        return 1
    elif 8470 >= indp >= 8370:
        return 3
    elif 8690 >= indp >= 8560:
        return 3
    elif 9290 >= indp >= 8870:
        return 3
    elif 9590 >= indp >= 9370:
        return 2
    elif 9870 >= indp >= 9670:
        return 0
    elif indp == 9920:
        return 0
    else:
        return 0

def employment(esr, agegrp):
    switch = {1: 1, 2: 2, 3: 4,
              4: 1, 5: 2}
    if esr == 6:
        if agegrp == 1: # If not in labor force, depends on age.
            return 8 # If young, probably in school
        else: # Can actually find if they're going to school or not!!!!!!
            return 5 # If older, probably retired
    else:
        return switch.get(esr, 0)

def education(schl):
    if 11 >= schl:
        return 1
    elif 15 >= schl >= 12:
        return 2
    elif 17 >= schl >= 16:
        return 3
    elif 20 >= schl >= 18:
        return 5
    elif schl > 21:
        return 6
    else:
        return 0

def marriage(mar):
    switch = {1: 1, 2: 5, 3: 3,
              4: 4, 5: 2}
    return switch.get(mar, 0)

def turnout(race):
    race_switch = {1: 'WHITE', 2: 'BLACK', 3: 'ASIAN', 5: 'HISPANIC'}
    
    race_p = state_turnout[race_switch.get(race, 'OTHER')][state]    
    
    return race_p
    

print('Loading CSV...', end = "")
df = pd.read_csv('psam_p35.csv')
print(state, 'Loaded!')
year = 2016

#df = df.loc[:, :][df.AGEP > 17]

with open('keepcols.txt') as file:
    keepkeys = [line.rstrip('\n') for line in file]

print('Trimming...')
for key in df.keys():
    if key not in keepkeys:
        df.drop(columns = key, inplace = True)
    else:
        df[key] = pd.to_numeric(df[key], errors = 'coerce')

#df = df[df.AGEP > 17]

#df.to_csv('acs_' + state + '_16_og.csv', index = False)

print('Making ANES columns...')

with open('makecols.txt') as file:
    makekeys = [line.rstrip('\n') for line in file]

for key in makekeys:
    df.loc[:, key] = np.zeros(len(df))

print('Formatting to ANES...')
for i in range(len(df)):
    if df['CIT'][i] == 5:
        df.drop([i], inplace = True)
        print('\nNonvoter:', i)
    else:
        print('Progress...', i+1, '/', len(df), end = '\r')
        df.loc[i, 'VCF0004'] = year
        df.loc[i, 'VCF0101'] = ageGroup(df['AGEP'][i])
        df.loc[i, 'VCF0104'] = df['SEX'][i]
        df.loc[i, 'VCF0105a'] = race(df['FHISP'][i], df['RAC1P'][i])
        df.loc[i, 'VCF0112'] = region(df['ST'][i])
        df.loc[i, 'VCF0114'] = percentile(df['PINCP'][i])
        df.loc[i, 'VCF0115'] = occupation(df['INDP'][i])
        df.loc[i, 'VCF0116'] = employment(df['ESR'][i], df['VCF0101'][i])
        df.loc[i, 'VCF0140'] = education(df['SCHL'][i])
        df.loc[i, 'VCF0147'] = marriage(df['MAR'][i])
        df.loc[i, 'VCF0009z'] = df['PWGTP'][i] * turnout(df['VCF0105a'][i])

df = df[df.AGEP > 17]

for key in df.keys():
    if key not in makekeys:
        df.drop(columns = key, inplace = True)

df.to_csv('acs_' + state + '_15.csv', index = False)
print('\nSaved and done!')
