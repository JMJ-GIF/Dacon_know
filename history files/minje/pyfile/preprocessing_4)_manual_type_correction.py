import pandas as pd
import numpy as np
from glob import glob
from tqdm import tqdm
import warnings

RANDOM_STATE = 42
np.seed = 42
DATA_PATH = "../data_0113/"

warnings.filterwarnings(action='ignore')
PATH_2017 = DATA_PATH + "train/KNOW_2017.csv"
PATH_2018 = DATA_PATH + "train/KNOW_2018.csv"
PATH_2019 = DATA_PATH + "train/KNOW_2019.csv"
PATH_2020 = DATA_PATH + "train/KNOW_2020.csv"

paths = [PATH_2017, PATH_2018, PATH_2019, PATH_2020]

know_train = [pd.read_csv(path) for path in paths]

TEST_PATH_2017 = DATA_PATH + "test/KNOW_2017_test.csv"
TEST_PATH_2018 = DATA_PATH + "test/KNOW_2018_test.csv"
TEST_PATH_2019 = DATA_PATH + "test/KNOW_2019_test.csv"
TEST_PATH_2020 = DATA_PATH + "test/KNOW_2020_test.csv"

TEST_PATHs = [TEST_PATH_2017, TEST_PATH_2018, TEST_PATH_2019, TEST_PATH_2020]

know_test = [pd.read_csv(path) for path in TEST_PATHs]

# -------------------------------------------------------------------------------------------------------------------------#
#                                                     2017                                                                 #
# -------------------------------------------------------------------------------------------------------------------------#
numeric_cols_2017 = ['aq1_1', 'aq1_2', 'aq2_1', 'aq2_2', 'aq3_1', 'aq3_2', 'aq4_1', 'aq4_2', 'aq5_1', 'aq5_2', 
                     'aq6_1', 'aq6_2', 'aq7_1', 'aq7_2', 'aq8_1', 'aq8_2', 'aq9_1', 'aq9_2', 'aq10_1', 'aq10_2',
                    'aq11_1', 'aq11_2', 'aq12_1', 'aq12_2', 'aq13_1', 'aq13_2', 'aq14_1', 'aq14_2', 'aq15_1', 'aq15_2', 
                     'aq16_1', 'aq16_2', 'aq17_1', 'aq17_2', 'aq18_1', 'aq18_2', 'aq19_1', 'aq19_2', 'aq20_1', 'aq20_2',
                    'aq21_1', 'aq21_2', 'aq22_1', 'aq22_2', 'aq23_1', 'aq23_2', 'aq24_1', 'aq24_2', 'aq25_1', 'aq25_2', 
                     'aq26_1', 'aq26_2', 'aq27_1', 'aq27_2', 'aq28_1', 'aq28_2', 'aq29_1', 'aq29_2', 'aq30_1', 'aq30_2',
                    'aq31_1', 'aq31_2', 'aq32_1', 'aq32_2', 'aq33_1', 'aq33_2', 'aq34_1', 'aq34_2', 'aq35_1', 'aq35_2', 
                     'aq36_1', 'aq36_2', 'aq37_1', 'aq37_2', 'aq38_1', 'aq38_2', 'aq39_1', 'aq39_2', 'aq40_1', 'aq40_2',
                     'aq41_1', 'aq41_2', 'bq3', 'bq5_1', 'bq7', 'bq8_1', 'bq8_2', 'bq8_3', 'bq9', 'bq10',
                     'bq11', 'bq12_1', 'bq12_5', 'bq13', 'bq14', 'bq15_1', 'bq15_2', 'bq15_3', 'bq16', 'bq17', 
                     'bq18_1', 'bq18_2', 'bq18_3', 'bq18_4', 'bq18_5', 'bq18_6', 'bq18_7', 'bq19', 'bq20', 'bq21', 
                     'bq22', 'bq25', 'bq26', 'bq27', 'bq28', 'bq29', 'bq35', 'bq37', 'bq38', ]
numeric_pure_cols_2017 = ['bq23', 'bq37', 'bq41_1', 'bq41_2', 'bq41_3',]
categorical_cols_2017 = ['bq1', 'bq2', 'bq39_1', 'bq39_2', ]
binary_cols_2017 = ['bq4', 'bq5', 'bq24_1', 'bq24_2', 'bq24_3', 'bq24_4', 'bq24_5', 'bq24_6', 'bq24_7', 'bq24_8', 'bq36', 'bq40', ]
half_cols_2017 = ['bq6', 'bq12_2', 'bq12_3', 'bq12_4', ]
text_cols_2017 = ['bq4_1a', 'bq4_1b', 'bq4_1c', 'bq5_2', 'bq19_1', 'bq30', 'bq31', 'bq32', 'bq33', 'bq34', 'bq38_1',]

all_cols_2017 = numeric_cols_2017 + numeric_pure_cols_2017\
                + categorical_cols_2017 + binary_cols_2017 + half_cols_2017 + text_cols_2017

# half cols 전처리
know_train[0]['bq6_mark']= 0
know_train[0]['bq6_mark'].loc[know_train[0]['bq6'] == 7] = 1
know_train[0]['bq12_2_mark']= 0
know_train[0]['bq12_2_mark'].loc[know_train[0]['bq12_2'] == 9] = 1
know_train[0]['bq12_3_mark']= 0
know_train[0]['bq12_3_mark'].loc[know_train[0]['bq12_3'] == 9] = 1
know_train[0]['bq12_4_mark']= 0
know_train[0]['bq12_4_mark'].loc[know_train[0]['bq12_4'] == 9] = 1

know_train[0].loc[know_train[0]['bq6'] == 7, 'bq6'] = 3 # median filling
know_train[0].loc[know_train[0]['bq12_2'] == 9, 'bq12_2'] = 3 # median filling
know_train[0].loc[know_train[0]['bq12_3'] == 9, 'bq12_3'] = 3 # median filling
know_train[0].loc[know_train[0]['bq12_4'] == 9, 'bq12_4'] = 3 # median filling

half_cols_mark_2017 = ['bq6_mark', 'bq12_2_mark', 'bq12_3_mark', 'bq12_4_mark',]
                
# -------------------------------------------------------------------------------------------------------------------------#
#                                                     2018                                                                 #
# -------------------------------------------------------------------------------------------------------------------------#
             
# half cols 전처리
know_train[0]['bq6_mark']= 0
know_train[0]['bq6_mark'].loc[know_train[0]['bq6'] == 7] = 1
know_train[0]['bq12_2_mark']= 0
know_train[0]['bq12_2_mark'].loc[know_train[0]['bq12_2'] == 9] = 1
know_train[0]['bq12_3_mark']= 0
know_train[0]['bq12_3_mark'].loc[know_train[0]['bq12_3'] == 9] = 1
know_train[0]['bq12_4_mark']= 0
know_train[0]['bq12_4_mark'].loc[know_train[0]['bq12_4'] == 9] = 1

know_train[0].loc[know_train[0]['bq6'] == 7, 'bq6'] = 3 # median filling
know_train[0].loc[know_train[0]['bq12_2'] == 9, 'bq12_2'] = 3 # median filling
know_train[0].loc[know_train[0]['bq12_3'] == 9, 'bq12_3'] = 3 # median filling
know_train[0].loc[know_train[0]['bq12_4'] == 9, 'bq12_4'] = 3 # median filling

half_cols_mark_2017 = ['bq6_mark', 'bq12_2_mark', 'bq12_3_mark', 'bq12_4_mark',]

numeric_cols_2018 = ['cq1', 'cq2', 'cq3', 'cq4', 'cq5', 'cq6', 'cq7', 'cq8', 'cq9', 'cq10', 
                    'cq11', 'cq12', 'cq13', 'cq14', 'cq15', 'cq16', 'cq17', 'cq18', 'cq19', 'cq20', 
                    'cq21', 'cq22', 'cq23', 'cq24', 'cq25', 'cq26', 'cq27', 'cq28', 'cq29', 'cq30',
                    'cq31', 'cq32', 'cq33', 'cq34', 'cq35', 'cq36', 'cq37', 'cq38', 'cq39', 'cq40', 
                    'cq41', 'cq42', 'cq43', 'cq44', 'cq45', 'cq46', 'cq47', 'cq49', 'cq50_1', 'cq50_2', 'cq50_3', 'cq50_4', 'cq50_5', 'cq50_6', 'cq50_7', 'cq50_8',
                    'iq1', 'iq2', 'iq3', 'iq4', 'iq5', 'iq6',
                    'bq3', 'bq5_1', 'bq7', 'bq8_1', 'bq8_2', 'bq8_3', 'bq9', 'bq10', 'bq11', 'bq12_1', 'bq12_5', 'bq18', 'bq19', 'bq20', 'bq25', 'bq26_1a', 
                     'bq26_2a', 'bq26_3a', 'bq26_4a', 'bq28', 'bq34', 'bq37', ]
numeric_pure_cols_2018 = ['bq21', 'bq36', 'bq40', 'bq41_1', 'bq41_2', 'bq41_3', ]
categorical_cols_2018 = ['bq1', 'bq2', 'bq13', 'bq15', 'bq17', 'bq22', 'bq23', 'bq24', 'bq26_1', 'bq26_2', 'bq26_3', 'bq26_4', 'bq38', 'bq38_1', 'bq38_2', ]
binary_cols_2018 = ['cq48', 'bq4', 'bq5', 'bq14', 'bq16', 'bq25_1', 'bq27', 'bq35', 'bq39', ]
half_cols_2018 = ['bq6', 'bq12_2', 'bq12_3', 'bq12_4', ]
text_cols_2018 = ['bq4_1a', 'bq4_1b', 'bq4_1c', 'bq5_2', 'bq28_1', 'bq29', 'bq30', 'bq31', 'bq32', 'bq33', 'bq37_1', ]

all_cols_2018 = numeric_cols_2018 + numeric_pure_cols_2018\
                + categorical_cols_2018 + binary_cols_2018 + half_cols_2018 + text_cols_2018

# half cols 전처리
know_train[1]['bq6_mark']= 0
know_train[1]['bq6_mark'].loc[know_train[1]['bq6'] == 7] = 1
know_train[1]['bq12_2_mark']= 0
know_train[1]['bq12_2_mark'].loc[know_train[1]['bq12_2'] == 9] = 1
know_train[1]['bq12_3_mark']= 0
know_train[1]['bq12_3_mark'].loc[know_train[1]['bq12_3'] == 9] = 1
know_train[1]['bq12_4_mark']= 0
know_train[1]['bq12_4_mark'].loc[know_train[1]['bq12_4'] == 9] = 1

know_train[1].loc[know_train[1]['bq6'] == 7, 'bq6'] = 3 # median filling
know_train[1].loc[know_train[1]['bq12_2'] == 9, 'bq12_2'] = 3 # median filling
know_train[1].loc[know_train[1]['bq12_3'] == 9, 'bq12_3'] = 3 # median filling
know_train[1].loc[know_train[1]['bq12_4'] == 9, 'bq12_4'] = 3 # median filling

half_cols_mark_2018 = ['bq6_mark', 'bq12_2_mark', 'bq12_3_mark', 'bq12_4_mark',]

know_test[1]['bq22'] =  know_test[1]['bq221'] + know_test[1]['bq222'] + know_test[1]['bq223']
know_test[1]['bq23'] =  know_test[1]['bq231'] + know_test[1]['bq232'] + know_test[1]['bq233']+\
                         know_test[1]['bq234'] + know_test[1]['bq235']
know_test[1]['bq24'] =  know_test[1]['bq241'] + know_test[1]['bq242'] + know_test[1]['bq243']+\
                         know_test[1]['bq244'] + know_test[1]['bq245']
know_test[1].drop(['bq221',
              'bq222',
              'bq223',
              'bq231',
              'bq232',
              'bq233',
              'bq234',
              'bq235',
              'bq241',
              'bq242',
              'bq243',
              'bq244',
              'bq245',], axis=1, inplace=True)   


# -------------------------------------------------------------------------------------------------------------------------#
#                                                     2019                                                                 #
# -------------------------------------------------------------------------------------------------------------------------#

numeric_cols_2019 = ['sq1', 'sq2', 'sq3', 'sq4', 'sq5', 'sq6', 'sq7', 'sq8', 'sq9', 'sq10', 
                     'sq11', 'sq12', 'sq13', 'sq14', 'sq15', 'sq16',
                    'kq1_1', 'kq1_2', 'kq2_1', 'kq2_2', 'kq3_1', 'kq3_2', 'kq4_1', 'kq4_2', 'kq5_1', 'kq5_2', 
                     'kq6_1', 'kq6_2', 'kq7_1', 'kq7_2', 'kq8_1', 'kq8_2', 'kq9_1', 'kq9_2', 'kq10_1', 'kq10_2',
                    'kq11_1', 'kq11_2', 'kq12_1', 'kq12_2', 'kq13_1', 'kq13_2', 'kq14_1', 'kq14_2', 'kq15_1', 'kq15_2',
                     'kq16_1', 'kq16_2', 'kq17_1', 'kq17_2', 'kq18_1', 'kq18_2', 'kq19_1', 'kq19_2', 'kq20_1', 'kq20_2',
                    'kq21_1', 'kq21_2', 'kq22_1', 'kq22_2', 'kq23_1', 'kq23_2', 'kq24_1', 'kq24_2', 'kq25_1', 'kq25_2', 
                     'kq26_1', 'kq26_2', 'kq27_1', 'kq27_2', 'kq28_1', 'kq28_2', 'kq29_1', 'kq29_2', 'kq30_1', 'kq30_2',
                    'kq31_1', 'kq31_2', 'kq32_1', 'kq32_2', 'kq33_1', 'kq33_2', 
                    'bq3', 'bq5_1', 'bq7', 'bq8_1', 'bq8_2', 'bq8_3', 'bq9', 'bq10', 'bq11', 'bq12_1', 'bq12_5', 
                     'bq13_1', 'bq13_2', 'bq13_3', 'bq14_1', 'bq14_2', 'bq14_3', 'bq14_4', 'bq14_5', 'bq15', 
                     'bq16_1', 'bq16_2', 'bq16_3', 'bq16_4', 'bq16_5', 'bq17', 'bq19', 'bq20', 'bq21_1', 'bq21_2', 'bq21_3', 'bq27',]
numeric_pure_cols_2019 = ['bq26', 'bq30', 'bq31_1', 'bq31_2', 'bq31_3', ]
categorical_cols_2019 = ['bq1', 'bq2', 'bq28', 'bq28_1', 'bq28_2', ]
binary_cols_2019 = ['bq4', 'bq5', 'bq18_1', 'bq18_2', 'bq18_3', 'bq18_4', 'bq18_5', 'bq18_6', 'bq18_7', 
                    'bq18_8', 'bq18_9', 'bq25', 'bq29', ]
half_cols_2019 = ['bq6', 'bq12_2', 'bq12_3', 'bq12_4', ]
text_cols_2019 = ['bq4_1a', 'bq4_1b', 'bq4_1c', 'bq5_2', 'bq18_10', 'bq20_1', 'bq22', 'bq23', 'bq24', 'bq27_1',]

all_cols_2019 = numeric_cols_2019 + numeric_pure_cols_2019\
                + categorical_cols_2019 + binary_cols_2019 + half_cols_2019 + text_cols_2019
                
# half cols 전처리
know_train[2]['bq6_mark']= 0
know_train[2]['bq6_mark'].loc[know_train[2]['bq6'] == 7] = 1
know_train[2]['bq12_2_mark']= 0
know_train[2]['bq12_2_mark'].loc[know_train[2]['bq12_2'] == 9] = 1
know_train[2]['bq12_3_mark']= 0
know_train[2]['bq12_3_mark'].loc[know_train[2]['bq12_3'] == 9] = 1
know_train[2]['bq12_4_mark']= 0
know_train[2]['bq12_4_mark'].loc[know_train[2]['bq12_4'] == 9] = 1

know_train[2].loc[know_train[2]['bq6'] == 7, 'bq6'] = 3 # median filling
know_train[2].loc[know_train[2]['bq12_2'] == 9, 'bq12_2'] = 3 # median filling
know_train[2].loc[know_train[2]['bq12_3'] == 9, 'bq12_3'] = 3 # median filling
know_train[2].loc[know_train[2]['bq12_4'] == 9, 'bq12_4'] = 3 # median filling

# -------------------------------------------------------------------------------------------------------------------------#
#                                                     2020                                                                 #
# -------------------------------------------------------------------------------------------------------------------------#

half_cols_mark_2019 = ['bq6_mark', 'bq12_2_mark', 'bq12_3_mark', 'bq12_4_mark',]

numeric_cols_2020 = ['saq1_1', 'saq1_2', 'saq2_1', 'saq2_2', 'saq3_1', 'saq3_2', 'saq4_1', 'saq4_2', 'saq5_1', 'saq5_2',
                     'saq6_1', 'saq6_2', 'saq7_1', 'saq7_2', 'saq8_1', 'saq8_2', 'saq9_1', 'saq9_2', 'saq10_1', 'saq10_2',
                    'saq11_1', 'saq11_2', 'saq12_1', 'saq12_2', 'saq13_1', 'saq13_2', 'saq14_1', 'saq14_2', 'saq15_1',
                     'saq15_2', 'saq16_1', 'saq16_2', 'saq17_1', 'saq17_2', 'saq18_1', 'saq18_2', 'saq19_1', 'saq19_2',
                     'saq20_1', 'saq20_2','saq21_1', 'saq21_2', 'saq22_1', 'saq22_2', 'saq23_1', 'saq23_2', 'saq24_1', 'saq24_2',
                     'saq25_1', 'saq25_2', 'saq26_1', 'saq26_2', 'saq27_1', 'saq27_2', 'saq28_1', 'saq28_2', 'saq29_1', 'saq29_2',
                     'saq30_1', 'saq30_2', 'saq31_1', 'saq31_2', 'saq32_1', 'saq32_2', 'saq33_1', 'saq33_2', 'saq34_1',
                     'saq34_2', 'saq35_1', 'saq35_2', 'saq36_1', 'saq36_2', 'saq37_1', 'saq37_2', 'saq38_1', 'saq38_2',
                     'saq39_1', 'saq39_2', 'saq40_1', 'saq40_2', 'saq41_1', 'saq41_2', 'saq42_1', 'saq42_2', 'saq43_1',
                     'saq43_2', 'saq44_1', 'saq44_2',
                    'vq1', 'vq2', 'vq3', 'vq4', 'vq5', 'vq6', 'vq7', 'vq8', 'vq9', 'vq10', 'vq11', 'vq12', 'vq13',
                    'bq3', 'bq5_1', 'bq7', 'bq8_1', 'bq8_2', 'bq8_3', 'bq9', 'bq10', 'bq11', 'bq12_1', 'bq12_5', 
                     'bq13_1', 'bq13_2', 'bq13_3', 'bq14_1', 'bq14_2', 'bq14_3', 'bq14_4', 'bq14_5', 'bq14_6', 'bq14_7', 
                     'bq15', 'bq16_1', 'bq16_2', 'bq16_3', 'bq16_4', 'bq16_5', 'bq16_6', 'bq16_7', 'bq16_8', 'bq16_9', 
                     'bq16_10', 'bq17', 'bq18_1', 'bq18_2', 'bq18_3', 'bq18_4', 'bq18_5', 'bq18_6', 'bq18_7', 'bq18_8', 
                     'bq18_9', 'bq19', 'bq20', 'bq21_1', 'bq21_2', 'bq21_3', 'bq22_1', 'bq22_2', 'bq22_3', 
                     'bq22_4', 'bq22_5', 'bq22_6', 'bq23_1', 'bq23_2', 'bq23_3', 'bq26',]
numeric_pure_cols_2020 = ['bq25', 'bq29', 'bq30_1', 'bq30_2', 'bq30_3', ]
categorical_cols_2020 = ['bq1', 'bq2', 'bq27_1', 'bq27_2',]
binary_cols_2020 = ['bq4', 'bq5', 'bq21_4', 'bq24', 'bq28',]
half_cols_2020 = ['bq6', 'bq12_2', 'bq12_3', 'bq12_4', ]
text_cols_2020 = ['bq4_1a', 'bq4_1b', 'bq4_1c', 'bq5_2', 'bq18_10', 'bq20_1', 'bq26_1',]

all_cols_2020 = numeric_cols_2020 + numeric_pure_cols_2020\
                + categorical_cols_2020 + binary_cols_2020 + half_cols_2020 + text_cols_2020
                
# half cols 전처리
know_train[3]['bq6_mark']= 0
know_train[3]['bq6_mark'].loc[know_train[3]['bq6'] == 7] = 1
know_train[3]['bq12_2_mark']= 0
know_train[3]['bq12_2_mark'].loc[know_train[3]['bq12_2'] == 9] = 1
know_train[3]['bq12_3_mark']= 0
know_train[3]['bq12_3_mark'].loc[know_train[3]['bq12_3'] == 9] = 1
know_train[3]['bq12_4_mark']= 0
know_train[3]['bq12_4_mark'].loc[know_train[3]['bq12_4'] == 9] = 1

know_train[3].loc[know_train[3]['bq6'] == 7, 'bq6'] = 3 # median filling
know_train[3].loc[know_train[3]['bq12_2'] == 9, 'bq12_2'] = 3 # median filling
know_train[3].loc[know_train[3]['bq12_3'] == 9, 'bq12_3'] = 3 # median filling
know_train[3].loc[know_train[3]['bq12_4'] == 9, 'bq12_4'] = 3 # median filling

half_cols_mark_2020 = ['bq6_mark', 'bq12_2_mark', 'bq12_3_mark', 'bq12_4_mark',]

# -------------------------------------------------------------------------------------------------------------------------#
#                                                     test                                                                 #
# -------------------------------------------------------------------------------------------------------------------------#

# half cols 전처리
for i in range(4):
    know_test[i]['bq6_mark']= 0
    know_test[i]['bq6_mark'].loc[know_test[i]['bq6'] == 7] = 1
    know_test[i]['bq12_2_mark']= 0
    know_test[i]['bq12_2_mark'].loc[know_test[i]['bq12_2'] == 9] = 1
    know_test[i]['bq12_3_mark']= 0
    know_test[i]['bq12_3_mark'].loc[know_test[i]['bq12_3'] == 9] = 1
    know_test[i]['bq12_4_mark']= 0
    know_test[i]['bq12_4_mark'].loc[know_test[i]['bq12_4'] == 9] = 1

for i in range(4):
    know_test[i].loc[know_test[i]['bq6'] == 7, 'bq6'] = 3 # median filling
    know_test[i].loc[know_test[i]['bq12_2'] == 9, 'bq12_2'] = 3 # median filling
    know_test[i].loc[know_test[i]['bq12_3'] == 9, 'bq12_3'] = 3 # median filling
    know_test[i].loc[know_test[i]['bq12_4'] == 9, 'bq12_4'] = 3 # median filling

years = ['2017', '2018', '2019', '2020']
for idx in range(4):
    train_data = know_train[idx]
    train_data.to_csv('KNOW_{}.csv'.format(years[idx]), index=False)
    test_data = know_test[idx]
    test_data.to_csv('KNOW_{}_test.csv'.format(years[idx]), index=False)