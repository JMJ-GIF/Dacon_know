import pandas as pd
import numpy as np
from glob import glob
from tqdm import tqdm
import warnings

RANDOM_STATE = 42
np.seed = 42
DATA_PATH = "./data_0115/"

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

half_cols_mark = ['bq6_mark', 'bq12_2_mark', 'bq12_3_mark', 'bq12_4_mark',]

# 전체적인 오류 0값의 분포를 봅시다
years = ['2017','2018','2019','2020']

zero_train_dict = {}
for year, df in zip(years,know_train):
    zero_dist = pd.DataFrame(index=['zero_sum'])
    for col in df.columns:
        zero_dist[col] = df[df[col]==0].shape[0]
    zero_dist = zero_dist.drop(['idx','knowcode','text_response','description'] + half_cols_mark, axis=1)     
    zero_train_dict[year] = zero_dist.T

# 전체적인 오류 0값의 분포를 봅시다
zero_test_dict = {}
for year, df in zip(years,know_test):
    zero_dist = pd.DataFrame(index=['zero_sum'])
    for col in df.columns:
        zero_dist[col] = df[df[col]==0].shape[0]
    zero_dist = zero_dist.drop(['idx','text_response'] + half_cols_mark, axis=1)   
    zero_test_dict[year] = zero_dist.T
    
def fill_mean(x):
    
    filled_x = x.copy()
    
    zero_indice = filled_x.loc[filled_x==0].index
    non_zero_indice = filled_x.loc[filled_x!=0].index
    mean_values = round(filled_x.loc[non_zero_indice].mean(),0)
    if len(zero_indice) == 0:
        return x
    else:
        filled_x.loc[zero_indice] = mean_values
        return filled_x

def fill_mode(x):
    filled_x = x.copy()
    
    zero_indice = filled_x.loc[filled_x==0].index
    non_zero_indice = filled_x.loc[filled_x!=0].index
    mode_values = round(filled_x.loc[non_zero_indice].mode(),0)[0]
    if len(zero_indice) == 0:
        return x
    else:
        filled_x.loc[zero_indice] = mode_values
        return filled_x


def data_imputation(error_cols_year, numeric_pure_cols_year, train_data, test_data):

    mean_fill_cols = []
    mode_fill_cols = []

    for col in error_cols_year:
        if col in numeric_pure_cols_year:
            mean_fill_cols.append(col)
        else:
            mode_fill_cols.append(col)
            
    for col in mean_fill_cols:
        train_data[col] = fill_mean(train_data[col])
        test_data[col] = fill_mean(test_data[col])
        
    for col in mode_fill_cols:
        train_data[col] = fill_mode(train_data[col])
        test_data[col] = fill_mode(test_data[col])

    return train_data, test_data

## 2017
# 설문지에서 건너뛰어도 된다고 말한 문항
skip_cols_2017 = ['aq1_2', 'aq2_2', 'aq3_2', 'aq4_2', 'aq5_2', 'aq6_2', 'aq7_2', 'aq8_2', 'aq9_2', 'aq10_2'
                    ,'aq11_2', 'aq12_2', 'aq13_2', 'aq14_2', 'aq15_2', 'aq16_2', 'aq17_2', 'aq18_2', 'aq19_2', 'aq20_2'
                    ,'aq21_2', 'aq22_2', 'aq23_2', 'aq24_2', 'aq25_2', 'aq26_2', 'aq27_2', 'aq28_2', 'aq29_2', 'aq30_2'
                    ,'aq31_2', 'aq32_2', 'aq33_2', 'aq34_2', 'aq35_2', 'aq36_2', 'aq37_2', 'aq38_2', 'aq39_2', 'aq40_2'
                    ,'aq41_2','bq5_1', 'bq40','bq41_1', 'bq41_2', 'bq41_3']
skip_txt_col = ['sim_job','bef_job','able_job','major']
numeric_pure_cols_2017 = ['bq23', 'bq37', 'bq41_1', 'bq41_2', 'bq41_3',]
# 0이 존재한다면 설문자의 오류로 발생한 문항
error_cols_2017 = [col for col in zero_train_dict['2017'].T.columns if col not in skip_cols_2017 + skip_txt_col]

## 2018
# 설문지에서 건너뛰어도 된다고 말한 문항
skip_cols_2018 = ['bq5_1','bq25_1','bq39','bq40','bq41_1','bq41_2','bq41_3']
skip_txt_col = ['major','sim_job','bef_job','able_job']
numeric_pure_cols_2018 = ['bq21', 'bq36', 'bq40', 'bq41_1', 'bq41_2', 'bq41_3', ]

# 0이 존재한다면 설문자의 오류로 발생한 문항
error_cols_2018 = [col for col in zero_train_dict['2018'].T.columns if col not in skip_cols_2018 + skip_txt_col]

## 2019
# 설문지에서 건너뛰어도 된다고 말한 문항
skip_cols_2019 = ['kq1_2', 'kq2_2', 'kq3_2', 'kq4_2', 'kq5_2', 'kq6_2', 'kq7_2', 'kq8_2', 'kq9_2', 'kq10_2'
                ,'kq11_2', 'kq12_2', 'kq13_2', 'kq14_2', 'kq15_2','kq16_2', 'kq17_2', 'kq18_2', 'kq19_2','kq20_2'
                ,'kq21_2', 'kq22_2', 'kq23_2', 'kq24_2', 'kq25_2','kq26_2', 'kq27_2' 'kq28_2', 'kq29_2', 'kq30_2'
                ,'kq31_2', 'kq32_2','kq33_2','bq5_1','bq29','bq30','bq31_1','bq31_2','bq31_3'
                ]
skip_txt_col = ['major','bef_job','able_job']
numeric_pure_cols_2019 = ['bq26', 'bq30', 'bq31_1', 'bq31_2', 'bq31_3', ]

# 0이 존재한다면 설문자의 오류로 발생한 문항
error_cols_2019 = [col for col in zero_train_dict['2019'].T.columns if col not in skip_cols_2019 + skip_txt_col]

## 2020
# 설문지에서 건너뛰어도 된다고 말한 문항
skip_cols_2020 = ['saq1_2', 'saq2_2', 'saq3_2', 'saq4_2', 'saq5_2','saq6_2', 'saq7_2', 'saq8_2', 'saq9_2', 'saq10_2'
                 ,'saq11_2', 'saq12_2', 'saq13_2', 'saq14_2','saq15_2', 'saq16_2', 'saq17_2', 'saq18_2', 'saq19_2'
                 ,'saq20_2', 'saq21_2', 'saq22_2', 'saq23_2', 'saq24_2', 'saq25_2', 'saq26_2', 'saq27_2', 'saq28_2', 'saq29_2'
                 ,'saq30_2', 'saq31_2', 'saq32_2', 'saq33_2', 'saq34_2', 'saq35_2','saq36_2', 'saq37_2', 'saq38_2' 
                 , 'saq39_2', 'saq40_2', 'saq41_2', 'saq42_2',  'saq43_2', 'saq44_2' 
                 ,'bq5_1','bq28','bq29','bq30_1','bq30_2','bq30_3'
                ]
skip_txt_col = ['major']
numeric_pure_cols_2020 = ['bq25', 'bq29', 'bq30_1', 'bq30_2', 'bq30_3', ]

# 0이 존재한다면 설문자의 오류로 발생한 문항
error_cols_2020 = [col for col in zero_train_dict['2020'].T.columns if col not in skip_cols_2020 + skip_txt_col]



## 내보내기
years = ['2017','2018','2019','2020']
error_cols_list = [error_cols_2017, error_cols_2018, error_cols_2019, error_cols_2020]
numeric_pure_cols_list = [numeric_pure_cols_2017, numeric_pure_cols_2018, numeric_pure_cols_2019,numeric_pure_cols_2020]


for idx in range(4):
    train_data, test_data = data_imputation(error_cols_list[idx], numeric_pure_cols_list[idx], know_train[idx], know_test[idx])
    train_data.to_csv('KNOW_{}.csv'.format(years[idx]),index=False)
    test_data.to_csv('KNOW_{}_test.csv'.format(years[idx]),index=False)