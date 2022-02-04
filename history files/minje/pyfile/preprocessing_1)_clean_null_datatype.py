import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from glob import glob
import warnings


#--------------------------------------------데이터 import------------------------------------------------#
# 데이터 저장을 원하는 path를 미리 설정해주세요
# 현재 디렉토리에 그냥 저장하고 싶으면 해당 셀을 지우고 실행하시면 되겠습니다
# 단, path내에 train이라는 이름으로 train 데이터가 있어야하고, test라는 이름으로 test 데이터가 있어야 합니다.

# 데이터가 담겨있는 path로 설정하기
warnings.filterwarnings(action='ignore') 

know_train = [pd.read_csv(path) for path in sorted(glob('./data_0103/train/*.csv'))]
know_test = [pd.read_csv(path) for path in sorted(glob('./data_0103/test/*.csv'))]

#--------------------------------------------------------------------------------------------------------#

# 연도별로 텍스트로 명시된 문항
years = ['2017','2018','2019','2020']

text_2017_question = ['bq4_1a','bq4_1b','bq4_1c','bq5_2','bq19_1','bq30','bq31','bq32','bq33','bq34','bq38_1']
text_2018_question = ['bq4_1a','bq4_1b','bq4_1c','bq5_2','bq28_1','bq29','bq30','bq31','bq32','bq33','bq37_1']
text_2019_question = ['bq4_1a','bq4_1b','bq4_1c','bq5_2','bq18_10','bq20_1','bq22','bq23','bq24','bq27_1']
text_2020_question = ['bq4_1a','bq4_1b','bq4_1c','bq5_2','bq18_10','bq20_1','bq26_1']
whole_list = [text_2017_question, text_2018_question, text_2019_question, text_2020_question]

text_dict = {}
for year, lst in zip(years, whole_list):
    text_dict[year] = lst
    
#--------------------------------------------------------------------------------------------------------#
#                                               NaN 처리                                                 #
#--------------------------------------------------------------------------------------------------------#
# Descripyion : Nan 값과 공백을 모두 str 0으로 처리함(문항 상에 0이 존재하지 않기에 해석 상의 오류 없음)

# nan 값을 0으로 대체합시다
for df in know_train:
    df.fillna('0',inplace=True)

for df in know_test:
    df.fillna('0',inplace=True)
    
# train set에 대하여 공백은 모두 str 0으로 전환
for df in know_train:
    for col in df.columns:
        df[col].replace(' ', '0', inplace=True)
        
# test set에 대하여 공백은 모두 str 0으로 전환
for df in know_test:
    for col in df.columns:
        df[col].replace(' ', '0', inplace=True)

#--------------------------------------------------------------------------------------------------------#
#                                            Data Type 처리                                              #
#--------------------------------------------------------------------------------------------------------#
# Descripyion : 숫자로 질문했는데 주관식으로 답변한 응답자가 존재하여 데이터 타입 변경이 불가능한 문항이 연도별로 존재함
#               해당 질문을 드랍하고 무결한 데이터타입을 가진 csv를 생성함
# train 
# {
    # '2017': [] 
    # '2018': ['bq37', 'bq36', 'bq40', 'bq35', 'bq28', 'bq34', 'bq4', 'bq5_1'],
    # '2019': ['bq25','bq10','bq8_3','bq12_4','bq20','bq9','bq11','bq27','bq21_3','bq6','bq7'],
    # '2020': []
#  }
# test
#{
    # '2017': [],
    # '2018': ['bq37', 'bq35', 'bq28', 'bq4', 'bq5_1'],
    # '2019': ['bq19', 'bq21_3', 'bq27', 'bq20'],
    # '2020': []
# }


# int type으로 바꾸는 것을 시도해보고 안된다면 str로 취급하여 정리해봅시다 
print('train')
for year, df in zip(years, know_train):
    print(year)
    
    for col in df.columns:
        try:
            df[col] = df[col].map(int) 
        except:
            df[col] = df[col].map(str)
print()
print('test')
for year, df in zip(years, know_test):
    print(year)
    
    for col in df.columns:
        try:
            df[col] = df[col].map(int) 
        except:
            df[col] = df[col].map(str)

def check_datatype_integrity(years, know_data, verbose=True):
    
    trouble_candidate_dict = {}
    
    # 데이터 타입 변환이 잘 이루어졌는지 체크하고 문제가 있는 문항을 딕셔너리에 담습니다
    for year,df in zip(years,know_data):
        compare_a = set(df.dtypes[df.dtypes=='object'].index)
        compare_b = set(text_dict[year])
        if verbose == True:
            print(year)
        if (compare_a - compare_b) == set():
            if verbose == True:
                print('데이터 타입 변환이 잘 이루어 졌습니다')
            trouble_candidate_dict[year] = (compare_a - compare_b)
        else:
            if verbose == True:
                print('문제가 생겼습니다 아래 문항들의 데이터 타입을 확인해보세요')
                print(compare_a - compare_b)
            trouble_candidate_dict[year] = (compare_a - compare_b)
    if verbose == True:
        print('-'*20)    
    
    # 문제가 있는 문항을 출력해서 하나씩 살펴봅시다
    trouble_dict = {}
    
    for year in years:
        trouble_dict[year] = []
        
    for year,df in zip(years,know_data):
        if verbose == True:
            print(year,'년 오류문항 Overview')
            for col in trouble_candidate_dict[year]:
                print(col,':',df[col].unique())
            print()
        for col in trouble_candidate_dict[year]:
            try:
                df[col] = df[col].map(float) # float으로 바꿀 수 있다면 바꿔주고
            except:
                trouble_dict[year].append(col) # 오류가 있다면 딕셔너리에 담아줍시다
    
    return trouble_dict

# 데이터 타입에 문제가 있는 문항들을 딕셔너리로 추출함
trouble_train_dict = check_datatype_integrity(years, know_train, verbose=False)
trouble_test_dict = check_datatype_integrity(years, know_test, verbose=False)


#--------------------------------------------------------------------------------------------------------#
#                            잘못 답한 사람들의 대답들을 일일이 0으로 바꿔주기                              #
#--------------------------------------------------------------------------------------------------------#

# ---------------------------------------------2017년----------------------------------------------------#
# 문제되는 문항이 없음
df = know_train[0]
df.to_csv('KNOW_2017.csv',index=False)
df = know_test[0]
df.to_csv('KNOW_2017_test.csv',index=False)

# ---------------------------------------------2018년----------------------------------------------------#
# train : ['bq36', 'bq34', 'bq37', 'bq40', 'bq35', 'bq5_1', 'bq28', 'bq4'] 
# test : ['bq37', 'bq35', 'bq5_1', 'bq28', 'bq4']

#### train
df = know_train[1]

# bq36
df.loc[df[df['bq36']=='디자이너'].index,'bq36'] = 0 
df['bq36'] = df['bq36'].map(float) 

# bq34
df.loc[df[df['bq34']=='자동차과'].index,'bq34'] = 0
df.loc[df[df['bq34']==' 포토샵'].index,'bq34'] = 0 
df['bq34'] = df['bq34'].map(float) 

# bq37
remove_set = {'1','2','3','4','5','6'}
new_words_list = [i for i in df['bq37'].unique() if i not in remove_set]

for word in new_words_list:
    df.loc[df[df['bq37']==word].index,'bq37'] = 0 
df['bq37'] = df['bq37'].map(float) 

# bq40
df.loc[df[df['bq40']=='지식재산학'].index,'bq40'] = 0 
df['bq40'] = df['bq40'].map(float) 

# bq35
df.loc[df[df['bq35']==' 라이트룸 등의 컴퓨터 프로그램"'].index,'bq35'] = 0 
df['bq35'] = df['bq35'].map(float) 

# bq5_1
remove_set = {'1','2','3','4','5','6','0'}
new_words_list = [i for i in df['bq5_1'].unique() if i not in remove_set]

for word in new_words_list:
    df.loc[df[df['bq5_1']==word].index,'bq5_1'] = 0 
df['bq5_1'] = df['bq5_1'].map(float) 

# bq28
remove_set = {'1','2','3','4','5'}
new_words_list = [i for i in df['bq28'].unique() if i not in remove_set]

for word in new_words_list:
    df.loc[df[df['bq28']==word].index,'bq28'] = 0 
df['bq28'] = df['bq28'].map(float) 

# bq4
remove_set = {'1','2','0'}
new_words_list = [i for i in df['bq4'].unique() if i not in remove_set]

for word in new_words_list:
    df.loc[df[df['bq4']==word].index,'bq4'] = 0 
df['bq4'] = df['bq4'].map(float) 

df.to_csv('KNOW_2018.csv',index=False)

#### test

df = know_test[1]

# bq37
remove_set = {'1','2','3','4','5','6','0'}
new_words_list = [i for i in df['bq37'].unique() if i not in remove_set]

for word in new_words_list:
    df.loc[df[df['bq37']==word].index,'bq37'] = 0 
df['bq37'] = df['bq37'].map(float) 

# bq35
df.loc[df[df['bq35']=='문예창작'].index,'bq35'] = 0 
df['bq35'] = df['bq35'].map(float) 

# bq28
remove_set = {'1','2','3','4','5'}
new_words_list = [i for i in df['bq28'].unique() if i not in remove_set]

for word in new_words_list:
    df.loc[df[df['bq28']==word].index,'bq28'] = 0 
df['bq28'] = df['bq28'].map(float) 

# bq5_1
remove_set = {'1','2','3','4','5','0'}
new_words_list = [i for i in df['bq5_1'].unique() if i not in remove_set]

for word in new_words_list:
    df.loc[df[df['bq5_1']==word].index,'bq5_1'] = 0 
df['bq5_1'] = df['bq5_1'].map(float)

# bq4
remove_set = {'1','2','0'}
new_words_list = [i for i in df['bq4'].unique() if i not in remove_set]

for word in new_words_list:
    df.loc[df[df['bq4']==word].index,'bq4'] = 0 
df['bq4'] = df['bq4'].map(float) 

df.to_csv('KNOW_2018_test.csv',index=False)

# ---------------------------------------------2019년----------------------------------------------------#
# train : ['bq25','bq10','bq8_3','bq12_4','bq20','bq9','bq11','bq27','bq21_3','bq6','bq7'] 
# test : ['bq19', 'bq21_3', 'bq27', 'bq20']

#### train
df = know_train[2]

# bq25
df.loc[df[df['bq25']==' 일러스트레이터"'].index,'bq25'] = 0 
df['bq25'] = df['bq25'].map(float) 

# bq10
remove_set = {'1','2','3','4','5'}
new_words_list = [i for i in df['bq10'].unique() if i not in remove_set]

for word in new_words_list:
    df.loc[df[df['bq10']==word].index,'bq10'] = 0 
df['bq10'] = df['bq10'].map(float) 

# bq8_3
df.loc[df[df['bq8_3']=='정보·기록학'].index,'bq8_3'] = 0 
df['bq8_3'] = df['bq8_3'].map(float) 

# bq12_4
remove_set = {'1','2','3','4','5','9'}
new_words_list = [i for i in df['bq12_4'].unique() if i not in remove_set]

for word in new_words_list:
    df.loc[df[df['bq12_4']==word].index,'bq12_4'] = 0 
df['bq12_4'] = df['bq12_4'].map(float) 

# bq20
remove_set = {'1','2','3','4','5','0'}
new_words_list = [i for i in df['bq20'].unique() if i not in remove_set]

for word in new_words_list:
    df.loc[df[df['bq20']==word].index,'bq20'] = 0 
df['bq20'] = df['bq20'].map(float) 

# bq9
df.loc[df[df['bq9']=='디자이너'].index,'bq9'] = 0 
df['bq9'] = df['bq9'].map(float) 

# bq11
remove_set = {'1','2','3','4','5','0'}
new_words_list = [i for i in df['bq11'].unique() if i not in remove_set]

for word in new_words_list:
    df.loc[df[df['bq11']==word].index,'bq11'] = 0 
df['bq11'] = df['bq11'].map(float) 

# bq27
remove_set = {'1','2','3','4','5','0'}
new_words_list = [i for i in df['bq27'].unique() if i not in remove_set]

for word in new_words_list:
    df.loc[df[df['bq27']==word].index,'bq27'] = 0 
df['bq27'] = df['bq27'].map(float) 

# bq21_3
remove_set = {'1','2','3','4','5','0'}
new_words_list = [i for i in df['bq21_3'].unique() if i not in remove_set]

for word in new_words_list:
    df.loc[df[df['bq21_3']==word].index,'bq21_3'] = 0 
df['bq21_3'] = df['bq21_3'].map(float) 

# bq6
remove_set = {'1','2','3','4','5','6','7','0'}
new_words_list = [i for i in df['bq6'].unique() if i not in remove_set]

for word in new_words_list:
    df.loc[df[df['bq6']==word].index,'bq6'] = 0 
df['bq6'] = df['bq6'].map(float) 

# bq7
remove_set = {'1','2','3','4','5','6','7'}
new_words_list = [i for i in df['bq7'].unique() if i not in remove_set]

for word in new_words_list:
    df.loc[df[df['bq7']==word].index,'bq7'] = 0 
df['bq7'] = df['bq7'].map(float) 

df.to_csv('KNOW_2019.csv',index=False)

#### test 

df = know_test[2]

# bq19
remove_set = {'1','2','3','4','5'}
new_words_list = [i for i in df['bq19'].unique() if i not in remove_set]

for word in new_words_list:
    df.loc[df[df['bq19']==word].index,'bq19'] = 0 
df['bq19'] = df['bq19'].map(float) 

# bq21_3
df.loc[df[df['bq21_3']=='사회학/광고홍보학'].index,'bq21_3'] = 0 
df['bq21_3'] = df['bq21_3'].map(float) 

# bq27
remove_set = {'1','2','3','4','5','6'}
new_words_list = [i for i in df['bq27'].unique() if i not in remove_set]

for word in new_words_list:
    df.loc[df[df['bq27']==word].index,'bq27'] = 0 
df['bq27'] = df['bq27'].map(float) 

# bq20
remove_set = {'1','2','3','4','5'}
new_words_list = [i for i in df['bq20'].unique() if i not in remove_set]

for word in new_words_list:
    df.loc[df[df['bq20']==word].index,'bq20'] = 0 
df['bq20'] = df['bq20'].map(float) 

df.to_csv('KNOW_2019_test.csv',index=False)

# ---------------------------------------------2020년----------------------------------------------------#
# 문제되는 문항이 없음

df = know_train[3]
df.to_csv('KNOW_2020.csv',index=False)
df = know_test[3]
df.to_csv('KNOW_2020_test.csv',index=False)