# 라이브러리 임포트하기
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from tqdm import tqdm
from glob import glob
import warnings

from konlpy.tag import Okt

# 데이터가 담겨있는 path로 설정하기
warnings.filterwarnings(action='ignore') 

know_train = [pd.read_csv(path) for path in sorted(glob('./data_0104/train/*.csv'))]
know_test = [pd.read_csv(path) for path in sorted(glob('./data_0104/test/*.csv'))]
pdf_list =  [pd.read_csv(path) for path in sorted(glob('./data_pdf_description/*.csv'))]

# Stopwords가 담겨있는 path를 설정하기
Stopwords_path = '../minje/stopwords.txt'

# -------------------------------------------------------------------------------------------------------------------------#
#                                                Input global variable                                                     #
# -------------------------------------------------------------------------------------------------------------------------#
# 1) years(4개의 연도)
# type : list

years = ['2017','2018','2019','2020']


# 2) text_dict (연도별로 str 처리된 문항들) 
# type : dict

text_2017_question = ['bq4_1a','bq4_1b','bq4_1c','bq5_2','bq19_1','bq30','bq31','bq32','bq33','bq34','bq38_1']
text_2018_question = ['bq4_1a','bq4_1b','bq4_1c','bq5_2','bq28_1','bq29','bq30','bq31','bq32','bq33','bq37_1']
text_2019_question = ['bq4_1a','bq4_1b','bq4_1c','bq5_2','bq18_10','bq20_1','bq22','bq23','bq24','bq27_1']
text_2020_question = ['bq4_1a','bq4_1b','bq4_1c','bq5_2','bq18_10','bq20_1','bq26_1']
whole_list = [text_2017_question, text_2018_question, text_2019_question, text_2020_question]

text_dict = {}
for year, lst in zip(years, whole_list):
    text_dict[year] = lst
       
        
# 3) stopwords(불용어 목록)
# type : list
 
# 불용어 목록 읽어오기(인터넷에서 일단 다운로드 받았습니다)
stopwords = []
f = open(Stopwords_path, 'rt', encoding='UTF8')
line = f.readline()
while line:
    line = f.readline().replace('\n',"")
    stopwords.append(line)
f.close()

# stopwords에서 공백을 제거합시다
remove_set = {''}
stopwords = [i for i in stopwords if i not in remove_set]

def preprocessing_text_data(year,know_data,preprocessing_question,exception_question):
    ''' 
    데이터 내에 존재하는 text 문항들을 konlpy를 통해 명사화 하고, 하나의 칼럼으로 합쳐줍니다.
    또한 text데이터 내에 존재하는 '0'도 모두 제거해줍니다.
    
    Parameters
    year(str)                    : 전처리 대상이 되는 데이터의 연도
    know_data(dataframe)         : 전처리 대상이 되는 데이터프레임
    preprocessing_question(list) : 약술형, 서술형으로 명사 추출 과정이 필요한 문항을 명시(이 리스트에 들어간 문항만 명사 추출)
    exception_question(list)     : 명사추출과 칼럼 병합에 있어 예외를 두고 싶은 문항(전처리 하지 않음)
    
    Returns
    know_data(dataframe) : 전처리가 완료된 데이터프레임
    '''

    # -------------------------------------------------------------------------------------------------------------------------#
    #                               약술형과 서술형 문항에 대해서 명사를 추출하고 따로 전처리 해주기                               #
    # -------------------------------------------------------------------------------------------------------------------------#
    
    ## konlpy를 이용한 lemmatization - 조사나 어미같이 단어에서 의미론적으로 필요없는 부분 정리하기
    for question in preprocessing_question:
        okt = Okt()
        target_question = [' '.join(okt.nouns(doc)) for doc in know_data[question]] # 명사를 뽑기
        
        # [(stopwords에 포함되어 있고) and (길이가 1인 명사)]는 제외
        doc_nouns_list = target_question
        new_doc_nouns_list = []

        for idx in range(len(doc_nouns_list)):
            doc_list = doc_nouns_list[idx].split()
            new_words = ''
            for word in doc_list:
                if word not in set(stopwords): # 단어가 불용어 사전에 존재하지 않고
                    if len(word) > 1: # 길이가 1인 경우는 필터링합니다
                        new_words += word
                        new_words += ' '
            new_doc_nouns_list.append(new_words[:-1])
            
        # 모두 필터링되어 아무것도 나타나지 않는 단어는 0으로 치환합니다
        result_list = []
        for word in new_doc_nouns_list:
            if word == '':
                result_list.append('0')
            else:
                result_list.append(word)
                
        # 전처리 과정을 모두 거친 데이터를 반환합니다
        know_data[question] = result_list

    # -------------------------------------------------------------------------------------------------------------------------#
    #                                           다른 문항들과 명사추출한 문항들을 합쳐주기                                        #
    # -------------------------------------------------------------------------------------------------------------------------#
       
    # preprocessing_question을 제외한 나머지 text 칼럼들은 전부 합쳐줍니다
    text_agg_col = pd.DataFrame(index=range(know_data.shape[0]))

    # 맨 처음 열은 미리 넣어줍시다
    idx = 0
    while True:
        if text_agg_col.shape[1] != 0: # 하나라도 데이터프레임에 들어가면 break
            break
        else:
            first_col = text_dict[year][idx] 
            if first_col not in exception_question: # 넣고자 하는 열이 예외문항이 아닌경우에만 병합
                text_agg_col['text_agg_col'] = know_data[first_col]
            else: # 아니면 다른 열을 탐색
                idx += 1

    # 넣은 열을 제외한 agg_col
    remove_set = {first_col}
    target_agg_col = [i for i in text_dict[year] if i not in remove_set]

    # 예외로 빼는 문항을 제외하고 모든 주관식 문항을 하나의 열에 합칩니다
    for text_col in target_agg_col:
        if text_col in exception_question: # 예외로 다룰 문항은 병합하면 안되므로 따로 빼주기
            continue
        text_agg_col['text_agg_col'] = text_agg_col['text_agg_col'] + ' ' + know_data[text_col]
        
    # 합친 문항을 원래 데이터에 넣어줍니다
    know_data['text_response'] = text_agg_col['text_agg_col']

    # -------------------------------------------------------------------------------------------------------------------------#
    #                                 최종적으로 구한 text_response열에 대해서 전처리하기                                        #
    # -------------------------------------------------------------------------------------------------------------------------#

    # 최종적으로 구한 text_response 열에 대해 마지막 전처리를 해줍니다
    # 1) text_response열에서 '0'인 데이터는 모두 제거해줍니다
    for idx in range(len(know_data['text_response'])):
        doc_list = know_data['text_response'][idx].split()
        remove_set = {'0'}
        doc_list = [i for i in doc_list if i not in remove_set]
        doc_string = ' '.join(doc_list)
        know_data.loc[idx,'text_response'] = doc_string

    # 2) text_response열에서 특수문자는 모두 제거해줍니다
    know_data["text_response"] = know_data["text_response"].str.replace(pat=r'[^\w]', repl=r" ", regex=True) # 모든 특수문자를 제거합니다
    
    # 3) text_response열에서 nan값은 공란으로 대체합니다
    know_data["text_response"] = know_data["text_response"].fillna('공란') # 생각보다 주관식을 작성하지 않은 사람들이 많습니다. 그런 유저는 공란으로 처리합니다

    # 4) text_response열에서 단어의 길이가 1인 경우 제외하고, ''으로 표현된 단어도 공란으로 대체합니다
    for idx in range(len(know_data['text_response'])):
        doc_list = know_data['text_response'][idx].split()
        new_words = ''
        for word in doc_list:
            if word.isdigit() == False: # 숫자로만 되어있는 단어가 아니어야 함
                if len(word) > 1: # 길이가 1인 경우 필터링하기
                    new_words += word
                    new_words += ' '
        if new_words[:-1] == '': # 모든 필터링을 거치고 데이터가 없을 경우도 공란으로 처리합니다
            know_data.loc[idx,'text_response'] = '공란' 
        else:
            know_data.loc[idx,'text_response'] = new_words[:-1]
    
    # 합치기 이전 문항들은 모두 드랍합니다
    know_data = know_data.drop(list(set(text_dict[year]) - set(exception_question)), axis=1)
    
    return know_data

if __name__ == '__main__':
    preprocessing_question_2017 = ['bq19_1', 'bq31']
    preprocessing_question_2018 = ['bq28_1', 'bq29']
    preprocessing_question_2019 = ['bq18_10','bq20_1']
    preprocessing_question_2020 = ['bq18_10','bq20_1']

    whole_preprocessing_question = [preprocessing_question_2017, preprocessing_question_2018, preprocessing_question_2019, preprocessing_question_2020]

    exception_question_2017 = ['bq30','bq32','bq33','bq38_1']
    exception_question_2018 = ['bq29','bq31','bq32','bq37_1']
    exception_question_2019 = ['bq22','bq23','bq27_1']
    exception_question_2020 = ['bq26_1']

    whole_exception_question = [exception_question_2017, exception_question_2018, exception_question_2019, exception_question_2020]


    # 모든 데이터 셋에 대해 전처리를 진행합니다
    idx = 0
    for year, df, pre_q, exc_q in tqdm(zip(years, know_train, whole_preprocessing_question, whole_exception_question)):
        print(year,':','train_set 진행중')
        new_data = preprocessing_text_data(year,df,pre_q,exc_q)
        
        if year == '2017':
            new_data['sim_job'] = new_data['bq30']
            new_data['bef_job'] = new_data['bq32']
            new_data['able_job'] = new_data['bq33']
            new_data['major'] = new_data['bq38_1']
            new_data = new_data.drop(exc_q,axis=1)
            
        elif year == '2018':
            new_data['sim_job'] = new_data['bq29']
            new_data['bef_job'] = new_data['bq31']
            new_data['able_job'] = new_data['bq32']
            new_data['major'] = new_data['bq37_1']
            new_data = new_data.drop(exc_q,axis=1)

        elif year == '2019':
            new_data['bef_job'] = new_data['bq22']
            new_data['able_job'] = new_data['bq23']
            new_data['major'] = new_data['bq27_1']
            new_data = new_data.drop(exc_q,axis=1)
    
        elif year == '2020':
            new_data['major'] = new_data['bq26_1']
            new_data = new_data.drop(exc_q,axis=1)
        
        new_data = pd.merge(new_data, pdf_list[idx],on='knowcode',how='left').fillna('0')
        new_data.to_csv('KNOW_{0}.csv'.format(year), index=False)
        idx += 1
        

    for year, df, pre_q, exc_q in tqdm(zip(years, know_test, whole_preprocessing_question, whole_exception_question)):
        print(year,':','test_set 진행중')
        new_data = preprocessing_text_data(year,df,pre_q,exc_q)
        
        if year == '2017':
            new_data['sim_job'] = new_data['bq30']
            new_data['bef_job'] = new_data['bq32']
            new_data['able_job'] = new_data['bq33']
            new_data['major'] = new_data['bq38_1']
            new_data = new_data.drop(exc_q,axis=1)
            
        elif year == '2018':
            new_data['sim_job'] = new_data['bq29']
            new_data['bef_job'] = new_data['bq31']
            new_data['able_job'] = new_data['bq32']
            new_data['major'] = new_data['bq37_1']
            new_data = new_data.drop(exc_q,axis=1)

        elif year == '2019':
            new_data['bef_job'] = new_data['bq22']
            new_data['able_job'] = new_data['bq23']
            new_data['major'] = new_data['bq27_1']
            new_data = new_data.drop(exc_q,axis=1)
    
        elif year == '2020':
            new_data['major'] = new_data['bq26_1']
            new_data = new_data.drop(exc_q,axis=1)
            
        new_data.to_csv('KNOW_{0}_test.csv'.format(year), index=False)