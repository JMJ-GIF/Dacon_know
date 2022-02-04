import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from glob import glob
import warnings
import re

from tqdm import tqdm
from konlpy.tag import Okt
from collections import Counter

from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfpage import PDFPage
from io import StringIO

class preprocessing():
    
    # 클래스 변수(상수)
    years = ['2017','2018','2019','2020']
    
    text_2017_question = ['bq4_1a','bq4_1b','bq4_1c','bq5_2','bq19_1','bq30','bq31','bq32','bq33','bq34','bq38_1']
    text_2018_question = ['bq4_1a','bq4_1b','bq4_1c','bq5_2','bq28_1','bq29','bq30','bq31','bq32','bq33','bq37_1']
    text_2019_question = ['bq4_1a','bq4_1b','bq4_1c','bq5_2','bq18_10','bq20_1','bq22','bq23','bq24','bq27_1']
    text_2020_question = ['bq4_1a','bq4_1b','bq4_1c','bq5_2','bq18_10','bq20_1','bq26_1']
    
    text_dict = {'2017':text_2017_question, 
                 '2018':text_2018_question, 
                 '2019':text_2019_question, 
                 '2020':text_2020_question}
    
    Stopwords_path = './stopwords.txt'
    pdf_path = './KNOW 메타데이터'
    
    selection_size = 150

    
    # pdf를 읽어오고 description에 대한 정보를 생성합니다
    
    # 생성자
    def __init__(self, train, test):
        self.know_train = train
        self.know_test = test
        
    
    # 주어진 KNOW METADATA.pdf를 활용하기 위한 함수입니다
    def _get_pdf_data(self):
        '''
        # DataFrame
        columns = [`knowcode`, `description`]

        # How to Preprocess
        1. pdf reader로 데이터를 모두 읽어와 가공함
            * print해서 손수 오류를 수정함

        
        # Limitation
        * pdf의 자체적인 오류로서, 하나의 knowcode에 2개의 직업이 매칭되는 경우가 있었음 -> __하나의 셀에 2개의 직업을 합쳐놓았음(이때 sep=','임에 유의)__
        * train data의 knowcode와 변수값.pdf 내에 있는 knowcode와 매칭이 되지 않는 경우가 있었음 -> merge시 nan값이 있을수 있으니 유의할 것
        * 위 2가지 오류가 있으니 merge시 how = (`left` or `right`) 옵션을 반드시 줄 것
        '''
        
        def convert_pdf_to_txt(file):
            ''' 
            pdf file을 python파일로 불러오고, 하나의 string으로 반환해줍니다
            
            Parameters
            file(pdf)    : 읽어올 pdf file
            
            Returns
            text(string) : 읽어온 pdf string
            
            '''
            # pdf 리소스 매니저 객체 생성
            rsrcmgr = PDFResourceManager()
            
            # 문자열 데이터를 파일처럼 처리하는 STRINGIO -> PDF 파일 내용이 여기 담긴다
            retstr = StringIO()
            codec = 'utf-8'
            laparams = LAParams()
            device = TextConverter(rsrcmgr, retstr, codec = codec, laparams = laparams)
            fp = open(file,'rb')
            interpreter = PDFPageInterpreter(rsrcmgr, device)
            password = ''
            maxpages = 0
            caching = True
            pagenos = set()
            
            for page in PDFPage.get_pages(fp,pagenos, maxpages=maxpages, password=password):
                interpreter.process_page(page)
            # text에 결과가 담김
            text = retstr.getvalue()
            
            fp.close()
            device.close()
            retstr.close()
            
            return text
        
        def extract_clean_knowcode_pdf(pdf_text, verbose=False):
            ''' 
            읽어온 pdf string을 다루기 쉽게 가공해줍니다
            pdf를 읽어오면서 생긴 자잘한 오류들은 일일이 수작업으로 바꿔줘야 합니다.
            오류를 확인하려면 verbose=True로 설정하고 어느지점에서 오류가 났는지 확인해야합니다.
                
            Parameters
            pdf_text(string)  : 읽어온 pdf string
            
            Returns
            pdf_df(DataFrame) : 로직에 따라 전처리가 완료된 pdf
            
            '''
            
            split_list = pdf_text[48:].split('\n\n')
            pdf_df = pd.DataFrame(index=range(len(split_list)))

            for idx in range(len(split_list)):
                string_list = split_list[idx].split(maxsplit=1) # 첫번째로 등장하는 공백을 기준으로 한 번만 split 한다
                if (len(string_list) == 2): # split한 리스트의 길이가 2이여야 하고
                    try: 
                        knowcode = re.match('[0-9]+', string_list[0]) # 첫 번째 원소가 숫자로만 이루어져 있어야 한다
                        pdf_df.loc[idx,'knowcode'] = knowcode.group()
                        pdf_df.loc[idx,'description'] = string_list[1]
                    except:
                        if verbose == True:
                            print('조건 에러 :',string_list) # 오류인 것은 수동으로 체크해야합니다
                else:
                    if verbose == True:
                        print('길이 에러 :',idx,'번 인덱스=>',string_list) # 오류인 것은 수동으로 체크해야합니다
            return pdf_df
        
        def fix_bug_in_pdf_df(pdf_df):
        
            ''' 
            하나의 knowcode의 2개의 직업이 매칭되는 중복오류를 수정하기 위한 함수입니다.
            오류에 해당되는 코드인 경우 한 셀에 두 가지 직업을 명시하도록 했습니다
                
            Parameters
            pdf_df(string)  : 읽어온 pdf string
            
            Returns
            modified_pdf_df(DataFrame) : 로직에 따라 전처리가 완료된 pdf
            
            '''

            pdf_df = pdf_df.drop_duplicates()
            
            value_cnt_series = pdf_df['knowcode'].value_counts()
            duplicated_knowcode = [knowcode for knowcode in value_cnt_series.index if value_cnt_series.loc[knowcode] == 2]

            error_df = pd.DataFrame()
            for knowcode in duplicated_knowcode:
                error_tmp_df = pdf_df[pdf_df['knowcode']==knowcode]
                error_df = pd.concat([error_df,error_tmp_df])

            total_indice = error_df.index
            even_indice = [even_idx for even_idx in range(0,error_df.shape[0],2)]
            one_indice = [1]*len(even_indice)
            odd_indice = [even_indice[i] + one_indice[i] for i in range(len(even_indice))]

            even_df = error_df.iloc[even_indice,:].reset_index()
            odd_df = error_df.iloc[odd_indice,:].reset_index()

            modified_df = pd.DataFrame(index = range(len(even_indice)))
            modified_df['knowcode'] = even_df['knowcode']
            modified_df['even_description'] = even_df['description']
            modified_df['odd_description'] = odd_df['description']
            modified_df['description'] = modified_df['even_description'].str.cat(modified_df['odd_description'], sep=',')

            modified_pdf_df = pdf_df.drop(total_indice)
            modified_pdf_df = pd.concat([modified_pdf_df, modified_df[['knowcode','description']]]).reset_index(drop=True)
            
            return modified_pdf_df
        
        
        raw_pdf_2017 = convert_pdf_to_txt(self.pdf_path + '/2017_변수값.pdf')
        raw_pdf_2018 = convert_pdf_to_txt(self.pdf_path + '/2018_변수값.pdf')
        raw_pdf_2019 = convert_pdf_to_txt(self.pdf_path + '/2019_변수값.pdf')
        raw_pdf_2020 = convert_pdf_to_txt(self.pdf_path + '/2020_변수값.pdf')

        ## 2017년 pdf
        pdf_2017 = extract_clean_knowcode_pdf(raw_pdf_2017, verbose=False)
        # 사소한 오류는 print로 확인하고 수동으로 조정하자    
        pdf_2017.loc[797,'knowcode'] = '08562'
        pdf_2017.loc[797,'description'] = '프린팅운영전문가'
        # dropna 및 오류인 행 drop
        pdf_2017 = pdf_2017.dropna().reset_index(drop=True)
        pdf_2017 = pdf_2017.drop(range(1170,pdf_2017.shape[0]))
        pdf_2017['knowcode'] = pdf_2017['knowcode'].map(int)

        ## 2018년 pdf
        pdf_2018 = extract_clean_knowcode_pdf(raw_pdf_2018, verbose=False)
        # dropna 및 오류인 행 drop 
        pdf_2018 = pdf_2018.dropna().reset_index(drop=True)
        pdf_2018 = pdf_2018.drop(range(1137,pdf_2018.shape[0]))
        pdf_2018['knowcode'] = pdf_2018['knowcode'].map(int)


        # 2019년 pdf
        pdf_2019 = extract_clean_knowcode_pdf(raw_pdf_2019, verbose=False)
        # 사소한 오류는 print로 확인하고 수동으로 조정하자  
        pdf_2019.loc[670,'knowcode'] = '133901'
        pdf_2019.loc[670,'description'] = 'IT 테스터 및 IT QA 전문가(SW 테스터)'
        pdf_2019.loc[685,'knowcode'] = '134303'
        pdf_2019.loc[685,'description'] = 'IT 기술지원 전문가'
        pdf_2019.loc[764,'knowcode'] = '159103'
        pdf_2019.loc[764,'description'] = '3D 프린팅모델러'
        pdf_2019.loc[887,'knowcode'] = '415504'
        pdf_2019.loc[887,'description'] = 'UX/UI 디자이너'
        # dropna 및 오류인 행 drop
        pdf_2019 = pdf_2019.dropna().reset_index(drop=True)
        pdf_2019 = pdf_2019.drop(range(1138,pdf_2019.shape[0]))
        pdf_2019['knowcode'] = pdf_2019['knowcode'].map(int)


        # 2020년 pdf
        pdf_2020 = extract_clean_knowcode_pdf(raw_pdf_2020, verbose=False)
        # 사소한 오류는 print로 확인하고 수동으로 조정하자  
        pdf_2020.loc[659,'knowcode'] = '20324'
        pdf_2020.loc[659,'description'] = 'IT 테스터 및 IT QA 전문가(SW 테스터)'
        pdf_2020.loc[667,'knowcode'] = '20296'
        pdf_2020.loc[667,'description'] = 'IT 기술지원 전문가'
        pdf_2020.loc[736,'knowcode'] = '08562'
        pdf_2020.loc[736,'description'] = '3D 프린팅모델러'
        pdf_2020.loc[845,'knowcode'] = '08558'
        pdf_2020.loc[845,'description'] = 'UX/UI 디자이너'
        # dropna 및 오류인 행 drop
        pdf_2020 = pdf_2020.dropna().reset_index(drop=True)
        pdf_2020 = pdf_2020.drop(range(1074,pdf_2020.shape[0]))
        pdf_2020['knowcode'] = pdf_2020['knowcode'].map(int)
        
        # pdf_df내 존재하는 bug를 수정합니다
        pdf_list = [pdf_2017, pdf_2018, pdf_2019, pdf_2020]
        modified_pdf_list = []
        for idx in range(4):
            modified_pdf_df = fix_bug_in_pdf_df(pdf_list[idx])
            modified_pdf_list.append(modified_pdf_df)
        
        print('description csv 생성완료')
        
        return modified_pdf_list
    
    # 주관식 문항을 control하기 위해 정의한 함수입니다
    def _preprocessing_text_data(self, year, know_data, preprocessing_question, exception_question):
        ''' 
        STEP 2 전처리에서 사용합니다
        
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
        # 불용어 목록 읽어오기(인터넷에서 일단 다운로드 받았습니다)
        stopwords = []
        f = open(self.Stopwords_path, 'rt', encoding='UTF8')
        line = f.readline()
        while line:
            line = f.readline().replace('\n',"")
            stopwords.append(line)
        f.close()
        
        # stopwords에서 공백을 제거합시다
        remove_set = {''}
        stopwords = [i for i in stopwords if i not in remove_set]
        
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
                first_col = self.text_dict[year][idx] 
                if first_col not in exception_question: # 넣고자 하는 열이 예외문항이 아닌경우에만 병합
                    text_agg_col['text_agg_col'] = know_data[first_col]
                else: # 아니면 다른 열을 탐색
                    idx += 1

        # 넣은 열을 제외한 agg_col
        remove_set = {first_col}
        target_agg_col = [i for i in self.text_dict[year] if i not in remove_set]

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
        know_data = know_data.drop(list(set(self.text_dict[year]) - set(exception_question)), axis=1)
        
        return know_data
    
    # 전처리 1단계
    def step_1_prep(self):
        '''
        1. Null 값을 모두 str '0'으로 대체하여 처리함

        2. 공백 값을 모두 str '0'으로 대체하여 처리함

        3. data type을 int / float / object로 정리함
            * int나 float으로 대답해야하는데 object로 대답한 문항들을 수기로 다 고쳤음. 
            * 오류인 데이터는 모두 '0'으로 치환함  
        '''
        #--------------------------------------------------------------------------------------------------------#
        #                                               NaN 처리                                                 #
        #--------------------------------------------------------------------------------------------------------#
        # Descripyion : Nan 값과 공백을 모두 str 0으로 처리함(문항 상에 0이 존재하지 않기에 해석 상의 오류 없음)
        
        # train,test set에 대하여 공백은 모두 str 0으로 전환, nan 값을 0으로 대체합시다
        for train,test in zip(self.know_train, self.know_test):
            
            train.fillna('0',inplace=True)
            for col in train.columns:
                train[col].replace(' ', '0', inplace=True)
                
            test.fillna('0',inplace=True)
            for col in test.columns:
                test[col].replace(' ', '0', inplace=True)

        #--------------------------------------------------------------------------------------------------------#
        #                                            Data Type 처리                                              #
        #--------------------------------------------------------------------------------------------------------#
        # Descripyion : 숫자로 질문했는데 주관식으로 답변한 응답자가 존재하여 데이터 타입 변경이 불가능한 문항이 연도별로 존재함
        #               해당 질문을 드랍하고 무결한 데이터타입을 가진 csv를 생성함
        #               잘못 대답한 칼럼에 대해서 일일이 수기로 변경함
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
        for train,test in zip(self.know_train, self.know_test):
            for col in train.columns:
                try:
                    train[col] = train[col].map(int) 
                except:
                    train[col] = train[col].map(str)
                    
            for col in test.columns:
                try:
                    test[col] = test[col].map(int) 
                except:
                    test[col] = test[col].map(str)
        
        # 연도별로 일일이 수기로 오류 문항에 대해 0으로 바꾸어줍니다
        for i,(train,test) in enumerate(zip(self.know_train, self.know_test)):
            # ---------------------------------------------2017년----------------------------------------------------#
            # 문제되는 문항이 없음
            if i == 0:
                pass
            # ---------------------------------------------2018년----------------------------------------------------#
            # train : ['bq4','bq5_1', 'bq28', 'bq34', 'bq35','bq36', 'bq37', 'bq40' ] 
            # test : ['bq4', 'bq5_1', 'bq28', bq35', 'bq37']
            elif i == 1:
                # bq4
                remove_set = {'1','2','0'}
                new_words_list = [i for i in train['bq4'].unique() if i not in remove_set]

                for word in new_words_list:
                    train.loc[train[train['bq4']==word].index,'bq4'] = 0 
                train['bq4'] = train['bq4'].map(float) 
                
                remove_set = {'1','2','0'}
                new_words_list = [i for i in test['bq4'].unique() if i not in remove_set]

                for word in new_words_list:
                    test.loc[test[test['bq4']==word].index,'bq4'] = 0 
                test['bq4'] = test['bq4'].map(float) 
                
                # bq5_1
                remove_set = {'1','2','3','4','5','6','0'}
                new_words_list = [i for i in train['bq5_1'].unique() if i not in remove_set]

                for word in new_words_list:
                    train.loc[train[train['bq5_1']==word].index,'bq5_1'] = 0 
                train['bq5_1'] = train['bq5_1'].map(float)
                
                remove_set = {'1','2','3','4','5','0'}
                new_words_list = [i for i in test['bq5_1'].unique() if i not in remove_set]

                for word in new_words_list:
                    test.loc[test[test['bq5_1']==word].index,'bq5_1'] = 0 
                test['bq5_1'] = test['bq5_1'].map(float)

                
                # bq28
                remove_set = {'1','2','3','4','5'}
                new_words_list = [i for i in train['bq28'].unique() if i not in remove_set]

                for word in new_words_list:
                    train.loc[train[train['bq28']==word].index,'bq28'] = 0 
                train['bq28'] = train['bq28'].map(float)
                
                remove_set = {'1','2','3','4','5'}
                new_words_list = [i for i in test['bq28'].unique() if i not in remove_set]

                for word in new_words_list:
                    test.loc[test[test['bq28']==word].index,'bq28'] = 0 
                test['bq28'] = test['bq28'].map(float) 
                                
                # bq36
                train.loc[train[train['bq36']=='디자이너'].index,'bq36'] = 0 
                train['bq36'] = train['bq36'].map(float) 
                
                # bq34
                train.loc[train[train['bq34']=='자동차과'].index,'bq34'] = 0
                train.loc[train[train['bq34']==' 포토샵'].index,'bq34'] = 0 
                train['bq34'] = train['bq34'].map(float)
                
                # bq35
                train.loc[train[train['bq35']==' 라이트룸 등의 컴퓨터 프로그램"'].index,'bq35'] = 0 
                train['bq35'] = train['bq35'].map(float) 
                
                test.loc[test[test['bq35']=='문예창작'].index,'bq35'] = 0 
                test['bq35'] = test['bq35'].map(float) 
                
                # bq37
                remove_set = {'1','2','3','4','5','6'}
                new_words_list = [i for i in train['bq37'].unique() if i not in remove_set]

                for word in new_words_list:
                    train.loc[train[train['bq37']==word].index,'bq37'] = 0 
                train['bq37'] = train['bq37'].map(float)
                
                # bq37
                remove_set = {'1','2','3','4','5','6','0'}
                new_words_list = [i for i in test['bq37'].unique() if i not in remove_set]

                for word in new_words_list:
                    test.loc[test[test['bq37']==word].index,'bq37'] = 0 
                test['bq37'] = test['bq37'].map(float) 
                
                # bq40
                train.loc[train[train['bq40']=='지식재산학'].index,'bq40'] = 0 
                train['bq40'] = train['bq40'].map(float) 
            # ---------------------------------------------2019년----------------------------------------------------#
            # train : ['bq6','bq7','bq8_3','bq9','bq10', 'bq11', 'bq12_4','bq20','bq21_3','bq25','bq27']
            # test : ['bq19', 'bq20', 'bq21_3', 'bq27']
            elif i == 2:
                # bq6
                remove_set = {'1','2','3','4','5','6','7','0'}
                new_words_list = [i for i in train['bq6'].unique() if i not in remove_set]

                for word in new_words_list:
                    train.loc[train[train['bq6']==word].index,'bq6'] = 0 
                train['bq6'] = train['bq6'].map(float) 
                
                # bq7
                remove_set = {'1','2','3','4','5','6','7'}
                new_words_list = [i for i in train['bq7'].unique() if i not in remove_set]

                for word in new_words_list:
                    train.loc[train[train['bq7']==word].index,'bq7'] = 0 
                train['bq7'] = train['bq7'].map(float) 
                
                # bq8_3
                train.loc[train[train['bq8_3']=='정보·기록학'].index,'bq8_3'] = 0 
                train['bq8_3'] = train['bq8_3'].map(float)
                
                # bq9
                train.loc[train[train['bq9']=='디자이너'].index,'bq9'] = 0 
                train['bq9'] = train['bq9'].map(float)  
                
                # bq10
                remove_set = {'1','2','3','4','5'}
                new_words_list = [i for i in train['bq10'].unique() if i not in remove_set]

                for word in new_words_list:
                    train.loc[train[train['bq10']==word].index,'bq10'] = 0 
                train['bq10'] = train['bq10'].map(float)
                
                # bq11
                remove_set = {'1','2','3','4','5','0'}
                new_words_list = [i for i in train['bq11'].unique() if i not in remove_set]

                for word in new_words_list:
                    train.loc[train[train['bq11']==word].index,'bq11'] = 0 
                train['bq11'] = train['bq11'].map(float)
                
                # bq12_4
                remove_set = {'1','2','3','4','5','9'}
                new_words_list = [i for i in train['bq12_4'].unique() if i not in remove_set]

                for word in new_words_list:
                    train.loc[train[train['bq12_4']==word].index,'bq12_4'] = 0 
                train['bq12_4'] = train['bq12_4'].map(float) 
                
                # bq19
                remove_set = {'1','2','3','4','5'}
                new_words_list = [i for i in test['bq19'].unique() if i not in remove_set]

                for word in new_words_list:
                    test.loc[test[test['bq19']==word].index,'bq19'] = 0 
                test['bq19'] = test['bq19'].map(float) 
                
                # bq20
                remove_set = {'1','2','3','4','5','0'}
                new_words_list = [i for i in train['bq20'].unique() if i not in remove_set]

                for word in new_words_list:
                    train.loc[train[train['bq20']==word].index,'bq20'] = 0 
                train['bq20'] = train['bq20'].map(float)
                
                remove_set = {'1','2','3','4','5'}
                new_words_list = [i for i in test['bq20'].unique() if i not in remove_set]

                for word in new_words_list:
                    test.loc[test[test['bq20']==word].index,'bq20'] = 0 
                test['bq20'] = test['bq20'].map(float) 
                                
                # bq21_3
                remove_set = {'1','2','3','4','5','0'}
                new_words_list = [i for i in train['bq21_3'].unique() if i not in remove_set]

                for word in new_words_list:
                    train.loc[train[train['bq21_3']==word].index,'bq21_3'] = 0 
                train['bq21_3'] = train['bq21_3'].map(float)
                
                test.loc[test[test['bq21_3']=='사회학/광고홍보학'].index,'bq21_3'] = 0 
                test['bq21_3'] = test['bq21_3'].map(float) 
                
                # bq25
                train.loc[train[train['bq25']==' 일러스트레이터"'].index,'bq25'] = 0 
                train['bq25'] = train['bq25'].map(float)
                
                # bq27
                remove_set = {'1','2','3','4','5','0'}
                new_words_list = [i for i in train['bq27'].unique() if i not in remove_set]

                for word in new_words_list:
                    train.loc[train[train['bq27']==word].index,'bq27'] = 0 
                train['bq27'] = train['bq27'].map(float) 
                
                remove_set = {'1','2','3','4','5','6'}
                new_words_list = [i for i in test['bq27'].unique() if i not in remove_set]

                for word in new_words_list:
                    test.loc[test[test['bq27']==word].index,'bq27'] = 0 
                test['bq27'] = test['bq27'].map(float) 
            # ---------------------------------------------2020년----------------------------------------------------#
            # 문제되는 문항이 없음
            else:
                pass
        
        print('전처리 1단계 완료!')
        
        return self.know_train, self.know_test
    
    # 전처리 2_1단계
    def step_2_1_prep(self):
        '''
        0. STEP 1 전처리를 먼저 시행함
        
        1. 연도별로 서술형이나 약술형으로 대답해야 하는 문항들을 추리고, konlpy를 통해 명사추출을 진행함  
            * 서술형, 약술형으로 명사추출을 진행한 연도별 문항
                - 2017 : ['bq19_1', 'bq31']
                - 2018 : ['bq28_1', 'bq29']
                - 2019 : ['bq18_10','bq20_1']
                - 2020 : ['bq18_10','bq20_1'] 

        2. 단답형 + 서술형 + 약술형 문항을 모두 "text_response"라는 하나의 열로 합침  
            * text_response내에 존재하는 '0'은 모두 삭제함

        3. text_response 열에서 nan에 해당하는 값들은 모두 "공란"으로 대체함  

        4. "major"에 대항하는 문항은 1.2.3.과정을 생략함  

        __때문에 결과 데이터 프레임은 obj에 해당하는 칼럼이 [major, text_response]로 단 2개이고, 나머지는 모두 int/float임.__
        '''
        
        # 전처리 1단계를 먼저 시행함
        self.know_train, self.know_test = self.step_1_prep()
        
        # konlpy를 통해 명사만 뽑아낼 문항들을 미리 정합니다
        preprocessing_question_2017 = ['bq19_1', 'bq31']
        preprocessing_question_2018 = ['bq28_1', 'bq29']
        preprocessing_question_2019 = ['bq18_10','bq20_1']
        preprocessing_question_2020 = ['bq18_10','bq20_1']

        whole_preprocessing_question = [preprocessing_question_2017, preprocessing_question_2018,\
                                        preprocessing_question_2019, preprocessing_question_2020]

        # 칼럼 병합에서 예외로 둘 칼럼들을 정의해줍니다
        exception_question_2017 = ['bq38_1']
        exception_question_2018 = ['bq37_1']
        exception_question_2019 = ['bq27_1']
        exception_question_2020 = ['bq26_1']

        whole_exception_question = [exception_question_2017, exception_question_2018, exception_question_2019, exception_question_2020]

        # 모든 데이터 셋에 대해 전처리를 진행합니다
        for i, (year, train, test, pre_q, exc_q) in tqdm(enumerate(zip(self.years, self.know_train, self.know_test,\
                                                                whole_preprocessing_question, whole_exception_question))):
            new_train = self._preprocessing_text_data(year,train,pre_q,exc_q)
            new_test = self._preprocessing_text_data(year,test,pre_q,exc_q)
            
            # exc_q열의 이름을 major로 변환해줍니다 
            new_train['major'] = new_train[exc_q]
            new_train = new_train.drop(exc_q,axis=1)
            
            new_test['major'] = new_test[exc_q]
            new_test = new_test.drop(exc_q,axis=1)
            
            # 저장해주기
            self.know_train[i] = new_train
            self.know_test[i] = new_test
        
        print('전처리 2_1단계 완료!')
        
        return self.know_train, self.know_test
    
    def step_2_2_prep(self):
        '''
        0. STEP 1 전처리를 먼저 시행함
        
        1. 연도별로 서술형이나 약술형으로 대답해야 하는 문항들을 추리고, konlpy를 통해 명사추출을 진행함  
            * 서술형, 약술형으로 명사추출을 진행한 연도별 문항
                - 2017 : ['bq19_1', 'bq31']
                - 2018 : ['bq28_1', 'bq29']
                - 2019 : ['bq18_10','bq20_1']
                - 2020 : ['bq18_10','bq20_1'] 

        2. 단답형 + 서술형 + 약술형 문항을 모두 "text_response"라는 하나의 열로 합침  
            * text_response내에 존재하는 '0'은 모두 삭제함

        3. text_response 열에서 nan에 해당하는 값들은 모두 "공란"으로 대체함  

        4. 아래 문항들은 1.2.3. 과정을 생략함.
            - 2017 : ['bq30','bq32','bq33','bq38_1'] = (rename) ['sim_job','bef_job','able_job','major']
            - 2018 : ['bq29','bq31','bq32','bq37_1'] = (rename) ['sim_job','bef_job','able_job','major']
            - 2019 : ['bq22','bq23','bq27_1'] = (rename) ['bef_job','able_job','major']
            - 2020 : ['bq26_1'] = (rename) ['major'] 

        5. knowcode에 대한 설명인 description 열을 추가함(data_pdf_description의 자료들을 merge해줌)
            - data_pdf_description 내 파일과의 병합과정에서 발생한 nan값은 모두 '0'으로 채움


        __때문에 결과 데이터 프레임은 obj에 해당하는 칼럼이 [4에 명시한 연도별 칼럼들, text_response, description]로 3 ~ 6개이고, 나머지는 모두 int/float임.__
        '''
        
        # 전처리 1단계를 먼저 시행함
        self.know_train, self.know_test = self.step_1_prep()
        
        # konlpy를 통해 명사만 뽑아낼 문항들을 미리 정합니다
        preprocessing_question_2017 = ['bq19_1', 'bq31']
        preprocessing_question_2018 = ['bq28_1', 'bq29']
        preprocessing_question_2019 = ['bq18_10','bq20_1']
        preprocessing_question_2020 = ['bq18_10','bq20_1']

        whole_preprocessing_question = [preprocessing_question_2017, preprocessing_question_2018,\
                                        preprocessing_question_2019, preprocessing_question_2020]
        
        # 칼럼 병합에서 예외로 둘 칼럼들을 정의해줍니다
        exception_question_2017 = ['bq30','bq32','bq33','bq38_1']
        exception_question_2018 = ['bq29','bq31','bq32','bq37_1']
        exception_question_2019 = ['bq22','bq23','bq27_1']
        exception_question_2020 = ['bq26_1']

        whole_exception_question = [exception_question_2017, exception_question_2018, exception_question_2019, exception_question_2020]

        pdf_list = self._get_pdf_data()    # pdf_list를 생성해줍니다
        
        # 모든 데이터 셋에 대해 전처리를 진행합니다
        for i, (year, train, test, pre_q, exc_q) in tqdm(enumerate(zip(self.years, self.know_train, self.know_test,\
                                                                whole_preprocessing_question, whole_exception_question))):
            new_train = self._preprocessing_text_data(year,train,pre_q,exc_q)
            new_test = self._preprocessing_text_data(year,test,pre_q,exc_q)
            
            if year == '2017':
                new_train['sim_job'] = new_train['bq30']
                new_train['bef_job'] = new_train['bq32']
                new_train['able_job'] = new_train['bq33']
                new_train['major'] = new_train['bq38_1']
                new_train = new_train.drop(exc_q,axis=1)
                
                new_test['sim_job'] = new_test['bq30']
                new_test['bef_job'] = new_test['bq32']
                new_test['able_job'] = new_test['bq33']
                new_test['major'] = new_test['bq38_1']
                new_test = new_test.drop(exc_q,axis=1)
                
            elif year == '2018':
                new_train['sim_job'] = new_train['bq29']
                new_train['bef_job'] = new_train['bq31']
                new_train['able_job'] = new_train['bq32']
                new_train['major'] = new_train['bq37_1']
                new_train = new_train.drop(exc_q,axis=1)
                
                new_test['sim_job'] = new_test['bq29']
                new_test['bef_job'] = new_test['bq31']
                new_test['able_job'] = new_test['bq32']
                new_test['major'] = new_test['bq37_1']
                new_test = new_test.drop(exc_q,axis=1)
            
            elif year == '2019':
                new_train['bef_job'] = new_train['bq22']
                new_train['able_job'] = new_train['bq23']
                new_train['major'] = new_train['bq27_1']
                new_train = new_train.drop(exc_q,axis=1)
                
                new_test['bef_job'] = new_test['bq22']
                new_test['able_job'] = new_test['bq23']
                new_test['major'] = new_test['bq27_1']
                new_test = new_test.drop(exc_q,axis=1)
                
            elif year == '2020':
                new_train['major'] = new_train['bq26_1']
                new_train = new_train.drop(exc_q,axis=1)
                
                new_test['major'] = new_test['bq26_1']
                new_test = new_test.drop(exc_q,axis=1)
            
            # 저장해주기
            self.know_train[i] = new_train
            self.know_test[i] = new_test
            
            # description 열을 추가해줍니다
            self.know_train[i] = pd.merge(self.know_train[i], pdf_list[i],on='knowcode',how='left').fillna('0')
            
        print('전처리 2_2단계 완료!')
        
        return self.know_train, self.know_test 

    def step_3_1_prep(self, branch='2_1'):
        '''
        0. STEP 1 전처리를 먼저 시행함 / 이후 STEP 2_1을 시행함 
        
        1. 2017년 text_response들의 단어들 중 자주 등장하는 단어들을 column으로 사용
            * stop_words (ignore)  
                ```없다 증가 없음 감소 때문 수요 대한 모름 도구 사람 산업 관련 유지 업무 일자리 관심 계속 인력 발달 발전 필요 직업 분야 대체 환경 추세 사회 확대```
                
        2. 모든 단어를 다 column으로 쓰면 너무 많으므로, 상위 150개 단어에서 자르고, stop_words는 제외함
        
        3. text_response 중 '없음', '없다', '모름'의 개수를 `ubda_cnt`로 둠
        '''
        if branch == '2_1':
            # 전처리 2_1단계를 시행함
            self.know_train, self.know_test = self.step_2_1_prep()
        elif branch == '2_2':
            raise '잘못된 branch를 사용하였습니다'
        else:
            print('올바른 이전 branch를 입력해주세요 : [branch = "2_1" or branch = "2_2"]')
        
        # 주관식 문자열 내에 존재하는 word들을 추출하고, count를 세봅니다
        word_dict = {}
        for sentence in self.know_train[0]['text_response']:
            words = sentence.split()
            for word in words:
                if word in word_dict.keys():
                    word_dict[word] += 1
                else:
                    word_dict[word] = 1
        
        stop_words = '''
                    없다 증가 없음 감소 때문 수요 대한 모름 도구 사람 산업 관련 유지 업무 일자리 관심
                    계속 인력 발달 발전 필요 직업 분야 대체 환경 추세 사회 확대
                    '''.split()
        
        # 모든 단어를 다 column으로 쓰면 너무 많으므로, 상위 150개 단어에서 자르고, stop_words는 제외함
        # selected_words를 만들때 2017년 train data를 기준으로 만들었음
        selection_size = 150
        selected_words = [x[0] for x in sorted(word_dict.items(), reverse=True, key=lambda x:x[1])[:selection_size]]
        selected_words = list(set(selected_words) - set(stop_words))
        
        def add_subj_cols(know_df, selected_words, train):
            
            text_col_dict = {}
            for word in selected_words:
                text_col_dict[word] = []

            for text in know_df['text_response']: # O(rows)
                words = text.split()
                for selected_word in selected_words: # O(n) n = 150
                    if selected_word in words:
                        text_col_dict[selected_word].append(1)
                    else:
                        text_col_dict[selected_word].append(0)

            text_df = pd.DataFrame(text_col_dict)
            text_df['idx'] = know_df['idx']

            result_df = pd.merge(know_df, text_df, 
                                left_on='idx', right_on='idx')
            
            ubda_cnt_lst = []

            # ubda_cnt역시 2017년을 기준으로만 만들었음
            for text in self.know_train[0]['text_response']: # O(rows) rows = 9486
                words = text.split()
                counter = Counter(words)
                ubda_cnt_lst.append(counter['없다'] + counter['없음'] + counter['모름'])

            result_df['ubda_cnt'] = pd.Series(ubda_cnt_lst)
            
            new_cols = result_df.columns.tolist()
            
            remove_set = {'text_response','knowcode'}
            new_cols = [i for i in new_cols if i not in remove_set]
            
            if train == True:
                result_df = result_df[new_cols + ['knowcode']]
            else:
                result_df = result_df[new_cols]
            
            return result_df
        
        test_list = []
        train_list = []
        for i in range(4):
            test_list.append(add_subj_cols(self.know_test[i], selected_words, train = False))
            train_list.append(add_subj_cols(self.know_train[i], selected_words, train = True))
        
        self.know_train = train_list
        self.know_test = test_list

        
        print('전처리 3_1단계 완료!')
        
        return self.know_train, self.know_test
                
        
    
    def step_3_2_prep(self, branch='2_2'):
        '''
        0. STEP 1 전처리를 먼저 시행함 / 이후 STEP 2_2를 시행함
        
        1. half_cols에 대해서 numercial한 대답은 그대로 두고, 7번이나 9번같이 categorical하게 답변한 경우 mark라는 열을 추가해서 따로 구분함. 그로인해 (7번이나 9번으로 대답하여 발생한) categorical nan값은 median으로 채워줌

            - half_cols_mark_2017 = ['bq6_mark', 'bq12_2_mark', 'bq12_3_mark', 'bq12_4_mark',]  
            - half_cols_mark_2018 = ['bq6_mark', 'bq12_2_mark', 'bq12_3_mark', 'bq12_4_mark',]  
            - half_cols_mark_2019 = ['bq6_mark', 'bq12_2_mark', 'bq12_3_mark', 'bq12_4_mark',]  
            - half_cols_mark_2020 = ['bq6_mark', 'bq12_2_mark', 'bq12_3_mark', 'bq12_4_mark',]

        2. 18년 한정으로 잘못 원핫인코딩된 칼럼을 고쳐줌

            - know_test[1]['bq22'] =  know_test[1]['bq221'] + know_test[1]['bq222'] + know_test[1]['bq223']
            - know_test[1]['bq23'] =  know_test[1]['bq231'] + know_test[1]['bq232'] + know_test[1]['bq233']+\
                                        know_test[1]['bq234'] + know_test[1]['bq235']
            - know_test[1]['bq24'] =  know_test[1]['bq241'] + know_test[1]['bq242'] + know_test[1]['bq243']+\
                                        know_test[1]['bq244'] + know_test[1]['bq245']
        '''
        
        if branch == '2_1':
            raise '잘못된 branch를 사용하였습니다'
        elif branch == '2_2':
            # 전처리 2_2단계를 시행함
            self.know_train, self.know_test = self.step_2_2_prep()
        else:
            print('올바른 이전 branch를 입력해주세요 : [branch = "2_1" or branch = "2_2"]')
        
        def cols_correction(data):
            data['bq6_mark']= 0
            data['bq6_mark'].loc[data['bq6'] == 7] = 1
            data['bq12_2_mark']= 0
            data['bq12_2_mark'].loc[data['bq12_2'] == 9] = 1
            data['bq12_3_mark']= 0
            data['bq12_3_mark'].loc[data['bq12_3'] == 9] = 1
            data['bq12_4_mark']= 0
            data['bq12_4_mark'].loc[data['bq12_4'] == 9] = 1

            data.loc[data['bq6'] == 7, 'bq6'] = 3 # median filling
            data.loc[data['bq12_2'] == 9, 'bq12_2'] = 3 # median filling
            data.loc[data['bq12_3'] == 9, 'bq12_3'] = 3 # median filling
            data.loc[data['bq12_4'] == 9, 'bq12_4'] = 3 # median filling
            
            return data
            
        # 수정해줍시다
        for i, (train, test) in enumerate(zip(self.know_train, self.know_test)):
            self.know_train[i] = cols_correction(train)
            self.know_test[i] = cols_correction(test)
            if i == 1:
                error_cols = ['bq221','bq222','bq223','bq231','bq232','bq233','bq234','bq235','bq241','bq242','bq243','bq244','bq245']
                
                for error_col in error_cols:
                    self.know_train[i][error_col] = self.know_train[i][error_col].map(float) 
                    self.know_test[i][error_col] = self.know_test[i][error_col].map(float)    
                         
                self.know_train[i]['bq22'] = self.know_train[i]['bq221'] + self.know_train[i]['bq222'] + self.know_train[i]['bq223']
                self.know_train[i]['bq23'] = self.know_train[i]['bq231'] + self.know_train[i]['bq232'] + self.know_train[i]['bq233']+\
                                        self.know_train[i]['bq234'] + self.know_train[i]['bq235']
                self.know_train[i]['bq24'] =  self.know_train[i]['bq241'] + self.know_train[i]['bq242'] + self.know_train[i]['bq243']+\
                                        self.know_train[i]['bq244'] + self.know_train[i]['bq245']
                self.know_train[i].drop(['bq221',
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
                
                self.know_test[i]['bq22'] = self.know_test[i]['bq221'] + self.know_test[i]['bq222'] + self.know_test[i]['bq223']
                self.know_test[i]['bq23'] = self.know_test[i]['bq231'] + self.know_test[i]['bq232'] + self.know_test[i]['bq233']+\
                                        self.know_test[i]['bq234'] + self.know_test[i]['bq235']
                self.know_test[i]['bq24'] =  self.know_test[i]['bq241'] + self.know_test[i]['bq242'] + self.know_test[i]['bq243']+\
                                        self.know_test[i]['bq244'] + self.know_test[i]['bq245']
                self.know_test[i].drop(['bq221',
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
        
        
        
        # int type으로 바꾸는 것을 시도해보고 안된다면 str로 취급하여 정리해봅시다 
        for train in self.know_train:
            for col in train.columns:
                try:
                    train[col] = train[col].map(float) 
                except:
                    train[col] = train[col].map(str)
        for test in self.know_test:
            for col in test.columns:
                try:
                    test[col] = test[col].map(float) 
                except:
                    test[col] = test[col].map(str)
        
        print('전처리 3_2단계 완료!')
        
        return self.know_train, self.know_test

        
    def step_4_prep(self, branch='2_2'):
        '''
        0. STEP 1 전처리를 먼저 시행함 /  STEP 2_2를 시행함 / STEP 3_2을 시행함
        
        1. 각 연도별로 설문지에서 건너뛰어도 좋다고 명시한 문항들을 수기로 기록함

            - skip_cols_2017 = ['aq1_2', 'aq2_2', 'aq3_2', 'aq4_2', 'aq5_2', 'aq6_2', 'aq7_2', 'aq8_2', 'aq9_2', 'aq10_2'
                            ,'aq11_2', 'aq12_2', 'aq13_2', 'aq14_2', 'aq15_2', 'aq16_2', 'aq17_2', 'aq18_2', 'aq19_2', 'aq20_2'
                            ,'aq21_2', 'aq22_2', 'aq23_2', 'aq24_2', 'aq25_2', 'aq26_2', 'aq27_2', 'aq28_2', 'aq29_2', 'aq30_2'
                            ,'aq31_2', 'aq32_2', 'aq33_2', 'aq34_2', 'aq35_2', 'aq36_2', 'aq37_2', 'aq38_2', 'aq39_2', 'aq40_2'
                            ,'aq41_2','bq5_1', 'bq40','bq41_1', 'bq41_2', 'bq41_3']  
            - skip_cols_2018 = ['bq5_1','bq25_1','bq39','bq40','bq41_1','bq41_2','bq41_3']  
            - skip_cols_2019 = ['kq1_2', 'kq2_2', 'kq3_2', 'kq4_2', 'kq5_2', 'kq6_2', 'kq7_2', 'kq8_2', 'kq9_2', 'kq10_2'
                        ,'kq11_2', 'kq12_2', 'kq13_2', 'kq14_2', 'kq15_2','kq16_2', 'kq17_2', 'kq18_2', 'kq19_2','kq20_2'
                        ,'kq21_2', 'kq22_2', 'kq23_2', 'kq24_2', 'kq25_2','kq26_2', 'kq27_2' 'kq28_2', 'kq29_2', 'kq30_2'
                        ,'kq31_2', 'kq32_2','kq33_2','bq5_1','bq29','bq30','bq31_1','bq31_2','bq31_3']  
            - skip_cols_2020 = ['saq1_2', 'saq2_2', 'saq3_2', 'saq4_2', 'saq5_2','saq6_2', 'saq7_2', 'saq8_2', 'saq9_2', 'saq10_2'
                        ,'saq11_2', 'saq12_2', 'saq13_2', 'saq14_2','saq15_2', 'saq16_2', 'saq17_2', 'saq18_2', 'saq19_2'
                        ,'saq20_2', 'saq21_2', 'saq22_2', 'saq23_2', 'saq24_2', 'saq25_2', 'saq26_2', 'saq27_2', 'saq28_2', 'saq29_2'
                        ,'saq30_2', 'saq31_2', 'saq32_2', 'saq33_2', 'saq34_2', 'saq35_2','saq36_2', 'saq37_2', 'saq38_2' 
                        , 'saq39_2', 'saq40_2', 'saq41_2', 'saq42_2',  'saq43_2', 'saq44_2' 
                        ,'bq5_1','bq28','bq29','bq30_1','bq30_2','bq30_3'] 
                                
        2. 1.에 속하지 않는 문항들에 대해서, numeric_pure_cols에 속하면 mean으로 imputation을 진행하였음. 아니라면 mode로 imputation을 진행하였음.
        '''
        if branch == '2_1':
            raise '해당 step은 branch="2_1"을 지원하지 않습니다'
        
        elif branch == '2_2':
            # 전처리 3_2단계를 시행함
            self.know_train, self.know_test = self.step_3_2_prep(branch = "2_2")
        else:
            print('올바른 이전 branch를 입력해주세요 : [branch = "2_1" or branch = "2_2"]')
        
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
        
        # 연도마다 imputation을 진행할 column들을 정의합니다
        imputation_cols_dict = {}
        half_cols_mark = ['bq6_mark', 'bq12_2_mark', 'bq12_3_mark', 'bq12_4_mark',]
        exception_cols = ['idx','knowcode','text_response','description'] + half_cols_mark
        for year, df in zip(self.years, self.know_train):
            imputation_cols_dict[year] = []
            for col in df.columns:
                if col not in exception_cols:
                    imputation_cols_dict[year].append(col)
        
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
        error_cols_2017 = [col for col in imputation_cols_dict['2017'] if col not in skip_cols_2017 + skip_txt_col]
        
        ## 2018
        # 설문지에서 건너뛰어도 된다고 말한 문항
        skip_cols_2018 = ['bq5_1','bq25_1','bq39','bq40','bq41_1','bq41_2','bq41_3']
        skip_txt_col = ['major','sim_job','bef_job','able_job']
        numeric_pure_cols_2018 = ['bq21', 'bq36', 'bq40', 'bq41_1', 'bq41_2', 'bq41_3', ]

        # 0이 존재한다면 설문자의 오류로 발생한 문항
        error_cols_2018 = [col for col in imputation_cols_dict['2018'] if col not in skip_cols_2018 + skip_txt_col]
        
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
        error_cols_2019 = [col for col in imputation_cols_dict['2019'] if col not in skip_cols_2019 + skip_txt_col]
        
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
        error_cols_2020 = [col for col in imputation_cols_dict['2020'] if col not in skip_cols_2020 + skip_txt_col]
        
        ## impuation 진행하기
        error_cols_list = [error_cols_2017, error_cols_2018, error_cols_2019, error_cols_2020]
        numeric_pure_cols_list = [numeric_pure_cols_2017, numeric_pure_cols_2018, numeric_pure_cols_2019,numeric_pure_cols_2020]
        
        train_list = []
        test_list = []
        
        for idx in range(4):
            train, test = data_imputation(error_cols_list[idx], numeric_pure_cols_list[idx], 
                                                              self.know_train[idx], self.know_test[idx])
            train_list.append(train)
            test_list.append(test)
            
        self.know_train = train_list
        self.know_test = test_list
        
        print('전처리 4단계 완료!')
        
        return self.know_train, self.know_test
    
    def step_5_1_prep(self, branch='2_2'):
        '''
        0. STEP 1 전처리를 먼저 시행함 / 이후 STEP 2_2, STEP 3_2, STEP 4을 시행함 
        
        1. 연도별 text_response들의 단어들 중 자주 등장하는 단어들을 column으로 사용
            * stop_words (ignore): 각 연도별로 모두 합집합한 set  
            2017  
                ```없다 증가 없음 감소 때문 수요 대한 모름 도구 사람 산업 관련 유지 업무 일자리 관심 계속 인력 발달 발전 필요 직업 분야 대체 환경 추세 사회 확대```  
            2018  
                ```없음 증가 감소 때문 사람 없다 대한 일자리 관련 업무 필요 예상 생각 모름 점점 갈수록 이기 크게 시대작업 능력 습득 부분```  
            2019  
                ```증가 도입 공란 대한 대체 때문 일자리 관련 필요 사용 발전 직업 생각 예상 분야 활용 부분 변경 상품 이용 이해 추가 증대 추세 실무 점점 진행 사회 업종 숙지 기존 위해 경우 대신 과정 조직 발생 이기 갈수록 일이 근무```  
            2020  
                ```업무 때문 대한 상태 관심 계속 관련 예상 상황 분야 필요 현재 지금 선호 점점 고용 한정 대신 시대 크게 부분 다양 중요성 문제```  
        2. 모든 단어를 다 column으로 쓰면 너무 많으므로, 상위 150개 단어에서 자르고, stop_words는 제외함
        3. text_response 중 '없음', '없다', '모름'의 개수를 `ubda_cnt`로 둠
        '''
        
        if branch == '2_1':
            raise '잘못된 branch를 사용하였습니다'
        elif branch == '2_2':
            # 전처리 4단계를 시행함
            self.know_train, self.know_test = self.step_4_prep(branch = "2_2")
        else:
            print('올바른 이전 branch를 입력해주세요 : [branch = "2_1" or branch = "2_2"]')
        
        # 연도별 words를 수집합니다
        word_dict_year = {}
        for i in range(4):
            word_dict = {}

            for sentence in self.know_train[i]['text_response']:
                words = sentence.split()
                for word in words:
                    if word in word_dict.keys():
                        word_dict[word] += 1
                    else:
                        word_dict[word] = 1

            word_dict_year[i] = word_dict
        
        # manually하게 stopwords를 찾아봅시다
        stop_words = set()
        
        # 2017
        stop_words.update(
                        set('''
                            없다 증가 없음 감소 때문 수요 대한 모름 도구 사람 산업 관련 유지 업무 일자리 관심
                            계속 인력 발달 발전 필요 직업 분야 대체 환경 추세 사회 확대 상태 이용 것이므 사용
                            '''.split())
                            )
        # 2018
        stop_words.update(
                        set('''
                            없음 증가 감소 때문 사람 없다 대한 일자리 관련 업무 필요 예상 생각 모름 점점
                            갈수록 이기 크게 시대작업 능력 습득 부분
                            '''.split())
                            )
        # 2019
        stop_words.update(
                        set('''
                            증가 도입 공란 대한 대체 때문 일자리 관련 필요 사용 발전 직업 생각 예상
                            분야 활용 부분 변경 상품 이용 이해 추가 증대 추세 실무 점점 진행 사회
                            업종 숙지 기존 위해 경우 대신 과정 조직 발생 이기 갈수록 일이 근무
                            '''.split())
                            )
        # 2020
        stop_words.update(
                        set('''
                            업무 때문 대한 상태 관심 계속 관련 예상 상황 분야 필요 현재 지금 선호 점점 고용 한정 대신
                            시대 크게 부분 다양 중요성 문제
                            '''.split())
                            )
        
        
        def add_subj_cols(know_df, word_dict, train):
            
            selected_words = [x[0] for x in sorted(word_dict.items(), reverse=True, key=lambda x:x[1])[:self.selection_size]]
            selected_words = list(set(selected_words) - stop_words)
            
            text_col_dict = {}
            for word in selected_words:
                text_col_dict[word] = []

            for text in know_df['text_response']: # O(rows)
                words = text.split()
                for selected_word in selected_words: # O(n) n = 150
                    if selected_word in words:
                        text_col_dict[selected_word].append(1)
                    else:
                        text_col_dict[selected_word].append(0)

            text_df = pd.DataFrame(text_col_dict)
            text_df['idx'] = know_df['idx']

            result_df = pd.merge(know_df, text_df, 
                                left_on='idx', right_on='idx')
            
            ubda_cnt_dict = []

            for text in know_df['text_response']: # O(rows) rows = 9486
                words = text.split()
                counter = Counter(words)
                ubda_cnt_dict.append(counter['없다'] + counter['없음'] + counter['모름'])

            result_df['ubda_cnt'] = pd.Series(ubda_cnt_dict)
            
            new_cols = result_df.columns.tolist()
            
            remove_set = {'text_response','knowcode'}
            new_cols = [i for i in new_cols if i not in remove_set]

            if train == True:
                result_df = result_df[new_cols + ['knowcode']]
            else:
                result_df = result_df[new_cols]
            
            return result_df
        test_list = []
        train_list = []
        for i in range(4):
            test_list.append(add_subj_cols(self.know_test[i], word_dict_year[i], train = False))
            train_list.append(add_subj_cols(self.know_train[i], word_dict_year[i], train = True))
        
        self.know_train = train_list
        self.know_test = test_list
        
        print('전처리 5_1단계 완료!')
        
        return self.know_train, self.know_test
    
    def step_5_2_prep(self, branch='2_2'):
        '''
        step_5_1_prep에서 하이퍼파라미터 selection_size만 150에서 200으로 변경하여 사용합니다
        '''
        if branch == '2_1':
            raise '잘못된 branch를 사용하였습니다'
        elif branch == '2_2':
            # 전처리 5_1단계를 시행함
            self.selection_size = 200
            self.know_train, self.know_test = self.step_5_1_prep(branch = "2_2")
        else:
            print('올바른 이전 branch를 입력해주세요 : [branch = "2_1" or branch = "2_2"]')
        
        print('전처리 5_2단계 완료!')
        
        return self.know_train, self.know_test