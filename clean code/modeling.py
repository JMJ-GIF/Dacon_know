import pandas as pd
import numpy as np
from glob import glob
from tqdm import tqdm
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
import re


from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
from catboost import CatBoostClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from gensim.models.doc2vec import Doc2Vec, TaggedDocument

from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfpage import PDFPage
from io import StringIO

import jellyfish
from difflib import SequenceMatcher
from tqdm.notebook import tqdm
from sklearn.ensemble import ExtraTreesClassifier

from prep import preprocessing

class Modeling(preprocessing):
    
    # 클래스 변수
    years = ['2017', '2018', '2019', '2020']
    RANDOM_STATE = 42
    pca_component = 2
    pdf_path = './dataset/KNOW 메타데이터'
    
    def __init__(self, know_train, know_test):
        self.know_train = know_train
        self.know_test = know_test
        
    def _label_encoding(self, trains, tests):
        # encoding train data
        year_encoder = {}
        for year, df in zip(self.years, trains):
            encoders = {}
            
            for col in df.columns:
                if col == 'idx':
                    continue
                try:
                    df[col] = df[col].map(int)
                except:
                    encoder = LabelEncoder()
                    df[col] = df[col].map(str)
                    df[col] = encoder.fit_transform(df[col])
                    encoders[col] = encoder 
            year_encoder[year] = encoders
        
        # encode test data
        for year, df in zip(self.years, tests):
            encoders = {}
        
            for col in df.columns:
                try:
                    df[col] = df[col].map(int)
                except:
                    encoder = year_encoder[year][col]
                    df[col] = df[col].map(str)
                    category_map = {category: idx for idx, category in enumerate(encoder.classes_)}
                    df[col] = df[col].apply(lambda x: category_map[x] if x in category_map else -1) # train set에서 보지못한 카테고리변수 -1(UNK) 처리
            
        return trains, tests
    
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
        
        def fix_bug_in_pdf_ver2_df(pdf_df):
            ''' 
            하나의 knowcode의 2개의 직업이 매칭되는 중복오류를 수정하기 위한 함수입니다.
            오류에 해당되는 코드인 경우 한 셀에 두 가지 직업을 명시하도록 했습니다
                
            Parameters
            pdf_text(string)  : 읽어온 pdf string
            
            Returns
            pdf_df(DataFrame) : 로직에 따라 전처리가 완료된 pdf
            
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
            modified_df['description'] = modified_df['even_description'].str.cat(modified_df['odd_description'], sep=' ')

            modified_pdf_df = pdf_df.drop(total_indice)
            modified_pdf_df = pd.concat([modified_pdf_df, modified_df[['knowcode','description']]]).reset_index(drop=True)
            
            error_indice = [list(modified_pdf_df[modified_pdf_df['knowcode']==knowcode].index)[0] for knowcode in duplicated_knowcode]
            modified_pdf_df['error'] = 0
            modified_pdf_df.loc[error_indice,'error'] = 1
            
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
            modified_pdf_df = fix_bug_in_pdf_ver2_df(pdf_list[idx])
            modified_pdf_list.append(modified_pdf_df)
        
        print('description csv 생성완료')
        
        return modified_pdf_list
    
    
    def fit_pred_et(self, n_estimator, prep_phase = None):
        '''
        extratree classifier를 적용하는 함수입니다.
        n_estimator를 조정하여 더 깊게 훈련할 수 있습니다.
        모델을 돌리기 전에, 주관식 문항에 대해 따로 처리가 되어있지 않다고 가정하므로, label encoder를 돌려줍니다
        
        해당 모델은 preprocessing step에 관련없이 시행 가능합니다.
        '''
        
        self.know_train, self.know_test = self._label_encoding(self.know_train, self.know_test)
        
        # train data와 test data를 다루기 편하게 딕셔너리에 넣어줍시다
        train_data = {}
        for year, df in zip(self.years, self.know_train):
            if 'descrpition' in df.columns:
                train_data[year] = {'X': df.drop(['idx','knowcode','description'], axis=1),
                                    'y': df['knowcode']}
            else:
                train_data[year] = {'X': df.drop(['idx','knowcode'], axis=1),
                                    'y': df['knowcode']}
            
        test_data = {}
        for year, df in zip(self.years, self.know_test):
            train_columns = train_data[year]['X'].columns
            test_data[year] =  {'X': df[train_columns]}
        
        # 메모리 부담을 피하기 위해 학습과 prediction을 동시에 진행합니다
        # 학습과 예측을 진행하고 메모리 상에서 들고 있는 model을 버립니다

        et_predicts = [] 
        for year in tqdm(self.years):
            # train
            model = ExtraTreesClassifier(n_estimators=n_estimator, random_state=self.RANDOM_STATE, n_jobs=8)
            model.fit(train_data[year]['X'].iloc[:, :], train_data[year]['y'])

            # predict
            pred = model.predict(test_data[year]['X'])
            et_predicts.extend(pred)
        
        return et_predicts

    def fit_pred_rf(self, n_estimator, prep_phase = None):
        '''
        RandomForest classifier를 적용하는 함수입니다.
        n_estimator를 조정하여 더 깊게 훈련할 수 있습니다.
        모델을 돌리기 전에, 주관식 문항에 대해 따로 처리가 되어있지 않다고 가정하므로, label encoder를 돌려줍니다
        
        해당 모델은 preprocessing step에 관련없이 시행 가능합니다.
        '''
        
        self.know_train, self.know_test = self._label_encoding(self.know_train, self.know_test)
        
        # train data와 test data를 다루기 편하게 딕셔너리에 넣어줍시다
        train_data = {}
        for year, df in zip(self.years, self.know_train):
            if 'descrpition' in df.columns:
                train_data[year] = {'X': df.drop(['idx','knowcode','description'], axis=1),
                                    'y': df['knowcode']}
            else:
                train_data[year] = {'X': df.drop(['idx','knowcode'], axis=1),
                                    'y': df['knowcode']}
            
        test_data = {}
        for year, df in zip(self.years, self.know_test):
            train_columns = train_data[year]['X'].columns
            test_data[year] =  {'X': df[train_columns]}
        
        # 메모리 부담을 피하기 위해 학습과 prediction을 동시에 진행합니다
        # 학습과 예측을 진행하고 메모리 상에서 들고 있는 model을 버립니다

        rf_predicts = [] 
        for year in tqdm(self.years):
            # train
            model = RandomForestClassifier(n_estimators=n_estimator, random_state=self.RANDOM_STATE, n_jobs=8)
            model.fit(train_data[year]['X'].iloc[:, :], train_data[year]['y'])

            # predict
            pred = model.predict(test_data[year]['X'])
            rf_predicts.extend(pred)
        
        return rf_predicts
    
    def fit_pred_catboost(self, iteration, gpu=False, prep_phase = None):
        '''
        Catboost를 적용하는 함수입니다.
        iteration 조정하여 더 깊게 훈련할 수 있습니다.
        모델을 돌리기 전에, 주관식 문항에 대해 따로 처리가 되어있지 않다고 가정하므로, label encoder를 돌려줍니다
        
        GPU를 사용하고자 한다면 가능한 환경에서 시행해주세요.
        
        해당 모델은 preprocessing step에 관련없이 시행 가능합니다.
        '''
        
        self.know_train, self.know_test = self._label_encoding(self.know_train, self.know_test)
        
        # train data와 test data를 다루기 편하게 딕셔너리에 넣어줍시다
        train_data = {}
        for year, df in zip(self.years, self.know_train):
            if 'descrpition' in df.columns:
                train_data[year] = {'X': df.drop(['idx','knowcode','description'], axis=1),
                                    'y': df['knowcode']}
            else:
                train_data[year] = {'X': df.drop(['idx','knowcode'], axis=1),
                                    'y': df['knowcode']}
            
        test_data = {}
        for year, df in zip(self.years, self.know_test):
            train_columns = train_data[year]['X'].columns
            test_data[year] =  {'X': df[train_columns]}
        
        # 메모리 부담을 피하기 위해 학습과 prediction을 동시에 진행합니다
        # 학습과 예측을 진행하고 메모리 상에서 들고 있는 model을 버립니다

        ct_predicts = [] 
        for year in tqdm(self.years):
            # train
            if gpu == True:
                model = CatBoostClassifier(iterations=iteration,
                                random_state=self.RANDOM_STATE,
                                task_type='GPU',
                                loss_function='MultiClass',
                                eval_metric='TotalF1'
                                )
            else:
                model = CatBoostClassifier(iterations=iteration,
                                random_state=self.RANDOM_STATE,
                                loss_function='MultiClass',
                                eval_metric='TotalF1'
                                )
            model.fit(train_data[year]['X'].iloc[:, :], train_data[year]['y'])

            # predict
            pred = model.predict(test_data[year]['X'])
            ct_predicts.extend(pred)
        
        return ct_predicts

    def fit_pred_tf_idf_pca_et(self, n_estimator, prep_phase):
        '''
        text_response열에 대해서 
        n_estimator를 조정하여 더 깊게 훈련할 수 있습니다.
        주관식 문항(text_response)에 대해서 tf_df_matrix를 만든뒤 pca해줍니다.(pca_component는 클래스변수에서 가져옵니다)
        그 후 et를 통해 예측합니다
        
        해당 모델은 preprocessing step = 4인 데이터를 반드시 넣어주어야 합니다
        (그 외 데이터는 호환안됨)
        '''
        if prep_phase == '4':
            pass
        else:
            raise '잘못된 prep_phase를 사용하였습니다'
        
        text_info_cols = ['text_response']

        ## train
        for i in range(4):
            for text_info_col in text_info_cols:
                self.know_train[i].loc[self.know_train[i][text_info_col]=='없다', text_info_col] = ''
                self.know_train[i].loc[self.know_train[i][text_info_col]=='없음', text_info_col] = ''
                self.know_train[i].loc[self.know_train[i][text_info_col]=='0', text_info_col] = ''
                self.know_train[i].loc[self.know_train[i][text_info_col]=='무', text_info_col] = ''
                self.know_train[i].loc[self.know_train[i][text_info_col]=='모름', text_info_col] = ''
                self.know_train[i].loc[self.know_train[i][text_info_col]=='공란', text_info_col] = ''

        ## test  
        for i in range(4):
            for text_info_col in text_info_cols:
                self.know_test[i].loc[self.know_test[i][text_info_col]=='없다', text_info_col] = ''
                self.know_test[i].loc[self.know_test[i][text_info_col]=='없음', text_info_col] = ''
                self.know_test[i].loc[self.know_test[i][text_info_col]=='0', text_info_col] = ''
                self.know_test[i].loc[self.know_test[i][text_info_col]=='무', text_info_col] = ''
                self.know_test[i].loc[self.know_test[i][text_info_col]=='모름', text_info_col] = ''
                self.know_test[i].loc[self.know_test[i][text_info_col]=='공란', text_info_col] = ''
        
                
        self.know_train, self.know_test = self._label_encoding(self.know_train, self.know_test)
        
        def get_tf_idf_df(know_data):
        
            doc_nouns_list = [doc for doc in (know_data['text_response'])]
            
            tfidf_vectorizer = TfidfVectorizer(min_df=1)
            tfidf_matrix = pd.DataFrame(tfidf_vectorizer.fit_transform(doc_nouns_list).toarray())
            
            return tfidf_matrix

        def use_pca(preprocessed_data, n_component):

            # 평균이 0이 되도록 조정
            data_scaled = StandardScaler().fit_transform(preprocessed_data)
            # PCA
            pca = PCA(n_components=n_component)
            
            data_pca = pd.DataFrame(pca.fit_transform(data_scaled), columns = range(n_component))
            
            return data_pca
        
        train_data = {}
        test_data = {}
        n_components = {}

        for year, df in tqdm(zip(self.years, self.know_train)):
            
            tf_idf_matrix = get_tf_idf_df(df)
            
            tf_idf_pca = use_pca(tf_idf_matrix, self.pca_component)
            
            train_pca = pd.concat([df,tf_idf_pca],axis=1).drop(['idx','knowcode','text_response','description'], axis = 1)
            
            train_data[year] = {'X': train_pca, # idx 제외
                                'y': df['knowcode']}
            
            n_components[year] = self.pca_component
             
        for year, df in tqdm(zip(self.years, self.know_test)):
            
            n_component = n_components[year]
            
            tf_idf_matrix = get_tf_idf_df(df)
            
            tf_idf_pca = use_pca(tf_idf_matrix, n_component)
            
            test_pca = pd.concat([df,tf_idf_pca],axis=1).drop(['idx','text_response'], axis = 1)
            
            test_data[year] = test_pca
            
        et_predicts = [] 
        for year in tqdm(self.years):
            # train
            model = ExtraTreesClassifier(n_estimators=n_estimator, random_state=self.RANDOM_STATE, n_jobs=8)
            model.fit(train_data[year]['X'].iloc[:, :], train_data[year]['y'])

            # predict
            pred = model.predict(test_data[year])
            et_predicts.extend(pred)
        
        return et_predicts
    
    def fit_pred_doc2vec_classifier(self, prep_phase):
        '''
        test set의 text_response와 가장 유사한 text_response를 train에서 찾습니다.
        해당 레코드의 knowcode를 prediction으로 생각합니다.
        
        해당 모델은 preprocessing step = 4인 데이터를 반드시 넣어주어야 합니다
        (그 외 데이터는 호환안됨)
        '''
        if prep_phase == '4':
            pass
        else:
            raise '잘못된 prep_phase를 사용하였습니다'
        
        # doc2vec을 사용할거라 txt열을 모두 하나로 모아줘야해요
        text_info_cols = {"2017": ['sim_job','bef_job','able_job','major'],
                        "2018": ['sim_job','bef_job','able_job','major'],
                        "2019": ['bef_job','able_job','major'],
                        "2020": ['major'],}

        for i, year in enumerate(self.years):
            text_info_col = text_info_cols[year]
            for col in text_info_col:
                self.know_train[i]['text_response'] = self.know_train[i]['text_response'] + ' ' + self.know_train[i][col]
                self.know_test[i]['text_response'] = self.know_test[i]['text_response'] + ' ' + self.know_test[i][col]
            self.know_train[i].drop(text_info_col,axis=1, inplace=True)
            self.know_test[i].drop(text_info_col,axis=1, inplace=True)

        # text_response 내 존재하는 stopwords들을 제거해줍시다
        stopwords = ['없다','없음','0','모름','공란']
        for i in tqdm(range(4)):
            for k, doc in enumerate(self.know_train[i]['text_response']):
                doc_list = doc.split(' ')
                new_list = [word for word in doc_list if word not in stopwords]
                new_string = ''
                for word in new_list:
                    new_string += word
                    new_string += ' '
                self.know_train[i].loc[k,'text_response'] = new_string[:-1]
            for k, doc in enumerate(self.know_test[i]['text_response']):
                doc_list = doc.split(' ')
                new_list = [word for word in doc_list if word not in stopwords]
                new_string = ''
                for word in new_list:
                    new_string += word
                    new_string += ' '
                self.know_test[i].loc[k,'text_response'] = new_string[:-1]
        
        def doc2vec_train(know_train):
            '''

            모든 text열이 합쳐진 text_response에 대해, Doc2vec을 적용하고, tagging함.
            
            '''
            my_texts = know_train[['idx','text_response']]
            my_texts_and_tags = [(row.text_response, [str(row.idx)]) for row in my_texts.itertuples()]
            TRAIN_documents = [TaggedDocument(words=text, tags=tags) for text, tags in my_texts_and_tags]
            model = Doc2Vec(TRAIN_documents, vector_size=1000, window=4, epochs=40, min_count=0, workers=4)
            
            return model
        
        def doc2vec_pred(know_train, know_test, model, viz = False):
            '''
        
            test set의 text_response와 가장 유사한 text_response를 train에서 찾음.
            해당 레코드의 knowcode를 prediction으로 생각함.
            
            '''
            Test_documents = [doc for doc in know_test['text_response']]
            
            pred_list = []
            sim_list = []
            for text in Test_documents:
                text_split = text.split(' ')
                inferred_v = model.infer_vector(text_split)
                most_similar_docs = model.docvecs.most_similar(positive=[inferred_v], topn=1)
                
                top1_train_idx = int(most_similar_docs[0][0])
                top1_sim = round(float(most_similar_docs[0][1]),2)
                sim_list.append(top1_sim)
                
                pred = list(know_train.loc[know_train['idx']==top1_train_idx]['knowcode'])[0]
                pred_list.append(pred)
                
            return pred_list, sim_list
        
        # 4개년도 모델을 학습하고 저장합니다
        models = []
        for i in tqdm(range(4)):
            model = doc2vec_train(self.know_train[i])
            models.append(model)
            
        # 4개년도 데이터로 예측합니다
        doc2vec_preds = []
        for i in tqdm(range(4)):
            pred, _ = doc2vec_pred(self.know_train[i], self.know_test[i], models[i])
            doc2vec_preds.extend(pred)
        
        return doc2vec_preds  
    
    def fit_pred_string_classifier(self, col, method, threshold, n_estimator, prep_phase):
        '''
        1. 데이터  
            * preprocessing step = 3_1인 데이터를 사용함  
            * 데이터에 knowcode의 설명인 description열을 추가함
            * train set과 test set의 str열인 major와 description에 대해서 Stopwords들을 제거함  
            * idx열을 인덱스로 만듦   
        # 
        2. fitting  
            * STEP 1 : Major열을 기준으로 정합성을 테스트하고, 올바르지 못한것은 et로 예측할 것이므로 따로 빼두기 **(string_compare fitting)**   
            * STEP 2 : Major열과 description을 jaro_distance로 비교하여 유사도를 도출함 **(string_compare fitting)**  
            * STEP 3 : 전체 train 데이터를 기준으로 et_300으로 학습함 **(et fitting)**
        #          
        3. Prediction  
            * STEP 1 : Similarity와 사전에 정한 threshold를 비교하여 높은 레코드에 대해서만 string_compare로 예측을 진행 **(string_compare predict)**
            * STEP 2 : 그렇지 못한 데이터에 대해서는 et_300으로 예측값을 도출함 **(et predict)**
        
        여러번 시행착오 결과 col = 'major', method='jaro_distance', threshold = 0.99정도가 괜찮음
        
        해당 모델은 preprocessing step = 3_1인 데이터를 반드시 넣어주어야 합니다
        (그 외 데이터는 호환안됨)
        '''
        if prep_phase == '3_1':
            pass
        else:
            raise '잘못된 prep_phase를 사용하였습니다'     
        
        # 모델에 쓸 수 있게 데이터를 전처리 합니다
        text_info_cols = {"2017": ['sim_job','bef_job','able_job','major'],
                        "2018": ['sim_job','bef_job','able_job','major'],
                        "2019": ['bef_job','able_job','major'],
                        "2020": ['major'],} 
        
        description_dfs = self._get_pdf_data()
        
        # description 열을 추가해줘야 합니다
        for i in range(4):
            self.know_train[i] = pd.merge(self.know_train[i], description_dfs[i], on='knowcode',how='left').fillna('0')

        text_info_cols = ['major','description']

        # major와 description 열에 대한 전처리를 해줍시다
        ## train
        for i, year in enumerate(self.years):
            know_train[i].drop(['상태', '이용', '것이므', '사용'], axis=1, inplace=True) # 쓸모없는 열 제거
            for text_info_col in text_info_cols:
                self.know_train[i].loc[self.know_train[i][text_info_col]=='없다', text_info_col] = ''
                self.know_train[i].loc[self.know_train[i][text_info_col]=='없음', text_info_col] = ''
                self.know_train[i].loc[self.know_train[i][text_info_col]=='0', text_info_col] = ''
                self.know_train[i].loc[self.know_train[i][text_info_col]=='무', text_info_col] = ''
                self.know_train[i].loc[self.know_train[i][text_info_col]=='모름', text_info_col] = ''
                self.know_train[i].loc[self.know_train[i][text_info_col]=='공란', text_info_col] = ''

        ## test(major만 가능)       
        for i, year in enumerate(self.years):
            text_info_col = 'major'
            self.know_test[i].loc[self.know_test[i][text_info_col]=='없다', text_info_col] = ''
            self.know_test[i].loc[self.know_test[i][text_info_col]=='없음', text_info_col] = ''
            self.know_test[i].loc[self.know_test[i][text_info_col]=='0', text_info_col] = ''
            self.know_test[i].loc[self.know_test[i][text_info_col]=='무', text_info_col] = ''
            self.know_test[i].loc[self.know_test[i][text_info_col]=='모름', text_info_col] = ''
            self.know_test[i].loc[self.know_test[i][text_info_col]=='공란', text_info_col] = ''
                
        # idx 열을 인덱스로 만들어줍시다
        for i in range(4):
            self.know_train[i].index = self.know_train[i]['idx']
            self.know_train[i].drop('idx',axis=1,inplace=True)
            self.know_test[i].index = self.know_test[i]['idx']
            self.know_test[i].drop('idx',axis=1,inplace=True)
            
        methods = {'SequenceMatcher':['knowcode_','similarity_'],
           'levenshtein_distance':['knowcode_lev_','similarity_lev_'],
           'damerau':['knowcode_dlev_','similarity_dlev_'],
           'jaro_distance':['knowcode_jaro_','similarity_jaro_'],
           'jaro_winkler':['knowcode_jarow_','similarity_jarow_'],
           'hamming_distance':['knowcode_ham_','similarity_ham_']}
        
        def similar(a, b):
            '''
            SequenceMatcher 방식을 사용할 경우 적용하는 함수임.
            '''
            return SequenceMatcher(None, a, b).ratio()

        
        ## STEP 1
        # 어떤 열을 기준으로 fit할지 정하고, 그 열의 정합성을 테스트합니다. 올바르지 못한것은 et로 무조건 예측해야하므로 빼둡니다

        # exception) 기준열이 공란이거나 description이 공란인 경우 simiarity 예측에서 제외
        error_indice = []
        for i in range(4):
            error_index = list(self.know_test[i].loc[self.know_test[i][col]=='', col].index)
            error_indice.extend(error_index)

        # error_indice의 중복값 제거
        error_indice = list(set(error_indice))

        ## STEP 2
        # string_compare를 model을 정의하고, 진행해보자

        def string_compare(know_data, col, description_dfs, method):
            '''
            적용하고자 하는 string_compare방법을 선택하고, 어떤 열을 기준으로 string의 유사도를 비교할 것인지 input으로 넣어주기.
            선택한 칼럼과 방법에 따라서 description열과 similarity를 비교하여 값을 도출함.
            
            '''

            data = know_data.copy()
            
            for i, year in enumerate(self.years):
                # iterate 4 years
                text_info_col = col
                text_info_list = list(data[i][text_info_col])
            
                knowcode_text_info_col = []
                similarity_text_info_col = []

                for possible_answer in tqdm(text_info_list):
                    # iterate each string in know_train[i][sim_job, bef_job, ...]
                    knowcode = "0"
                    similarity = 0.0
                    max_similarity_index = 0
                    for descr_row in description_dfs[i].itertuples():
                        # iterate each row in description_dfs[i]
                        if possible_answer != '':
                            if method == 'SequenceMatcher':
                                score = similar(possible_answer, descr_row.description)
                            elif method == 'levenshtein_distance':
                                score = jellyfish.levenshtein_distance(possible_answer, descr_row.description)
                            elif method == 'damerau':
                                score = jellyfish.damerau_levenshtein_distance(possible_answer, descr_row.description)
                            elif method == 'jaro_distance':
                                score = jellyfish.jaro_distance(possible_answer, descr_row.description)
                            elif method == 'jaro_winkler':
                                score = jellyfish.jaro_winkler(possible_answer, descr_row.description)
                            elif method == 'hamming_distance':
                                score = jellyfish.hamming_distance(possible_answer, descr_row.description)
                            
                            if score > similarity:
                                similarity = score
                                max_similarity_index = descr_row.Index
                    if similarity == 0:
                        knowcode_text_info_col.append("0")
                    else:
                        knowcode_text_info_col.append(description_dfs[i].iloc[max_similarity_index, 0])
                    similarity_text_info_col.append(similarity)
                data[i][methods[method][0] + text_info_col] = knowcode_text_info_col
                data[i][methods[method][1] + text_info_col] = similarity_text_info_col
                
            return data

        string_model_data = string_compare(self.know_test, col, description_dfs, method)
        
        # predict
        ## string method로 맞춘 knowcode를 정답으로 배출합니다

        string_pred = {}
        for i, year in enumerate(self.years):
            # string으로 prediction한 dataFrame인 string_predict_df 구하기
            sim_cols = [method + col for method in methods[method]]
            sims_df = string_model_data[i][sim_cols]

            # string_predict_df에 대해서 error_index는 아닌지 체크해보기
            string_predict_tmp_indice = [idx for idx in sims_df.index]
            string_predict_pure_indice = list(set(string_predict_tmp_indice) - set(error_indice))

            # error가 아닌 index에 대해서 결과를 내보내기
            # 결과 데이터 프레임은 (index = idx, knowcode)인 df
            non_filtered_result_df = sims_df.loc[string_predict_pure_indice,[sim_cols[0],sim_cols[1]]]
            filter_result_df = non_filtered_result_df[non_filtered_result_df[sim_cols[1]] > threshold]\
                                .rename(columns = {sim_cols[0]:'knowcode'}) # simliarity에 대한 조건을 걸어줄 수 있음
            string_pred[year] = filter_result_df['knowcode']
            
        # train_data를 준비합니다
        et_train_data_dict = {}
        for year, df in zip(self.years, self.know_train):
            et_train_data_dict[year] = {'X': df.drop(['knowcode','error','description','major'], axis=1),
                                        'y': df['knowcode']} 
            
        # test data를 준비합니다
        # compare_models에서 예측하지 않았던 idx로만 예측을 진행해야하므로, 그것들을 빼고 예측 test 데이터를 만듭니다
        et_test_data_dict = {}
        for year, df in zip(self.years, self.know_test):
            train_columns = et_train_data_dict[year]['X'].columns
            
            compare_indice = list(string_pred[year].index)
            all_indice = list(df[train_columns].index)
            et_indice = list(set(all_indice) - set(compare_indice))
            et_df = df[train_columns].loc[et_indice,:]
            
            et_test_data_dict[year] = et_df

        
        predict_dfs = [] 
        for year in tqdm(self.years):
            # train
            model = ExtraTreesClassifier(n_estimators=n_estimator, random_state = self.RANDOM_STATE)
            model.fit(et_train_data_dict[year]['X'], et_train_data_dict[year]['y'])
            
            # predict
            et_indice = list(et_test_data_dict[year].index)
            et_pred= model.predict(et_test_data_dict[year])

            pred_df = pd.DataFrame(index=et_indice)
            pred_df['knowcode'] = et_pred
            predict_dfs.append(pred_df['knowcode'])
            
        ## STEP 3 et_pred와 compare_pred를 합쳐줘야합니다
        final_pred = []
        for i, year in enumerate(self.years):
            final_pred_df = pd.concat([string_pred[year],predict_dfs[i]]).sort_index()
            pred = list(final_pred_df)
            final_pred.extend(pred)
        
        return final_pred