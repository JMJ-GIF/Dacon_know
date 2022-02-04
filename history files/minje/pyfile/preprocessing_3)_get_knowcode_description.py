# -------------------------------------------------------------------------------------------------------------------------#
#                                                라이브러리 임포트하기                                                      #
# -------------------------------------------------------------------------------------------------------------------------#
# pip3 install pdfminer.six 를 터미널에 쳐서 pdf관련 라이브러리를 다운로드 해주세요~

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from glob import glob
import warnings
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfpage import PDFPage
from io import StringIO

# 데이터가 담겨있는 path로 설정하기
warnings.filterwarnings(action='ignore') 

know_train = [pd.read_csv(path) for path in sorted(glob('./data_0105/train/*.csv'))]
know_test = [pd.read_csv(path) for path in sorted(glob('./data_0105/test/*.csv'))]

# -------------------------------------------------------------------------------------------------------------------------#
#                                                       함수정의                                                            #
# -------------------------------------------------------------------------------------------------------------------------#

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

# -------------------------------------------------------------------------------------------------------------------------#
#                                                       pdf 고쳐주기                                                        #
# -------------------------------------------------------------------------------------------------------------------------#

raw_pdf_2017 = convert_pdf_to_txt('./data_0103/KNOW 메타데이터/2017_변수값.pdf')
raw_pdf_2018 = convert_pdf_to_txt('./data_0103/KNOW 메타데이터/2018_변수값.pdf')
raw_pdf_2019 = convert_pdf_to_txt('./data_0103/KNOW 메타데이터/2019_변수값.pdf')
raw_pdf_2020 = convert_pdf_to_txt('./data_0103/KNOW 메타데이터/2020_변수값.pdf')

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

# -------------------------------------------------------------------------------------------------------------------------#
#                                             데이터 내에 존재하는 오류 보정하기                                             #
# -------------------------------------------------------------------------------------------------------------------------#
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


def fix_bug_in_pdf_df(pdf_df):
    
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
    modified_df['description'] = modified_df['even_description'].str.cat(modified_df['odd_description'], sep=',')

    modified_pdf_df = pdf_df.drop(total_indice)
    modified_pdf_df = pd.concat([modified_pdf_df, modified_df[['knowcode','description']]]).reset_index(drop=True)
    
    return modified_pdf_df

# -------------------------------------------------------------------------------------------------------------------------#
#                                                      내보내기                                                             #
# -------------------------------------------------------------------------------------------------------------------------#
pdf_list = [pdf_2017, pdf_2018, pdf_2019, pdf_2020]
years = ['2017', '2018', '2019', '2020']

for idx in range(4):
    modified_pdf_df = fix_bug_in_pdf_ver2_df(pdf_list[idx])
    modified_pdf_df.to_csv('pdf_description_{}.csv'.format(years[idx]), index=False)