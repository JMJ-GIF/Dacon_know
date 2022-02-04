# data_0116에 대한 설명

출처 :

Stage 1 - 본 data는 ".\minje\pyfile\preprocessing_1)_clean_null_datatype.py"로 데이터를 생성한후  
Stage 2 -  ".\minje\pyfile\preprocessing_2_2)_handling_subj_question.py"로 2차 가공한 후
Stage 3 -  ".\minje\pyfile\preprocessing_4)_manual_type_correction.py"로 3차 가공한 후(by 정훈)
Stage 4 -  ".\minje\pyfile\preprocessing_5)_data_imputation.py"로 4차 가공한 것입니다

## Stage 1
1. Null 값을 모두 str '0'으로 대체하여 처리함

2. 공백 값을 모두 str '0'으로 대체하여 처리함

3. data type을 int / float / object로 정리함
    * int나 float으로 대답해야하는데 object로 대답한 문항들을 일일이 다 고쳤습니다. 어떻게 바뀌었는지는 ipynb를 확인하세요!  
    * 오류인 데이터는 모두 '0'으로 치환하였습니다

4. Data Shape

    2017   
        train (9486, 156) / test (9486, 155)  
    2018  
        train (9072, 141) / test (9069, 140)  
    2019    
        train (8555, 153) / test (8554, 152)  
    2020   
        train (8122, 185) / test (8122, 184)  
 
## Stage 2
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

6. Data Shape

    2017   
        train (9486, 151) / test (9486, 149)   
    2018  
        train (9072, 136) / test (9069, 134)  
    2019  
        train (8555, 148) / test (8554, 146)  
    2020  
        train (8122, 181) / test (8122, 179)

## Stage 3
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

3. Data Shape

    2017   
        train (9486, 155) / test (9486, 153)   
    2018  
        train (9072, 130) / test (9069, 128)  
    2019  
        train (8555, 152) / test (8554, 150)  
    2020  
        train (8122, 185) / test (8122, 183)

## Stage 4
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