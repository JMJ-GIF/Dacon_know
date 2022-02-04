# data_0105에 대한 설명

출처 :

Stage 1 - 본 data는 ".\minje\pyfile\preprocessing_1)_clean_null_datatype.py"로 데이터를 생성한후  
Stage 2 -  ".\minje\pyfile\preprocessing_2_1)_handling_subj_question.py"로 2차 가공한 것입니다  

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

4. "major"에 대항하는 문항은 1.2.3.과정을 생략함  

__때문에 결과 데이터 프레임은 obj에 해당하는 칼럼이 [major, text_response]로 단 2개이고, 나머지는 모두 int/float임.__

5. Data Shape

    2017   
        train (9486, 147) / test (9486, 146)   
    2018  
        train (9072, 132) / test (9069, 131)  
    2019  
        train (8555, 145) / test (8554, 144)  
    2020  
        train (8122, 180) / test (8122, 179)  
