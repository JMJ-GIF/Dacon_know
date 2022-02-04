# data_0104에 대한 설명

출처 :

Stage 1 - 본 data는 ".\minje\preprocessing_1)_drop_mismatch_data.py"로 데이터를 생성한 것입니다

## Stage 1
1. Null 값을 모두 str '0'으로 대체하여 처리함

2. 공백 값을 모두 str '0'으로 대체하여 처리함

3. data type을 int / float / object로 정리함
    * int나 float으로 대답해야하는데 object로 대답한 문항들을 일일이 다 고쳤습니다. 어떻게 바뀌었는지는 ipynb를 확인하세요!
    * * 오류인 데이터는 모두 '0'으로 치환하였습니다  
    
4. Data Shape

    2017   
        train (9486, 156) / test (9486, 155)  
    2018  
        train (9072, 141) / test (9069, 140)  
    2019  
        train (8555, 153) / test (8554, 152)  
    2020  
        train (8122, 185) / test (8122, 184)  


