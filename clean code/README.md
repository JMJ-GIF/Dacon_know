# Preprocessing

## necessary for running
1. konlpy  
konlpy 다운로드는 생각보다 복잡합니다. 관련해서 잘 설명해둔 링크를 첨부합니다.  
```
https://konlpy-ko.readthedocs.io/ko/v0.4.3/install/
```
  
2. pdfminer  
```
터미널에 아래 명령어를 입력하여 다운로드 받아주세요  
pip3 install pdfminer.six
```
  
3. stopwords_path 설정  
konlpy에서 명사를 효과적으로 분리하기 위해 불용어 사전이 있으면 좋습니다.  
본인이 원하는 stopowords를 다운로드 받거나 추가하여 경로 설정을 해주세요.  
```
prep = preprocessing(know_train, know_test)  
prep.Stopwords_path = '../../minje/stopwords.txt' (stopwords 파일 명까지 표기)
```
  
4. pdf_path 설정  
description을 추가하기 위해 주최즉에서 제공한 KNOWMETA_DATA.pdf가 있어야합니다  
관련한 pdf 경로를 설정해주세요  
```
prep = preprocessing(know_train, know_test)  
prep.pdf_path = './KNOW 메타데이터' (pdf파일이 들어있는 폴더까지만 표기)  
```


## Structure of class
산발적으로 흩어져있는 전처리 코드를 하나의 class로 묶어서 표현했습니다.  

```
Preprocessing
├─ 클래스 변수 정의 및 생성자(__init__)
|
├─_get_pdf_data : knowcode에 대한 description을 생성해주는 함수입니다.(step_2_2_prep에서 사용)
|
├─_preprocessing_text_data : 주관식 문항을 handling하기 위해 사용하는 함수입니다.(step_2에서 사용)
|
└─data0104(step_1_prep) : datatype, nan값을 handling하는 함수입니다
    |
    ├─data0105(step_2_1_prep) : 데이터 내 존재하는 주관식 문항을 handling 하는 함수입니다(major, text_response로 통합)
    |   |
    |   └─data0112(step_3_1_prep) : 정리한 text_response열을 임의로 잘라 단어별로 원핫 인코딩을 실시함
    |
    └─data0113(step_2_2_prep) : 데이터 내 존재하는 주관식 문항을 handling 하는 함수입니다(sim_job, bef_job, able_job, major, text_response로 통합)
        |
        └─data0115(step_3_2_prep) : 데이터 내 존재하는 오류를 fix하고, 데이터 타입을 엄밀하게 고려하여 재정의합니다.
            |
            └─data0116(step_4_prep) : 해당하는 칼럼에 대해서, mean과 mode로 data imputation을 진행합니다
                |
                └─data0119(step_5_1_prep) : step_3_1_prep의 한계점을 더 발전시켜 전처리해주는 함수입니다.(selection_size = 150)
                |
                └─data0120(step_5_2_prep) : step_3_1_prep의 한계점을 더 발전시켜 전처리해주는 함수입니다.(selection_size = 200)

```

## example for using
하위 prep을 시행하면 상위 prep은 자동으로 시행됩니다.  
(ex. step_3_prep을 시행하면 step_1_prep -> step_2_2_prep -> step_3_prep 순으로 시행됨)  

* step_1_prep
```
prep = preprocessing(know_train, know_test)
know_train, know_test = prep.step_1_prep()
```
* step_2_1_prep
```
prep = preprocessing(know_train, know_test)
know_train, know_test = prep.step_2_1_prep()
```
* step_2_2_prep
```
prep = preprocessing(know_train, know_test)
know_train, know_test = prep.step_2_2_prep()
```
* step_3_1_prep
```
prep = preprocessing(know_train, know_test)
know_train, know_test = prep.step_3_1_prep(branch='2_1')
```
* step_3_2_prep
```
prep = preprocessing(know_train, know_test)
know_train, know_test = prep.step_3_2_prep(branch='2_2')
```
* step_4_prep
```
prep = preprocessing(know_train, know_test)
know_train, know_test = prep.step_4_prep(branch='2_2')
```
* step_5_1_prep
```
prep = preprocessing(know_train, know_test)
know_train, know_test = prep.step_5_1_prep(branch='2_2')
```
* step_5_2_prep
```
prep = preprocessing(know_train, know_test)
know_train, know_test = prep.step_5_2_prep(branch='2_2')
```

# Modeling

## prerequisite for running
1. gensim  
gensim 라이브러리를 깔아주세요  
```
pip install --upgrade gensim
```
  
2. jellyfish  
string 간 유사도를 계산해주는 라이브러리입니다  

```
pip install jellyfish
```
  
3. pdf_path 설정  
description을 추가하기 위해 주최즉에서 제공한 KNOWMETA_DATA.pdf가 있어야합니다  
관련한 pdf 경로를 설정해주세요  
```
model = Modeling(know_train, know_test)
model.pdf_path = './KNOW 메타데이터' (pdf파일이 들어있는 폴더까지만 표기)
```

## Structure of class
산발적으로 흩어져있는 전처리 코드를 하나의 class로 묶어서 표현했습니다.

```
Modeling
├─ 클래스 변수 정의 및 생성자(__init__)
|
├─_get_pdf_data : knowcode에 대한 description을 생성해주는 함수입니다.
|
├─_label_encoding : 주관식 문항들에 대한 label encoding을 해주는 함수입니다
|
├─ fit_pred_et(n_estimator, prep_phase= None) : ExtraTree Classifier를 통해 예측하는 함수입니다
|
├─ fit_pred_rf(n_estimator, prep_phase= None) : RandomForest Classifier를 통해 예측하는 함수입니다
|
├─ fit_pred_catboost(iteration, gpu=False, prep_phase = None) : Catboost Classifier를 통해 예측하는 함수입니다
|
├─ fit_pred_tf_idf_pca_et(n_estimator, prep_phase) : 주관식 문항에 대해 tf_idf_matrix를 생성하고 이를 pca하여 et로 예측합니다.
|                                                   (해당 모델은 preprocessing step = 4인 데이터를 반드시 넣어주어야 합니다)
├─ fit_pred_doc2vec_classifier(prep_phase) : train set과 test set의 주관식 문항간의 문서 유사도를 비교하여 종속변수를 예측합니다.
|                                                   (해당 모델은 preprocessing step = 4인 데이터를 반드시 넣어주어야 합니다)
├─ fit_pred_string_classifier(col, method, threshold, n_estimator, prep_phase) : knowcode에 대한 description과 사용자가 지정한 col간의 유사도를 비교하여 가장 비슷한 직업을 예측치로 내놓습니다.
                                                    (해당 모델은 preprocessing step = 3_1인 데이터를 반드시 넣어주어야 합니다)
```

## example for using

* fit_pred_et
```
model = Modeling(know_train, know_test)
et_preds =  model.fit_pred_et(n_estimator=300)
```
* fit_pred_rf
```
model = Modeling(know_train, know_test)
rf_preds =  model.fit_pred_rf(n_estimator=300)
```
* fit_pred_catboost
```
model = Modeling(know_train, know_test)
cat_preds =  model.fit_pred_catboost(iteration=300)
```
* fit_pred_tf_idf_pca_et
```
model = Modeling(know_train, know_test)
tf_idf_et_preds =  model.fit_pred_tf_idf_pca_et(n_estimator=300, prep_phase='4')
```
* fit_pred_doc2vec_classifier
```
model = Modeling(know_train, know_test)
doc2vec_preds =  model.fit_pred_doc2vec_classifier(prep_phase='4')
```
* fit_pred_string_classifier
```
model = Modeling(know_train, know_test)
doc2vec_preds =  model.fit_pred_string_classifier(col='major', method='jaro_distance', threshold=0.99, n_estimator=300, prep_phase='3_1')
```
