from sklearn import preprocessing
from prep import preprocessing
from modeling import Modeling
import pandas as pd
import numpy as np
from glob import glob
import warnings
warnings.filterwarnings(action='ignore') 
from statistics import multimode

# 전처리와 모델링을 반복하고, 이들을 hard voting방식을 통해 ensemble합니다
def ensembles_hard_voting(*preds):
    # 예측값들을 리스트로 받습니다.

    ROW_len = len(preds[0])
    
    # 이중 리스트로 정답들을 모아줍니다
    list_choices = []
    for i in range(ROW_len):
        choices = []
        for pred in preds:
            choices.append(pred[i])
        list_choices.append(choices)
    
    # Hard Voting을 진행합니다
    modes = []
    for i in range(ROW_len):
        modes.append(multimode(list_choices[i])[0])
    
    # 결과값을 모아볼까요?
    voting_matrix = pd.DataFrame(modes, columns=['knowcode'])
    # index 열 추가하기 위함. 내용물은 중요하지 않음
    sample_submission = pd.read_csv('./dataset/sample_submission.csv')
    voting_matrix['idx'] = sample_submission['idx']
    #열 순서 재배열
    voting_matrix = voting_matrix[['idx','knowcode']]
    
    return voting_matrix

# 실행예시
if __name__ == '__main__':
    # rf model
    know_train = [pd.read_csv(path) for path in sorted(glob('../../data_0103/train/*.csv'))]
    know_test = [pd.read_csv(path) for path in sorted(glob('../../data_0103/test/*.csv'))]

    prep = preprocessing(know_train, know_test)
    know_train, know_test = prep.step_5_2_prep()

    model = Modeling(know_train, know_test)
    rf_preds =  model.fit_pred_rf(n_estimator=300)

    # et model
    know_train = [pd.read_csv(path) for path in sorted(glob('../../data_0103/train/*.csv'))]
    know_test = [pd.read_csv(path) for path in sorted(glob('../../data_0103/test/*.csv'))]

    prep = preprocessing(know_train, know_test)
    know_train, know_test = prep.step_5_2_prep()

    model = Modeling(know_train, know_test)
    et_preds =  model.fit_pred_et(n_estimator=300)


    # catboost model
    know_train = [pd.read_csv(path) for path in sorted(glob('../../data_0103/train/*.csv'))]
    know_test = [pd.read_csv(path) for path in sorted(glob('../../data_0103/test/*.csv'))]

    prep = preprocessing(know_train, know_test)
    know_train, know_test = prep.step_5_2_prep()

    model = Modeling(know_train, know_test)
    cat_preds =  model.fit_pred_catboost(iteration=300)


    # 앙상블 해보기
    submit_matrix = ensembles_hard_voting(rf_preds, et_preds, cat_preds)