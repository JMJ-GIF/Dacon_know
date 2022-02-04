# Preprocess Ideation
1. Normalization
    * 개인마다 문항에 주는 점수가 다르다
        ex. 극단적으로 주는 사람 vs '보통'을 많이 주는 사람
    * 점수를 개인별로 정규화 (개인 별 min-max)
        - A: 5 5 1 1 1 -> 1 1 0 0 0
        - B: 3 3 2 2 2 -> 1 1 0 0 0

2. Text Preprocessing
    * 원래 주관식 답변인 문항들 중 중요한 문항(자격증, 전공 등)들은 전처리해서 model에 넣어야 한다.
    * 자격증과 전공에서 keyword를 추출 (ex. 건축, 토목, 컴퓨터, 일본어 등)
    * keyword를 category화해서 feature로 사용

3. Data Augmentation
    * class 별로 data가 불균형하다(대부분 15개)
    * masking을 하거나, 복붙을 해서 class 별로 개수를 늘려 데이터 수를 더 확보한다
    * Neural Net을 쓰려면 이런 식으로 데이터 양을 더 늘려야 할 것 같음