# data_0112에 대한 설명

출처: 
* data_0105를 가공

# DataFrame
columns = [`객관식 문항들`, `keywords`, `ubda_cnt`]

# How to Preprocess
1. 2017년 text_response들의 단어들 중 자주 등장하는 단어들을 column으로 사용
    * stop_words (ignore)  
        ```없다 증가 없음 감소 때문 수요 대한 모름 도구 사람 산업 관련 유지 업무 일자리 관심 계속 인력 발달 발전 필요 직업 분야 대체 환경 추세 사회 확대```
2. 모든 단어를 다 column으로 쓰면 너무 많으므로, 상위 150개 단어에서 자르고, stop_words는 제외함
3. text_response 중 '없음', '없다', '모름'의 개수를 `ubda_cnt`로 둠
 
# Limitation
* 상위 150개라는 임의의 숫자로 자름
* stop_words를 2017년 text_response만 보고 대충 눈대중으로 파악함
