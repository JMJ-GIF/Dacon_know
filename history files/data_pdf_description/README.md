# data_pdf_description에 대한 설명

출처: 
* KNOW METADATA 내부에 있는 {YEAR}_변수값.pdf에서 knowcode에 대한 설명만 읽어와 만든 연도별 데이터 프레임
*  ".\minje\pyfile\preprocessing_3)_get_knowcode_description.py"으로 생성가능함

# DataFrame
columns = [`knowcode`, `description`]

# How to Preprocess
1. pdf reader로 데이터를 모두 읽어와 가공함
    * print해서 손수 오류를 수정함

 
# Limitation
* pdf의 자체적인 오류로서, 하나의 knowcode에 2개의 직업이 매칭되는 경우가 있었음 -> __하나의 셀에 2개의 직업을 합쳐놓았음(이때 sep=','임에 유의)__
* train data의 knowcode와 변수값.pdf 내에 있는 knowcode와 매칭이 되지 않는 경우가 있었음 -> merge시 nan값이 있을수 있으니 유의할 것
* 위 2가지 오류가 있으니 merge시 how = (`left` or `right`) 옵션을 반드시 줄 것