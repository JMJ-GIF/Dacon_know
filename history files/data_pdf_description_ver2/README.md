# data_pdf_description_ver2에 대한 설명

* ver1에서 있는 error를 찾아 조금 수정함
    - 중복행에 대해서 drop_duplicates()를 시행하여 행을 줄임
    - 하나의 코드에 2개의 직업이 매칭되는 셀을 표현하기 위해 error라는 열을 추가함
        * error = 1은 2개의 직업이 매칭됨을 의미