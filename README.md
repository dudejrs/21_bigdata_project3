

1. 설치 과정


1. 실행 방법
```sh
streamlit run main.py

```


1. Data size error
```
RuntimeError: Data of size 145.4MB exceeds write limit of 100.0MB
```
	- 위와 같은 에러 발생시 다음과 같은 순서를 거친다
		1. 깔려 있는 streamlit 패키지의 위치를 찾는다
			```sh
			> python -c "import streamlit as st; print(st.__path__)"
			
			```
		2. 패키지 내의 server/server_util.py를 수정해정해준다
			```py
			# MESSAGE_SIZE_LIMIT = 50 * int(1e6)  # 50MB
			MESSAGE_SIZE_LIMIT = 100 * int(1e6)  # 100MB
			
			```
		3. 다시 steramlit를 다시 시작해준다# 21_bigdata_project3