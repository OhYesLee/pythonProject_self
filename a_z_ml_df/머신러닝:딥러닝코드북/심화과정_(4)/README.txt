기존에 존재하는 'bipedwalker'라는 환경에서 학습이 진행될수록 다리의 폭 및 길이가 계속 변화하면서 학습이 되는 환경을 numpy로 구성하여 customized된 rl환경을 만들고 이 환경과 ray를 연동하여 학습을 진행한다.
1.ray, gym 등 설치 환경 세팅(ray의 경우, 0.8.7 ver)
2. 커맨드 환경에서 파일에 접근하여 "python train_agent.py augmentbipedsmalllegs -n 숫자 -e 숫자 -t 숫자"를 입력하여 agent의 학습을 진행하면 saved_model 디렉토리에 모델이 저장된다.여기서 augmentbipedsmalllegs는 customized된 rl환경을 의미하고 -n은 number of worker, -e은 number of episodes per trial, -t는 number of trials per worker를 의미한다.
ex) python train_agent.py augmentbipedsmalllegs -n 2 -e 2 -t 2
3. 학습 이후, "python rl_model.py( or rl_model_gui.py) augmentbipedsmalllegs ./saved_model/(생성된 모델명).json를 입력하여 학습하여 생성된 모델에 대하여 인퍼런스를 진행한다. 여기서 augmentbipedsmalllegs는 customized된 rl환경을 의미하고 gui로 보는 기능없이 reward를 확인하고 싶으면 rl_model.py를 사용하고 gui 기능도 켜서 인퍼런스 결과를 확인하고 싶으면 rl_model_gui.py를 사용한다.
ex) python rl_model.py augmentbipedsmalllegs ./saved_model/augmentbipedsmalllegs.pepg.2.4_100_-38.46.json