Nodule segmentation 모델을 학습한 후, 인퍼런스를 진행하고 결과를 확인해본다.
1. Nodule CT Data 다운로드: https://luna16.grand-challenge.org/download/ 에 게시 되어있는 url 2개(https://doi.org/10.5281/zenodo.2595812 , https://doi.org/10.5281/zenodo.2596478)에 접근하여 데이터를 다운로드한다.
2. url을 통해 annotations.csv 데이터와 zip파일들을 다운로드하고 zip 파일들은 "subset_data"라는 디렉토리를 만든 후, 이 디렉토리에 압축을 푼다. (압축을 풀면 subset0, subset1, subset2 .. 파일들이 생성됨) 
3. create_image_mask.ipynb를 실행하여 preprocessed_data 디렉토리에 image, mask, lungmask npy파일들 생성한다.
4. create_dataset.ipynb를 실행하여 traindata 디렉토리에 npy파일 형태의 train, validation daata를 생성한다.
5. train.ipynb를 실행하여 학습을 진행하고 weight 디렉토리에 모델의 weight를 저장한다.
6. 학습 이후, inference.ipynb를 실행하여 validation 데이터에 대해 인퍼런스를 진행하여 성능을 평가하고 모델의 mask 예측 결과(prediction_result.npy)를 저장한다.
7. 모델의 예측 결과 및 CT 이미지에 mask를 적용한 nodule segmentation 결과를 보고 싶으면 visualize_result.ipynb를 실행한다.
*순서: create_image_mask.ipynb > create_dataset.ipynb > train.ipynb > inference.ipynb > visualize_result.ipynb