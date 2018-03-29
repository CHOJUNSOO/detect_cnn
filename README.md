# detect_cnn

1.요약
입력 받은 이미지에서 얼굴을 찾고서 해당 얼굴의 성별, 나이를 판별하는 프로그램

2.요구조건
1)dlib
	http://dlib.net/ 에서 최신버전 다운로드 후
	python setup.py install --yes USE_AVX_INSTRUCTIONS --yes DLIB_USE_CUDA
	위 명령어를 통해 cuda를 활성화한 dlib 설치
2)Opencv
	3.4.0이상 버전 권장
3)Caffe
4)Python2.7


3.구조 - 폴더는 초기생성이 필요
1)face폴더 = 판별이 필요한 대상 이미지가 들어가는 폴더
2)trim폴더 = 추출된 얼굴 이미지가 저장되는 폴더
	trim폴더는 Female, Male를 하위폴더로 가짐
		각 성별 폴더는 (0, 2), (4, 6), (8, 12), (15, 20), (25, 32), (38, 43), (48, 53), (60, 100)를 하위폴더로 가짐
3)valid폴더 = 얼굴이 존재한다고 판별된 이미지가 카피되어 저장
4,5)age_net.caffemodel, gender_net.caffemodel
	하단 링크의 파일을 미리 훈련된 모델로 사용함
	https://www.openu.ac.il/home/hassner/projects/cnn_agegender/cnn_age_gender_models_and_data.0.0.2.zip
6)mmod_human_face_detector.dat
	cnn face detection을 사용하기위해서는 하단 링크의 파일 사용
	http://dlib.net/files/mmod_human_face_detector.dat.bz2
7,8)deploy_age.prototxt, deploy_gender.prototxt
9)mean.binaryproto
10)detect_cnn.py
	python detect_cnn.py 로 실행
