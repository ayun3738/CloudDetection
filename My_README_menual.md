

# ☁️ 구름 패턴 분류 모델 ☁️
<div align="center">

<img src="doc/Kor_pred2023-04-20(09).png" width="600" height="400"> 

</div>
</br>
위성사진에서 강수확률과 관련된 구름을 디텍팅하여 기상예보 분석을 보조해주는 딥러닝 모델 설계 및 개발 레포입니다.

## 👨🏿‍🤝‍👨🏿Member
[노아윤](https://github.com/ayun3738) | [유미리](https://github.com/Yu-Miri) |[이기준](https://github.com/gijun0725)
:-: | :-: | :-: 
<a href="https://github.com/ayun3738"><img src="https://avatars.githubusercontent.com/u/96457781?v=4" width="100" height="100"/></a>|<a href="https://github.com/Yu-Miri"><img src="https://avatars.githubusercontent.com/u/121469490?v=4" width="100" height="100" ></a>|<a href="https://github.com/suted2"><img src="https://avatars.githubusercontent.com/u/119472512?v=4" width="100" height="100" ></a>

## 📋Index
- [📝 Project Summary](#📝project-summary)
- [👀 데이터셋 ](#👀-데이터셋 )
- [⚙️ Modeling](#⚙️-modeling)
- [🔍 Conclusion](#🔍-conclusion)
- [⚒️ Appendix](#⚒️-appendix)

## 📝Project Summary
- 개요
  > 위성사진에서 구름의 패턴을 빠르게 디텍팅하여 기상예보 분석을 보조해줄 수 있는 딥러닝 모델 설계
- 목적 및 배경
![단기예보관](doc/배경1.png)
    > 기상청에서는 AI 인공지능을 이용한 수치형 table 데이터 통계모델을 사용하는 것으로 알려져 있습니다. 하지만 초단기 일기예보는 10분마다 들어오는 자료들을 분석해야합니다. 자동화된 자료들 사이에서도 결국 종합적인 분석은 예보관이 합니다.

    출처 : https://www.youtube.com/watch?v=bMpBXqjuGlI

<div align="center">

<img src="doc/sugar1.png" width="300" height="200"> 
<img src="doc/flower1.png" width="300" height="200"> 
</div>
    
  > 위성사진 상에서 구름의 형태에 따라 강수량에 영향을 줄 수 있다는 보고와 함께 정성적으로 라벨링된 데이터셋을 발견하게 됐습니다. 이 데이터셋을 활용하여 Object Detection 모델을 학습시킨다면, 구름의 형태에 따라 강수량의 유추가 가능할 수 있게 구름의 모양을 실시간으로 디텍팅할 수 있습니다.

  출처 : https://rmets.onlinelibrary.wiley.com/doi/full/10.1002/qj.3662

 💡 우리의 AI 모델을 사용한다면,
  1. 예보관에게 가시적으로 상황을 알려줄 수 있습니다. 경력이 상당한 예보관이나 기상학자분들은 바로 구름형태를 보고 빠른 판단을 할 수 있겠지만, 신입 예보관 등의 상황에 지침으로 구름의 형태를 실시간 모니터링하며 변하는 모습과 흐르는 방향을 볼 수 있습니다.
  2. 구간별 지역별 디텍팅 카운팅이나 디텍팅 확률 등을 수치형 table 데이터의 자료로 추가 제공하여 통계모델을 더 깊이 있게 활용하는 보조로서의 가치가 있을 것이라 생각했습니다.
  
- 모델 설계
  > 위성사진을 input 이미지로 받았을 때, 가능성 있는 위치에 어떤 패턴인지 알아야 하기 때문에 Object Detection이나 Segmentation 기법을 이용해 구름 패턴을 파악하기로 설계했습니다. Object Detection이 통상적으로 잘 되고 튜닝이 용이한 YOLO의 버전들을 확인하며 모델을 결정하기로 했습니다. 가시적으로 패턴을 잘 디텍팅하는지 확인하면서, mAP를 기준으로 모델을 선정했습니다.

- 활용 장비 및 재료
  - 라이브러리 : pytorch, sklearn, OpenCV, roboflow, ultralytics
  - 개발 및 협업 툴 : python, colab notebook, vscode(windows)

## 👀 데이터셋 

초기 모델 설계 과정에서 결정한 YOLO를 학습시키기 위해 라벨링된 위성사진이 필요합니다.

### NASA WORLD VIEW
- 출처 : [cloud types Computer Vision Project](https://universe.roboflow.com/roboflow-100/cloud-types)
- 소개 : 2100 x 1400의 5050장의 나사에서 특정 위치에서 지속적으로 찍은 인공위성 사진입니다. 인공위성 사진이다 보니 이미지 노이즈로 태양빛이나 인공위성 지지대가 사진에 일부 포함되어 있습니다. 라벨당 2000장 이상의 충분한 데이터양이라고 판단하고 학습에 이용하기로 했습니다.
- 라벨 : 

<div align="center">
<img src="doc/label_sugar.png"> 
</div>

  2. 따라서 판결문 200개에 대한 문장 조합쌍을 python code를 통해 테이블형식 데이터로 구성했습니다.
  3. 각 문장 조합쌍에 대해 SBERT에 통과시켜 문장 유사도를 Auto labeling하여 데이터셋을 구상했습니다.
  
- 전체 문장 쌍 개수 : 약 125,000

### KLUE 데이터셋
- 출처 : [KorNLI, KorSTS(카카오 브레인 깃허브)](https://github.com/kakaobrain/kor-nlu-datasets)
- 소개 : 영어로 먼저 문장간의 유사도를 라벨링한 데이터를 한국어로 번역하여 데이터를 재구성한 데이터셋
- 라벨 : 
1. KorNLI(Kor Natural Language Inference) : 기존에 영어로 구성됐던 NLI 데이터셋의 문장을 한글로 번역한 데이터. 문장 Premise(전제), Hypothesis(가설)에 대해 **Entailment(일치하는 문장)**, **Contradiction(반대의미 문장)**, **Neutral(애매한 문장)** 의 3class로 라벨링되어 있습니다.

| Example                                                      | English Translation                                          | Label         |
| ------------------------------------------------------------ | ------------------------------------------------------------ | ------------- |
| P: 저는, 그냥 알아내려고 거기 있었어요.<br />H: 이해하려고 노력하고 있었어요. | I was just there just trying to figure it out.<br />I was trying to understand. | Entailment    |
| P: 저는, 그냥 알아내려고 거기 있었어요.<br />H: 나는 처음부터 그것을 잘 이해했다. | I was just there just trying to figure it out.<br />I understood it well from the beginning. | Contradiction |
| P: 저는, 그냥 알아내려고 거기 있었어요.<br />H: 나는 돈이 어디로 갔는지 이해하려고 했어요. | I was just there just trying to figure it out.<br />I was trying to understand where the money went. | Neutral       |
  
2. KorSTS(Kor Semantic textual similarity) : 기존에 영어로 구성됐던 STS 데이터셋의 문장을 한글로 번역한 데이터. 두 문장에 대해 **0(관련 없음) ~ 5(문장의미가 일치)** 로 사람이 스코어를 메긴 점수로 라벨링이 되어 있습니다.

| Example                                                      | English Translation                                      | Label |
| ------------------------------------------------------------ | -------------------------------------------------------- | ----- |
| 한 남자가 음식을 먹고 있다.<br />한 남자가 뭔가를 먹고 있다. | A man is eating food.<br />A man is eating something.    | 4.2   |
| 한 비행기가 착륙하고 있다.<br />애니메이션화된 비행기 하나가 착륙하고 있다. | A plane is landing.<br />A animated airplane is landing. | 2.8   |
| 한 여성이 고기를 요리하고 있다.<br />한 남자가 말하고 있다. | A woman is cooking meat.<br />A man is speaking.      | 0.0   |

## ⚙️ Modeling

참고 깃허브 : https://github.com/jhgan00/ko-sentence-transformers

### 1. 데이터 전처리(KULE 데이터)

>- KorNLI : KorNLI 데이터에서 일치하는 문장 두개와 무작위의 반대의미 문장까지 3가지 문장쌍을 데이터셋으로 구성하는 Pairing 방식을 사용하여 구상했습니다.
<div align="center">
<img src="doc/KLUE3.png"> 
</div>

>- KorSTS : KorSTS 데이터에서 0~5의 점수를 **0 ~ 1**으로 normalization 했습니다.

<div align="center">
<img src="doc/KLUE1.png"> 
</div>

### 2. 데이터 전처리(크롤링 custom 데이터)

>- casenote에서 크롤링한 판례문들의 판례문에서 '이유' 부분의 텍스트가 판례의 내용을 담고 있다고 판단하여 그부분을 .csv 파일로 저장했습니다,
<div align="center">

<img src="doc/크롤링데이터1.png" width="450" height="250"> 
<img src="doc/크롤링데이터2.png" width="200" height="300">

</div>

>- 이후 한글 외의 문자, 숫자를 제거하고, 데이터 특성상 판결문의 글머리(ex. 가, 나, 다)와 형식적 단어(ex. 사건개요) 등의 불용어 제거했습니다.

<div align="center">
<img src="doc/크롤링데이터3.png"> 
</div>

>- 그 뒤 하나의 판결문에 대해 문장들을 분리하여 각각의 cosine 유사도를 포함한 데이터쌍을 구상하여 KorSTS에서의 값들처럼 **0 ~ 1**사이의 값으로 구성했습니다.

<div align="center">
<img src="doc/KLUE2.png"> 
</div>

>- 기존 참고한 Sentence BERT 학습 코드에서 판결문 관련 custom 데이터를 추가함으로 유사 판결문을 더 잘 판별할 것으로 기대하였고, transformer도 바꿔가며 학습을 진행했습니다.
>- 트레이닝 코드 : [training_last.py](training_last.py), [data_util.py](data_util.py)

## 🔍 Conclusion

### Inference

#### 1. 문장 스코어 자체의 성능

> 기존 reference로 학습된 open model인 SBERT를 SBERT-0로 잡고 학습시킨 모델들을 비교하기로 했습니다.

<div align="center">
<img src="doc/con1.png"> 
</div>

> - 실제 사례나 판례의 유사한 두 문장을 입력했을 때, 유사도의 성능을 측정한 결과 높게 측정되었습니다.

<div align="center">
<img src="doc/con2.png"> 
</div>

> - 실제 사례나 판례의 엉뚱한 두 문장을 입력했을 때, 유사도의 성능을 측정한 결과 낮게 측정되었습니다.
- SBERT-0의 경우, 유사한 문장의 스코어가 애매한 0.5 부근으로 측정되어 적절한 모델이 아님으로 생각됐습니다.
- SBERT-1의 경우, 엉뚱한 문장의 스코어가 애매한 0.5 부근으로 측정되어 적절한 모델이 아님으로 생각됐습니다.
- SBERT-2의 경우, 유사한 문장과 엉뚱한 문장의 스코어가 둘 다 높은 수준으로 나와 적절한 모델이 아님으로 생각됐습니다.
- 하지만 데이터를 늘린 SBERT-3의 경우, 유사한 문장과 엉뚱한 문장의 스코어가 어느정도 구분 가능한 수준으로 나와 적합한 방식의 학습이 진행됐다고 판단했습니다.

#### 2. 실제 사례를 통한 유사 판결문 추천

- 실제 상담사례를 가져와 넣었을 때, 의료판결문 중에서도 백내장 관련 판례를 가장 높은 유사성 판례로 추천해주는 것을 확인했습니다.

<div align="center">
<img src="doc/infer1.png"> 
</div>

### Conclusion & Extension
1. 생성형 NLP에서 대표적인 BERT모델의 문장유사도를 통해 판결문의 각 문장과 연결지어 전체 판결문과의 유사도를 구하는 유사판례문 추천모델을 학습시켜 동작함을 확인했습니다.
하지만 아직 그저 문장간 평균을 내거나, 유사 문장의 최대값으로 살펴본다던지하는 단순한 알고리즘으로 판결문을 도출합니다.
-> 모델의 output에서 가장 유사한 판결문을 판단하는 알고리즘을 개선시켜 좋은 시스템으로 발전시킬 수 있습니다.
2. NLP 모델의 특성상 task에 잘 맞는 정제된 많은 문장이 필요합니다. 하지만 사이트 및 판결문마다 작성된 구조가 달라서 시간상 크롤링 데이터를 많이 담지 못한 점이 모델 성능의 아쉬운 점이라고 생각됩니다. 또한 데이터셋 생성 과정에서 cosine 유사도를 통한 labeling을 진행했는데, 이의 타당성을 검증하는 과정도 필요하다고 생각됩니다.
3. 판결문 200개에서 문장 조합이 125000개로 불어나면서 많은 판례문을 타겟으로 삼을수 없어 우선 분야를 '의료'분야의 판례문만 가지고 프로젝트를 진행했지만, 이러한 방식으로 다른 분야에 맞는 여러 모델을 통해 판결문 추천도 가능할 것입니다.

## ⚒️ Appendix


| Reference | Git | Paper |
| ---- | ---- | ---- |
| SBERT 학습 코드 | https://github.com/jhgan00/ko-sentence-transformers| <a href='https://arxiv.org/abs/1908.10084'><img src='https://img.shields.io/badge/ArXiv-PDF-red'></a> &nbsp;
|  |  | <a href='https://arxiv.org/abs/2004.03289'><img src='https://img.shields.io/badge/ArXiv-PDF-red'></a> &nbsp;