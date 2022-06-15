# HD011
Project name : HD011  

## Description
<span style="font-size:50%">본 프로젝트는 단백질-리간드 복합체의 binding affinity를 예측하는 인공지능 기반 모델을 개발하는 프로젝트입니다.  
해당 모델은 신약 개발 단계 중 초기 단계인 Hit discovery에서 Target 단백질에 대해 대규모 화합물 라이브러리를 가상 스크리닝을 위해 사용할 수 있습니다.  </span>

## Environment
OS : window  
GPU : NVIDIA GeForce RTX 3080 Ti  
CPU : i9-11900  
RAM : 32GB  
CUDA : 11.3  
Language : Python 3  
Framework : Pytorch 1.10.1  

## Prerequisite
numpy, pandas, rdkit, deepchem, dgl, sklearn  


## Details
#### 1. model input
입력 데이터로 사용되는 데이터는 단백질-리간드 복합체의 구조 정보가 포함된 데이터로, 복합체의 binding pose를 얻기 위한 사전 Docking 작업이 필요합니다.  
입력 형태는 단백질 데이터의 경우 pdb format, 리간드 데이터의 경우 sdf 또는 mol2 format의 데이터를 사용합니다.  
#### 2. model architecture 
예측 모델은 그래프 기반의 딥러닝 회귀 모델입니다.  

**모델 구성**  
- 입력 데이터 전처리를 진행하는 Graph converting part  
- 변환된 그래프의 특징을 학습하는 Graph learning layers  
- 학습된 그래프 특징에서 binding affinity 값을 예측하는 Affinity calculation layers  

![image](https://user-images.githubusercontent.com/86610517/173760137-10cfff36-dd2d-4e1d-9caa-51ef6a588346.png)


##### 2-1. Graph converting part
단백질/리간드 데이터를 atom-level에서 graph로 각각 변환하는 작업을 진행합니다.
단백질의 경우 알려진 pocket site를 활용하여 리간드를 구성하는 원자와의 거리가 5A 이하인 원자를 포함하는 잔기들만 단백질 그래프로 변환합니다.
리간드 물질의 경우 구성하고 있는 모든 원자들을 그래프로 변환합니다.

##### 2-2. Graph learning part  
단백질/리간드 그래프의 특징을 그래프 기반 레이어를 통해 학습하는 작업을 진행합니다.  
변환된 단백질/리간드 그래프는 각각 네트워크로 전달되어 학습됩니다.  
- 단백질 그래프는 Graph-convolution layer(GCN)로 구성된 네트워크를 통해 학습됩니다.  
- 리간드 그래프는 Graph-attention layer(GAT)로 구성된 네트워크를 통해 학습됩니다.  

##### 2-3. Affinity calculation part
단백질/리간드 특징으로 fully-connected 레이어를 통해 binding affinity를 예측하는 작업을 진행합니다.  
학습된 각각의 그래프 특징들은 벡터화되어 summation 후 fully-connected layer에 전달됩니다.
마지막 layer의 output이 단백질-리간드 복합체의 binding aiffinity 값 입니다.  

## Files
각 파일들의 역할

## Usage
작성한 코드 실행 가이드라인
