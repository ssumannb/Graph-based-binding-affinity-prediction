# HD011
Project name : HD011  

## Description
This project is the development project that AI-based binding affinity prediction model for the protein-ligand complex.  
This prediction model can be used for virtual screening of large chemical libraries for target proteins in a hit discovery step that is early-stage on drug discovery.  

## Environment
- OS : window  
- GPU : NVIDIA GeForce RTX 3080 Ti  
- CPU : i9-11900  
- RAM : 32GB  
- Language : Python 3  
- Framework : Pytorch 
- CUDA 11.3 cudnn 8.0


## Prerequisite
- python 3.8.12  
- pytorch nightly 1.11.0  
- rdkit 2021.09.4  
- deepchem 2.6.1  
- dgl-cuda11.3 0.7.2   
- mlflow 1.22.0  
- numpy 1.20.3  
- pandas 1.3.5  
- sklearn 1.0.1  
- scipy 1.6.2  


## Details
#### Model input
입력 데이터로 사용되는 데이터는 단백질-리간드 복합체의 구조 정보가 포함된 데이터로, 복합체의 binding pose를 얻기 위한 사전 Docking 작업이 필요합니다.  
입력 형태는 단백질 데이터의 경우 pdb format, 리간드 데이터의 경우 sdf 또는 mol2 format의 데이터를 사용합니다.  
#### Model architecture 
예측 모델은 그래프 기반의 딥러닝 회귀 모델입니다.  
- 입력 데이터 전처리를 진행하는 Graph converting part  
- 변환된 그래프의 특징을 학습하는 Graph learning layers  
- 학습된 그래프 특징에서 binding affinity 값을 예측하는 Affinity calculation layers  

![image](https://user-images.githubusercontent.com/86610517/173760137-10cfff36-dd2d-4e1d-9caa-51ef6a588346.png)


##### 1. Graph converting part
단백질/리간드 데이터를 atom-level에서 graph로 각각 변환하는 작업을 진행합니다.
단백질의 경우 알려진 pocket site를 활용하여 리간드를 구성하는 원자와의 거리가 5A 이하인 원자를 포함하는 잔기들만 단백질 그래프로 변환합니다.
리간드 물질의 경우 구성하고 있는 모든 원자들을 그래프로 변환합니다.

##### 2. Graph learning part  
단백질/리간드 그래프의 특징을 그래프 기반 레이어를 통해 학습하는 작업을 진행합니다.  
변환된 단백질/리간드 그래프는 선형 임베딩을 거쳐 각각 네트워크로 전달되어 학습됩니다.  
- 단백질 그래프는 Graph-convolution layer(GCN)로 구성된 네트워크를 통해 학습됩니다.  
- 리간드 그래프는 Graph-attention layer(GAT)로 구성된 네트워크를 통해 학습됩니다.  

##### 3. Affinity calculation part
단백질/리간드 특징으로 fully-connected 레이어를 통해 binding affinity를 예측하는 작업을 진행합니다.  
학습된 각각의 그래프 특징들은 벡터화되어 summation 후 fully-connected layer에 전달됩니다.
마지막 layer의 output이 단백질-리간드 복합체의 binding aiffinity 값 입니다.  

## Files
##### 📁 DB
: DB folder includes codes related to data    
##### 📁 MD
: MD folder includes all experiment codes about prediction model  
- 📁 libs : library files for experiment  
- train.py : files for training  
- test.py : files for testing  
- (prediction.py) : files for predicting   


## Usage
작성한 코드 실행 가이드라인
