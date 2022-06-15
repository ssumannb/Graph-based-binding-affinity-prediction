# HD011
Project name : HD011

## Description
본 프로젝트는 단백질-리간드 복합체의 binding affinity를 예측하는 인공지능 기반 모델을 개발하는 프로젝트입니다.
해당 모델은 신약 개발 단계 중 초기 단계인 Hit discovery에서 Target 단백질에 대해 대규모 화합물 라이브러리를 가상 스크리닝을 위해 사용할 수 있습니다.

### 1. model input
입력 데이터로 사용되는 데이터는 단백질-리간드 복합체의 구조 정보가 포함된 데이터로, 복합체의 binding pose를 얻기 위한 사전 Docking 작업이 필요합니다.
입력 형태는 단백질 데이터의 경우 pdb format, 리간드 데이터의 경우 sdf 또는 mol2 format의 데이터를 사용합니다.



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

## Files

각 파일들의 역할

## Usage

작성한 코드 실행 가이드라인
