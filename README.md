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

#### Language & Framework
- python 3.8.12
- pytorch nightly 1.11.0

#### Prerequisite Library
  ###### static related library
  - numpy 1.20.3  
  - pandas 1.3.5  
  - sklearn 1.0.1  
  - scipy 1.6.2  
  ###### Chemistry, graph related library
  - rdkit 2021.09.4  
  - deepchem 2.6.1  
  - dgl-cuda11.3 0.7.2   
  ###### etc
  - mlflow 1.22.0  



## Materials & Methods
#### Dataset
PDBbind v2020 dataset(n=19,443) : Protein-Ligand complex data 
- general dataset(n=19,443) : Biomolecular complex dataset deposited in the PDBbank which has experimentally measured binding affinity data _(Kd, Ki, or IC50)_ 
- refined dataset(n=5,316) : Dataset is compiled to select the protein-ligand complexes with better quality out of the general set   

__data exculsion criteria__
- Excluding the unavailable complexes which had no file or empty file (n=353)
- Excluding the complexes only with IC50 affinity values (n=7,135)  

__data splitting__
- training : 8,563 complexes / validation : 2,447 complexes / test : 1,223 compelxes  


#### Model architecture 
The prediction model is the deep learning regression model based on the graph.  
- The __Graph converting part__ preprocesses the input data  
- The __Graph learning part__ learns features from converted graphs  
- The __Affinity calculation part__ predict the binding affintiy using graph features  

<p align="center"><img src="https://user-images.githubusercontent.com/86610517/178200612-add1ec53-60c5-45fb-a658-5eddd1f77221.png" height="70%" width="70%">
<br>
<em>Model architecture</em>
</p>  


#### Model input
Input data is the data including the structure information of the protein-ligand complex. The docking has to be preprocessed on each protein-ligand pair to obtain the binding pose of the complex. The input format must be PDB format in case of the protein data and SDF or MOL2 format in case of the ligand.  

#### 1. Graph converting part
단백질/리간드 데이터를 atom-level에서 graph로 각각 변환하는 작업을 진행합니다.  
단백질의 경우 알려진 pocket site를 활용하여 리간드를 구성하는 원자와의 거리가 5A 이하인 원자를 포함하는 잔기들만 단백질 그래프로 변환합니다.  
리간드 물질의 경우 구성하고 있는 모든 원자들을 그래프로 변환합니다.  
In the graph converting part, the proteins and ligands are converted to graphs, respectively. In the case of the ligands, all atoms are converted to graphs. On the other hand, in the case of the proteins, only the selected residues are converted to graphs with the known pocket site. The criteria of residue selection are whether the residues include the atoms with a near distance (5A) from the atoms of ligands. 

 

<p align="center">
<img src="https://user-images.githubusercontent.com/86610517/177702493-3b2d755f-cd2a-460e-b115-909845410949.png" height="55%" width="55%" alt><br>
<em>Steps for converting a molecule to the graph</em><br>
</p><br>  
  
   
<p align="center">
<img src="https://user-images.githubusercontent.com/86610517/177702430-dac69fbf-de7b-4b36-98ac-4f1e3238165b.png" height="55%" width="55%" alt><br>
<em>Steps for converting a protein to the graph</em><br>
</p>  



#### 2. Graph learning part  
단백질/리간드 그래프의 특징을 그래프 기반 레이어를 통해 학습하는 작업을 진행합니다.  
변환된 단백질/리간드 그래프는 선형 임베딩을 거쳐 각각 네트워크로 전달되어 학습됩니다.  
- 단백질 그래프는 Graph-convolution layer(GCN)로 구성된 네트워크를 통해 학습됩니다.  
- 리간드 그래프는 Graph-attention layer(GAT)로 구성된 네트워크를 통해 학습됩니다.  

In the graph learning part, the each feature set of proteins and ligands are learned by the graph based layers. The features of the protein graphs are learned from the GCN (graph convolutional network) and the features of the ligands are learned from the GAT (graph-attention network).  

#### 3. Affinity calculation part
단백질/리간드 특징으로 fully-connected 레이어를 통해 binding affinity를 예측하는 작업을 진행합니다.  
학습된 각각의 그래프 특징들은 벡터화되어 summation 후 fully-connected layer에 전달됩니다.  
마지막 layer의 output이 단백질-리간드 복합체의 binding aiffinity 값 입니다.  
In the affinity calculation part, the binding affinity value of the input complex is calculated by fully connected layer. The each of features of protein and ligand are transformed to vector, respectively and then vector summation is performed on each vector. The summation vector of each graph is passed to the fully connected layers and the binding affinity prediction is performed.


## Files
🗃 __DB__ : including codes to load dataset to database, to filter the available dataset and to preprocess dataset into input of model    

- 📁 libs 
   - db_utils.py : utilities for connecting to the database and basic CRUD functions for PostgreSQL
   - filtering_utils.py : utilities for filtering the data according to the exclusion criteria
   - preprocess_utils.py : utilities for converting the protein and ligand to the graph  
  
- 📁 query : query files  

- 📁 Reports : outputs of data preprocessing  

- convert_graphs.py : converting the dataset loaded on postgreSQL to graph using preprocess_utils.py  
- filter_available.py : filtering the pdbbind and coreset data using filtering_utils.py  
- load_coreset2pgSQL.py : loading the coreset data to postgreSQL using db_utils.py   
- ~~load_INDEXfile2pgSQL.py :loading the pdbbind data to postgreSQL using db_utils.py~~ (데이터 정보 오류/데이터센터에 문의한 상태)  
- ~~load_pdbbind2neo4j.py : loading the pdbbind data to neo4j using db_utils.py~~  
- load_pdbbind2pgSQL.py : loading the pdbbind data to postgreSQL using db_utils.py  
  
  
  
🗃 __MD__ : including codes to construct the model and to run experiments and the including all outputs about the experiments   

- 📁 libs  
    - earlystop.py : earlystop class for training the model  
    - io_utils.py : utilities for input of the model  
    - layers.py : layer class for configuring the model layer  
    - models.py : model class for configuering the model architecture  
    - utils.py : utilities for training    
  
- 📁 mlruns : experiment log files managed by using mlflow   
- 📁 runs : summary of training logs for visualizing the loss graphs by tensorboard  

- train.py : files for training  
- prediction.py : files for predicting   


## Results
- loss graphs  _(loss function : MSE loss)_
  
![Loss_Train](https://user-images.githubusercontent.com/86610517/178211702-88d56c6e-511d-497e-b4c3-45dd79591388.png) 
<br><em>Train loss graphs</em><br><br>
![Loss_Valid](https://user-images.githubusercontent.com/86610517/178211713-e63448c4-9e5a-4700-aacd-d3d7809eefa8.png)
<br><em>Valid loss graphs</em><br>


- Test results  
![image](https://user-images.githubusercontent.com/86610517/178212913-765a16f8-8f63-4eb1-b30c-3571d4a13ab0.png)
