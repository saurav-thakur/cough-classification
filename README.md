# CSEE 903 Group Project 

# Project Title: Noise Robust Cough Sound Segmentation Using Audio Features and Machine Learning
- Project ID: 78427; 
- Supervisor: Roneel Sharan;
- Contact: roneel.sharan@essex.ac.uk

**Team Name:**
- Sono

**Project Documentation**
- [Google Drive](https://drive.google.com/drive/folders/1vPHf0wGuo_vNYT0-DNQPbkM_VB-gTpfA?usp=drive_link)
- [Requirements Specification](https://drive.google.com/file/d/1BIs8jz5bGKnLxz9eX3ZVy8V1eAT_6JXx/view?usp=drive_link)
- [Source code](https://cseegit.essex.ac.uk/23-24-ce903-su/23-24_CE903-SU_team06.git)

**Student registration number(s):**
- Warren Martin - Team Leader: 232569

- Breanne Felton: 2321566
- RANSFORD OWUSU: 2321357
- Saurav Thakur: 2322997
- Patrick Ogbuitepu: 2320824
- Max Garry: 2323118

**Project Description:**
Cough is a common symptom of respiratory diseases and the sound of cough can be indicative of the respiratory condition. Objective cough sound analysis has the potential to aid the assessment of the respiratory condition. Cough sound segmentation is an important step in objective cough sound analysis. This project focuses on developing methods to segment cough sounds from audio recordings. The students will explore various signal processing techniques, such temporal and spectral analysis, to identify distinctive features of coughs. Through machine learning algorithms, they'll create a classification model capable of distinguishing cough sounds from non-cough sounds. The students will also develop cough sound denoising algorithms for cough sound segmentation in noisy environments. Refer to https://doi.org/10.1016/j.bspc.2023.104580 and https://doi.org/10.1109/EMBC40787.2023.10340687 for more information.

# Setup and Running the Project Locally

**Prerequisites**

Make sure you have [Anaconda](https://www.anaconda.com/download) installed on your machine.

### Environment Setup

**Linux or Git Bash on Windows**

To create a conda environment with Python 3.10.0 and a virtual environment using poetry, run:


```
source setup.sh
```



**Windows CMD**

If you are on windowns terminal, run each command line by line:



```
conda create --prefix ./venv-team06 python=3.10.0 -y

conda activate ./venv-team06/

pip install poetry

poetry config virtualenvs.in-project true

poetry install

poetry shell

```

# Running the Project

After setting up the environment. To run the project:

**Using the API**

```
python app.py
```

Open a browser and navigate to:

```
http://localhost:8080/docs
```

There are two APIs available:

1. **Training API**: Use this to train the model.

2. **Predict API**: Use this to upload a raw audio file and segment the cough sounds.


If you don't want to use api. You can run:

```
python main.py
```


<!-- # Getting Started
1. [Serverless Environment Setup with Gitlab, Google Drive & Colab](./docs/Environment Setup & Version Control.pdf)
2. Demo -->


