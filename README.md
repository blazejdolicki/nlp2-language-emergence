# Visual concepts pressure in language emergence
This repository contains PyTorch code for the short paper "Visual concepts pressure in language emergence". 

## Installation
Create a new conda environment with Python 3.7 and activate it.
```
conda create -n le-nlp2 python=3.7
conda activate le-nlp2
```
Install PyTorch and related libraries.
```
conda install pytorch=1.8.1 torchvision=0.9.1 torchaudio=0.8.1 cudatoolkit=10.2 -c pytorch
```
Install other packages.
```
pip install -r requirements.txt
```

Install EGG library from Github repository (commit `ba7ba8f`).
```
pip install git+https://github.com/facebookresearch/EGG.git@ba7ba8f
```
## Running experiments
It is possible to run 3 kinds of tasks: the `standard` signaling game with a single loss (baseline) and two tasks with additional visual pressures. You should run all experiments from the project root directory.
### 1. Baseline
Run the following code to train and evaluate the baseline model.
```
python main.py
```
### 2. Multilabel binary image classification pressure
As our first pressure, the system additionally predicts for each image (distractors and target image) whether it is of the same class as the target image. The total loss is the loss for that task summed with the standard signaling game loss.
```
python main.py --task img_clas
```
### 3. Multiclass image classification pressure
As the second pressure, they system additionally predicts the target class of the target image. The total loss is the loss for that task summed with the standard signaling game loss.
```
python main.py --task target_clas
```
