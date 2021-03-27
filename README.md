# Visual concepts pressure in language emergence
This repository contains PyTorch code for the short paper "Visual concepts pressure in language emergence". 

## Installation
**TODO**: Add installation details
Install EGG library from Github repository (commit `ba7ba8f`)
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
