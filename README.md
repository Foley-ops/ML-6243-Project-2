# ML Tournament Project 2

Authors: Nicholas Foley, Logan Robinson

Due Date: December 5, 2025

## Overview


This project is a PyTorch-based image classifier for weather images. It automatically splits a single dataset folder into training and test sets, trains a CNN from scratch, and reports test accuracy.

## Features
- Automatically resizes and normalizes all images to 128x128 RGB
- skips corrupted or malformed images
- Uses a CNN with <5 million parameters
- Automatic 80/20 train/test split from a single `Weather/` folder
- Prints final test accuracy

## Requirements
- Python 3.8+
- torch==2.1.0
- torchvision==0.16.0

Install requirements with:
```bash
pip install -r requirements.txt
```


## Model Architecture
```
MediumCNN(
  features: [Conv2d(3,64), Conv2d(64,128), Conv2d(128,256), Conv2d(256,512), AdaptiveAvgPool2d]
  classifier: Linear(512, num_classes)
)
```

## Libraries Used
- torch
- torchvision

