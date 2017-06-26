# Dog-Breed Classifier
This repository contains Dog-Breed classifier using CNN

The web application is implemented in Flask

There are three major parts for This
1. Human Face detector- A neural network that detects a human face_detect
2. Dog detector- Uses resnet50 to detect a dog in an image
3. Dog-Breed Detector - USing Transfer learning we create a dog breed detector


pre-trained weights with **orig** tag in their folder  name , were trained on nvidia gtx 1080 GPU based on the models.

Required packages can be found in requirements.txt

**Run install_Data.sh before running the Flask app** 

The architecture uses augmentated data set in terms of position and rotations, It also uses increasing layers of filters and filter sizes, In accordance to starting with smaller features to bigger features. We do not go too big as well to avoid loosing generalization (overfitting).
