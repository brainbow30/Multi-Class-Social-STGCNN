# Video_Trajectory_Predictor

## Installation
install requirements ```$ pip install -r requirements.txt```

## Train
run ```train.py```

## Test
run ```test.py```

## Visualise
run ```videoStreaming.py``` and go to localhost:5000

## Config
```path``` - location of video or training data

```class_enc``` - True or False whether to include class encoding module in model

```class_weighting``` - when training weight loss based on class weight

```frameSkip``` - number of frames to skip between coordinate samples

```labels``` - class types to train/test on

```checkpoint``` - location of pretrained model

```percentageToRemove``` - remove edge data from training data

```showGroundTruth``` - show ground truth vs predictions on visualisations

```saveImages``` - save ground truth vs predictions frames when a new prediction is made

