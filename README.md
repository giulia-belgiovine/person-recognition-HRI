# Person Recognition HRI
This module is designed to provide the iCub robot with the ability to recognize people. Indeed, user identification is an essential step in creating a personalised long-term interaction with robots.
This module allows the robot to recognize both already known and unknown users. In the latter case, the user is enrolled in the dataset (open world recognition problem).

The model is for now based on face-recognition only. The multimodal version is work in progress.


## Model Description

## Dependencies
The person recognition is dependent on the following modules:

### 1. Camera


### 2. Face Detecor


### 3. Multiple Object Tracker (MOT)
The MOT module is used to track people in the scene and assign them a unique ID. By default it is run on icubsrv.
The MOT module sends to the Person Recognition a list of tuples made of tracker IDs and associated coordinates for each fac_box present in the frame.
Ex: ("21f24a1r" (226 163 255 214))

| port | in/out | description | connects to |
| --- | --- | --- | --- |


### 4. Objects Properties Collector

## How to use it
(rpc commands, etc..)
