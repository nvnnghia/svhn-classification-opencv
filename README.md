# svhn-classification-opencv
SVHN classification using tensorflow model in opencv.

## A TensorFlow implementation of Multi-digit Number Recognition from Street View Imagery using Deep Convolutional Neural Networks

## Setup
Clone the source code

$ git clone https://github.com/potterhsu/SVHNClassifier
$ cd SVHNClassifier
Download SVHN Dataset format 1

Extract to data folder, now your folder structure should be like below:

SVHNClassifier
    - data
        - extra
            - 1.png 
            - 2.png
            - ...
            - digitStruct.mat
        - test
            - 1.png 
            - 2.png
            - ...
            - digitStruct.mat
        - train
            - 1.png 
            - 2.png
            - ...
            - digitStruct.mat
## Usage

$ python convert_to_tfrecords.py --data_dir ./data

## Train

$ python train.py --data_dir ./data --train_logdir ./logs/train
Retrain if you need

$ python train.py --data_dir ./data --train_logdir ./logs/train2 --restore_checkpoint ./logs/train/latest.ckpt
Evaluate

$ python eval.py --data_dir ./data --checkpoint_dir ./logs/train --eval_logdir ./logs/eval
Visualize

$ tensorboard --logdir ./logs

## Traing code from: https://github.com/potterhsu/SVHNClassifier

## Inferene using opencv
1)Extract .pb file from checkpoint
Go to log directory: python3 export_graph.py
2)Optimize graph for opencv usage.
Go to log directory: python3 optimize_for_inference.py --input output_graph.pb --output opt_model6.pb --input_names shuffle_batch --output_names digit1/dense/BiasAdd,digit2/dense/BiasAdd,digit3/dense/BiasAdd,digit4/dense/BiasAdd,digit5/dense/BiasAdd
3) Run inferenec code: python3 test_cv1.py
