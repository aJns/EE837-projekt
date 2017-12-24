#!/bin/bash

floyd run --gpu --data nikulaj/datasets/ships/1:data --env keras "python train_classifier.py"
