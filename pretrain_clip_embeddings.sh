#!/bin/bash
echo "starting pretraining..."
nohup python3 -m CVAE.data_prep > log.txt 2>&1 &
