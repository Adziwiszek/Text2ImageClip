#!/bin/bash
echo "starting training..."
nohup python3 -m CVAE.cvae > log.txt 2>&1 &
