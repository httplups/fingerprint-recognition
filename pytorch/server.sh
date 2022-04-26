#!/bin/bash
cd /opt
tar -zcvf - bashrc-mc857 fingerprint_identification miniconda3 pytorch installation.txt | nc $1 9000
