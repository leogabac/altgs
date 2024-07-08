#!/bin/bash

python cleaning.py 
rm ../data/test02/10/vertices/*.csv
python compute_vertices.py
