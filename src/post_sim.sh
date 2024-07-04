#!/bin/bash

python cleaning.py 
rm ../data/test01/10/vertices/*.csv
python bulk_vertices.py
