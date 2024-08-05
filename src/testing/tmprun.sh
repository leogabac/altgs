#!/bin/bash

for x in {1..20}; do
  python test11.py -a 10 "$x"
done

python test11.py -a 10 30
python test11.py -a 10 40
python test11.py -a 10 50
python test11.py -a 10 70
python test11.py -a 10 80
python test11.py -a 10 90
python test11.py -a 10 100
