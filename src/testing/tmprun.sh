#!/bin/bash

for x in {1..20}; do
  python test11.py -s -v 10 "$x"
done
