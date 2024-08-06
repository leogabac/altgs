import sys
import os
import argparse

parser = argparse.ArgumentParser(description="Heavyside step function field profile")

# flags
parser.add_argument('-t', '--template', action='store', help='take a specific testno as template')

# positional arguments
parser.add_argument('name', type=str, help='new script name')

args = parser.parse_args()

testno = args.name

if args.template is None:
    previous = int(testno[-2:]) - 1
    pretest = f'test{previous}'
else:
    pretest = args.template

files = [
    f'{testno}.py',
    f'{testno}.ipynb',
    f'{testno}_field.py',
]

prefiles = [
    f'{pretest}.py',
    f'{pretest}.ipynb',
    f'{pretest}_field.py',
]

for pre, new in zip(prefiles, files):
    if not os.path.isfile(new):
        os.system(f'cp {pre} {new}')
        print(f'created... {new}')
