import sys 
import os

if len(sys.argv) != 2:
    print("Usage: python new_test.py <testXX>")
    sys.exit(1)

script_name = sys.argv[0][:-3]
testno = sys.argv[1]

previous = int(testno[-2:]) - 1
pretest = f'test{previous}'

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

for pre,new in zip(prefiles,files):
    if not os.path.isfile(new):
        os.system(f'cp {pre} {new}')
        print(f'created... {new}')
