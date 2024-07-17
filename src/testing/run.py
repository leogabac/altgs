import sys
import os


def main():
    if len(sys.argv) != 3:
        print("Usage: python run.py <testXX> <size>")
        sys.exit(1)
    
    script_name = sys.argv[0][:-3]
    usr_input = sys.argv[1]
    size = int(sys.argv[2])
    datadir = f'../../data/{usr_input}'

    # create the data folder if it does not exists
    if not os.path.isdir(datadir):
        os.mkdir(datadir)
        os.mkdir(os.path.join(datadir,str(size)))
        print(f'created {datadir}')

    # if the vertices dir is not empty, simply delete everything
    vrt_path = os.path.join(datadir,str(size),'vertices')

    if not os.path.isdir(vrt_path):
        os.makedirs(vrt_path, exist_ok=True)
    else:
        # if the folder exists, check if it contains files 
        if not os.listdir(vrt_path):
            pass
        else: 
            os.system(f'rm {vrt_path}/*.csv')

    # run the simulation
    os.system('clear')
    print('RUNNING SIMULATIONS')

    os.system(f'python {usr_input}.py {size}')

    # change the directory
    os.chdir('../')

    # cleaning script
    os.system('clear')
    print('CLEANING DATA')

    os.system(f'python cleaning.py {usr_input}')

    # vertices script
    os.system('clear')
    print('COMPUTING VERTICES')

    os.system(f'python compute_vertices.py {usr_input}')


if __name__ == "__main__":
    main()

