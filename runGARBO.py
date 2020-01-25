import sys, getopt
import random
import GARBO as ma
import cPickle as pickle
from multiprocessing import Event, Pipe, Process
from collections import deque
                             
## Run GARBO
def main(argv):
    # default setting
    ngen = 1000
    npop = 100
    minL = 40
    maxL = 50
    numN = 10
    rr = 1
    inputfile = ''
    outputdir = ''
    
    try:
        opts, args = getopt.getopt(argv,"hg:p:s:l:n:r:i:o:",["ngen=", "npop=", "minl=",
                                                             "maxl=", "nn=", "rn=", "ifile=", "ofolder="])
    except getopt.GetoptError:
        print('runGARBO.py -g <ngen> -p <npop> -s <min.len> -l <max.len> -n <num.islands> -r <rank> -i <inputfile> -o <outputdir> error')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('runGARBO.py -g <ngen> -p <npop> -s <min.len> -l <max.len> -n <num.islands> -r <rank> -i <inputfile> -o <outputdir>')
            sys.exit()
        elif opt in ("-i", "--ifile"):
            inputfile = arg
        elif opt in ("-o", "--odir"):
            outputdir = arg
        elif opt in ("-g", "--ngen"):
            ngen = int(arg)
        elif opt in ("-p", "--npop"):
            npop = int(arg)
        elif opt in ("-s"):
            minL = int(arg)
        elif opt in ("-l"):
            maxL = int(arg)
        elif opt in ("-n"):
            numN = int(arg)
        elif opt in ("-r"):
            rr = int(arg)
    print("started..")
    print(inputfile)
    if rr == 0:
        dat = ma.load_data_layer(inputfile, rank = False)
    else:
        dat = ma.load_data_layer(inputfile, rank = True)

    random.seed(64)
    #NBR_islandS = 10
    pipes = [Pipe(False) for _ in range(numN)]
    pipes_in = deque(p[0] for p in pipes)
    pipes_out = deque(p[1] for p in pipes)
    pipes_in.rotate(1)
    pipes_out.rotate(-1)
    # define the output files
    out_files = [outputdir + '/output_' + str(i) + '.pkl' for i in range(numN)]
    # set the processes
    e = Event()
    processes = [Process(target=ma.island, args=(i, dat['dat'],
                                                   dat['info'][0],
                                                   dat['info'][1],
                                                   ngen, npop, minL, maxL,
                                                   ipipe, opipe, e, out_files[i],
                                                   random.random()))
                 for i, (ipipe, opipe) in enumerate(zip(pipes_in, pipes_out))]

    # start the processes
    for proc in processes:
        proc.start()
    # ...
    for proc in processes:
        proc.join()

if __name__ == "__main__":
    main(sys.argv[1:])
    
