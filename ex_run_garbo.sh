# Install phython 2.7 and install  the following python modules within your virtual environment
# sklearn, multiprocessing, matplotlib, operator, bisect, deap, numpy, random, skfuzzy, cPickle, pandas, scipy, sys, getopt
# source activate 'your_virtual_env'
# ./ex_run_garbo.sh

############################### Run GARBO in parallel (example)
export NGEN=100      # Indicate the number of GA-iterations
export NPOP=50       # Indicate the number of the chromosomes to be generated for each niche
export MINL=30       # Indicate the initial minimum length of the generated chromosomes
export MAXL=50       # Indicate the initial maixmum length of the generated chromosomes
export NN=2          # Specify the number of niches
export RN=1          # Indicate if an initial rank of the features must be compiled (RN=1) otherwise it starts with no ranking information (RN=0).
export INPUT_FILE="data_ccle_erl_ge.csv" # Omics dataset (samples as columns and rows as features< The last feature must named 'class' and it correpsonds to the target label)
export OUTPUT_DIR="MRNA_run_1"    # Folder that will contain a file for each nicheserialized python-obejcts. Each file contains the
mkdir $OUTPUT_DIR
nohup python runGARBO.py -g $NGEN -p $NPOP -s $MINL -l $MAXL -n $NN -r $RN -i $INPUT_FILE -o $OUTPUT_DIR > output_mrna.log &
