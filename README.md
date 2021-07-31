# Cricca
This repository contains the source code for [Parameterized algorithms for identifying gene co-expression modules via weighted clique decomposition](https://arxiv.org/abs/2106.00657). 

## Requirements
numpy 

pandas

networkx

scipy

gurobi

# To Run Your Own Data

- From /cricca direcotory, run 'python main.py --graph_filename [your_fname.txt] --algorithm [either 'wecp', 'ipart', or 'lp'] --parameter [integer parameter value]'
- For information on other input arguments, run 'python main.py --help'


# To Recreate Paper Experiments

## 1. Graph Generation
Steps to generate the same corpus used in the paper experiments (see [paper](https://arxiv.org/abs/2106.00657) for additional graph generation details.):

    1. Download Transcription Factor Base Data: [TF Data](http://www.tfacts.org/TFactS-new/TFactS-v2/tfacts/data/Catalogues.xls) 

    2. Open Catalog.xls, save sheet 'TFactS_sign_less_version2' as 'gene_TF_dat.csv' in generator_dat directory. 

    3. Download Latent Variable Base Data (Note file is 84GB): [LV Data](https://ndownloader.figshare.com/files/14080160)

    4. Place the files named 'multiplier_model_z.tsv' and 'multiplier_model_summary.tsv' in generator_dat directory.

    5. From /cricca directory, run 'python src/graph_gen/gen_acda21_corpus.py'

Both TF and LV graphs will now exist in the src/data/pre_preprocessing directory. 


## 2. Preprocessing
To preprocess each graph, from /cricca directory, run: 

    'python src/exp_preprocess.py -f 0 -l 19'

-f/--first_seed and -l/--last_seed give the seeds of desired graphs to preprocess.


## 3. Kernelization
To kernelize each preprocessed graph with k [2-11] (after pre-processing), from /cricca directory, run:

    'python src/exp_runkernel.py -f 0 -l 19'


## 4. Matrix Decomposition
To run each matrix decomposition algorithm including wecp (original algorithm by Feldmann et al. 2020), ipart (integer partitioning based), and lp (linear programming based) algorithms, from /cricca directory, run:

    'python src/exp_runbsd.py -f 0 -l 19'

## 5. CSV Creation
To combine all data saved in pickle files (after running exp_runbsd.py)
 into a single csv file, from /cricca directory, run:

    'python src/tocsv.py'
    

## Additional Tests
To run the memory tests comparing wecp/ipart/lp algorithms, from /cricca directory, run:

    'python src/exp_runmemtests.py'









