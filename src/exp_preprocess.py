'''


'''

import argparse, os, pickle, time
import networkx as nx

import algs.preprocess as preprocess
import algs.utils_misc as utils_misc


def run(dirname, out_dirname, first_seed, last_seed):    
    print('in dirname: ', dirname)
    print('out_dirname: ', out_dirname)
    
    if not os.path.exists(out_dirname):
        os.makedirs(out_dirname)
        
    allfiles = utils_misc.get_files(dirname, '.txt')
    files = utils_misc.filter_files_seed(allfiles, first_seed, last_seed)
    
    # already processed graphs
    prerun_files = utils_misc.get_files(out_dirname, '.txt') 
    count=0
    
    
    for fname in files:
        count+=1
        # skips already processed graphs
        if out_dirname+fname.split('/')[-1] in prerun_files:
            continue
        
        print('\n__________________________________ {}/{} : {}'.format(count, 
                                                                       len(files),
                                                                       fname))
        
        G = nx.read_weighted_edgelist(fname, nodetype=int)
        
        k_distinct = utils_misc.get_kvalue(fname, 'kdistinct')
        
        # get pkl file info
        output=None
        pklfname = fname[0:-4]+'.pkl'
        with open(pklfname, 'rb') as infile: 
            output = pickle.load(infile)
        
        start = time.time()
        dat = preprocess.preprocess(G, k_distinct)
        end = time.time()   
        time_proc = end - start        
        print('preprocess time: ', time_proc)

        upd_output = preprocess.update_output_vals(G, output, dat, 
                                                   k_distinct, time_proc)
        
        # save graph and output data 
        preprocess_data = {'post_preprocessing' : upd_output}
    
        upd_fname = fname.split('/')[-1]
        print('out fname: ', out_dirname+upd_fname)
        nx.write_edgelist(G, out_dirname+upd_fname, data=['weight'])
        
        with open(out_dirname+upd_fname[0:-4]+'.pkl', 'wb') as f:
            pickle.dump(preprocess_data, f)
        
        
def main():
    parser = argparse.ArgumentParser()
   
    #----------------------------------- required args
    parser.add_argument('-f', '--first_seed', type=int,
        help="enter first seed value in seed range", required=True)
    parser.add_argument('-l', '--last_seed', type=int,
        help="enter last seed value in seed range", required=True)
    
    args = vars(parser.parse_args())
    
    first_seed = args.get('first_seed')
    second_seed = args.get('last_seed')
    
    tf_dirname = 'data/pre_preprocessing/tf/'
    lv_dirname = 'data/pre_preprocessing/lv/'
    
    tf_out_dirname = 'data/post_preprocessing/tf/'
    lv_out_dirname = 'data/post_preprocessing/lv/'
    

    # run tf graphs
    print('\nPreprocessing TF graphs')
    run(tf_dirname, tf_out_dirname, 
         first_seed, second_seed)
    
    # run lv graphs
    print('\nPreprocessing LV graphs')
    run(lv_dirname, lv_out_dirname, 
         first_seed, second_seed)


if __name__=="__main__":
    main()
    
    
