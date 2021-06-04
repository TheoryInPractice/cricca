'''

'''

import os, random, sys
import numpy as np
import pandas as pd
import bio_datagen as bio_datagen

from os.path import dirname,realpath


def get_max_fileid(dir_name):
    # to give each graph a unique id, find the max id in the directory
    pklfiles = []
    if os.path.isfile(dir_name):
        pklfiles.append(dir_name)
    elif os.path.isdir(dir_name):
        for fn in os.listdir(dir_name):
            if fn.endswith('.pkl'):
                pklfiles.append(dir_name+fn)
            
    # First get current max id number
    max_id = 0
    for pklf in pklfiles: 
        fname = pklf[0:-4]                   
        fname_l = fname.split('/')  
        
        curr_fname = fname_l[-1]
        comps = curr_fname.split('_')
        
        id_ = int(comps[0])
        if id_ > max_id:
            max_id = id_
    return max_id


def create_tf_corpus(k, num_seeds, dir_name):
    '''
    
    '''
    f_id = get_max_fileid(dir_name)
    
    dat_fname = 'generator_dat/TF_dat/gene_TF_dat.csv'
    dat = pd.read_csv(dat_fname, delimiter=',')
    
    for seed in range(num_seeds):
        random.seed(seed)
        np.random.seed(seed)
        
        for k_val in range(2, k+1):
            print('\n---- k=', k_val, ' seed= ', seed)
            f_id+=1
            
            # get gene subset of random tfs
            tfs, genes = bio_datagen.get_tf_gene_subset(dat, k_val)
            
            # generate multiple maximum clique weights per tf subset         
            max_weights = [1, 4, 16]
            for mw in max_weights:
                clique_weights = bio_datagen.gen_psuedo_powerlaw_weights(k_val, mw)
                
                if 0 in clique_weights:
                    print('ERROR: no 0 clique weights')
                    return
                
                # given clique weights, create graph + write to file
                # randomly shuffle clique_weights--varies clique weight assignment
                for wi in range(6): 
                    random.shuffle(clique_weights)
                    bio_datagen.generate_TF_data(f_id, k_val, mw, 
                                                dir_name, seed, wi, 
                                                clique_weights, tfs, genes)
    
    
def create_lv_corpus(k, num_seeds, dir_name):
    '''
    
    '''
    f_id = get_max_fileid(dir_name)
    
    z_fname = 'generator_dat/LV_dat/multiplier_model_z.tsv'
    sum_fname = 'generator_dat/LV_dat/multiplier_model_summary.tsv'
    
    z_tsv = pd.read_csv(z_fname, delimiter='\t')
    sum_tsv = pd.read_csv(sum_fname, delimiter='\t') 
        
    for seed in range(num_seeds):
        random.seed(seed)
        np.random.seed(seed)
        
        for k_val in range(2, k+1):
            print('---- k=', k_val, ' seed= ', seed)
            f_id+=1
            
            # get gene subset of random lvs
            lvs, genes, lv_labels = bio_datagen.get_lv_gene_subset(z_tsv, sum_tsv, k_val)
            
            thresholds = [0.6]
            scale_factors = [1, 2, 4]
            for threshold in thresholds:
                for scale_fac in scale_factors:
                    bio_datagen.generate_LV_data(f_id, k_val, dir_name, 
                                                seed, threshold, scale_fac, 
                                                lvs, genes, lv_labels)
  
  
def create_corpora():
    k = 20
    num_seeds = 20
    
    tf_dir_name = 'data/pre_preprocessing/tf/'
    lv_dir_name = 'data/pre_preprocessing/lv/'
    
    if not os.path.exists(tf_dir_name):
        os.makedirs(tf_dir_name)
    
    if not os.path.exists(lv_dir_name):
        os.makedirs(lv_dir_name)
    
    print('Starting Creation of TF Corpus')
    create_tf_corpus(k, num_seeds, tf_dir_name)
    
    print('\n\nStarting Creation of LV Corpus')
    create_lv_corpus(k, num_seeds, lv_dir_name)
    

if __name__=="__main__":
    create_corpora()

        
        
