'''
    Takes all pickle files from 1. pre-preprocessing, 2. post-preprocessing, 
    3. kernelizing, 4. decomposing 
    and creates single csv file with all data. 

'''

import pandas as pd
import networkx as nx
import os, sys, pickle, pprint, shutil, datetime, time
from os.path import dirname,realpath
sys.path.insert(0,dirname(realpath(__file__))[:-10])
#print(sys.path)

import algs.utils_misc as utils_misc    


def check_for_missing_data(final_output, kernel_output, kdistinct, wscale, witer, fname):
    '''
    true_total input: 
    TF: wecp, ipart, lp - witer=0, wscale=sml,med, kdistinct <= k_distinct_max
    LV: wecp, ipart, lp -          wscale=sml,med, kdistinct <= k_distinct_max
    
    true_dist input:
    TF: ipart, lp - witer <= 4 [0-4], kdistinct <= k_distinct_max
    LV: ipart, lp -                   kdistinct <= k_distinct_max
    
    guess input:
    TF: ipart, lp - witer=0,  5 <= kdistinct <= k_distinct_max
    LV: ipart, lp -           5 <= kdistinct <= k_distinct_max
    
    '''
    k_distinct_max = 11                                          ######### NOTE
    
    if kdistinct==0:
        return
    
    if witer is None: 
        witer=0
                
    # first check true_total input
    if wscale=='sml' or wscale=='med':
        if kdistinct <= k_distinct_max:
            if witer==0:
                wecp_dat = final_output['decomp_data']['bsd_dw']['true_total']
                ip_dat = final_output['decomp_data']['bswd_dw_ip']['true_total']
                lp_dat = final_output['decomp_data']['bswd_dw_lp']['true_total']
                
                if wecp_dat is None:
                    print('ERROR: {} (true_total input) missing wecp k={}'.format(fname,
                                                                                  kdistinct))
        
        # second check for true_dist input for weight permutations
        if witer <= 5 and kdistinct <= 7:
            ip_dat = final_output['decomp_data']['bswd_dw_ip']['true_distinct']
            lp_dat = final_output['decomp_data']['bswd_dw_lp']['true_distinct']
            
            if ip_dat is None:
                print('ERROR: {} (true_distinct input) missing ip_dat k={}'.format(fname, 
                                                                                   kdistinct))
            if lp_dat is None:
                print('ERROR: {} (true_distinct input) missing lp_dat k={}'.format(fname,
                                                                                   kdistinct))
        
        # third check for true_dist input for weight permutations
        if witer==0 and kdistinct <= k_distinct_max:
            ip_dat = final_output['decomp_data']['bswd_dw_ip']['true_distinct']
            lp_dat = final_output['decomp_data']['bswd_dw_lp']['true_distinct']
            
            if ip_dat is None:
                print('ERROR: {} (true_distinct input) missing ip_dat k={}'.format(fname, 
                                                                                   kdistinct))
            if lp_dat is None:
                print('ERROR: {} (true_distinct input) missing lp_dat k={}'.format(fname,
                                                                                   kdistinct))
        
        # fourth check for guess input
        if witer==0 and kdistinct >= 5 and kdistinct <= 7:
            vals = [-0.6, -0.4, -0.2, 0.2, 0.4, 0.6]
            for val in vals:
                ip_dat = final_output['decomp_data']['bswd_dw_ip']['guesses']
                lp_dat = final_output['decomp_data']['bswd_dw_lp']['guesses']
                
                kern_dat = kernel_output['guess_kernels'][val]
                
                if kern_dat['passed_kernel']:
                    if ip_dat[val] is None:
                        print('ERROR: {} (guess {} input) missing ip_dat k={}'.format(fname, 
                                                                                      val,
                                                                                      kdistinct))
                    if lp_dat[val] is None:
                        print('ERROR: {} (guess {} input) missing lp_dat k={}'.format(fname, 
                                                                                      val,
                                                                                      kdistinct))
                

def print_keys(final_output):
    pre_preprocessing = final_output['pre_preprocessing']
    post_preprocessing = final_output['post_preprocessing']
    
    kernel_data = final_output['kernel_data']
    true_total_kernel = kernel_data['true_total_kernel']
    true_distinct_kernel = kernel_data['true_distinct_kernel']
    guess_kernels = kernel_data['guess_kernels']
    
    decomp_data = final_output['decomp_data']
    bsd_dw = decomp_data['bsd_dw']
    bswd_dw_lp = decomp_data['bswd_dw_lp']
    bswd_dw_ip = decomp_data['bswd_dw_ip']
        
    print('\nMain keys: ', list(final_output.keys()))
    print('pre_preprocessing keys: ', list(pre_preprocessing.keys()))
    print('\npost_preprocessing keys: ', list(post_preprocessing.keys()))
    print('\nkernel_data keys: ', list(kernel_data.keys()))
    
    if true_total_kernel is not None:
        print('\ntrue_total_kernel keys: ', list(true_total_kernel.keys()))
    else:
        print('\ntrue_total_kernel keys: ', true_total_kernel)
    
    if true_distinct_kernel is not None:
        print('\ntrue_distinct_kernel keys: ',
                list(true_distinct_kernel.keys()))
    else:
        print('\ntrue_distinct_kernel keys: ', true_distinct_kernel)
        
    if guess_kernels is not None:
        print('\nguess_kernels keys: ', list(guess_kernels.keys()))
    else:
        print('\nguess_kernels keys: ', guess_kernels.keys())
        
    
    print('\ndecomp_data keys: ', list(decomp_data.keys()))
    print('bsd_dw keys: ', list(bsd_dw.keys()))
    print('bswd_dw_lp keys: ', list(bswd_dw_lp.keys()))
    print('bswd_dw_ip keys: ', list(bswd_dw_ip.keys()))


def get_data(witer, wscale, fname, datatype, final_output, kernel_output, postproc_output, preproc_output, get_colnames=False):
    '''
    filename
    pre_preprocessing:  n-init, m-init, kdistinct-init, ktotal-init, 
                        witer, wscale(sml, med, lrg)
    post_preprocessing/pre_kernel: n-postproc, m-postproc, 
                        kdistinct-postproc, ktotal-postproc, time_preproc

    - true_total input: 
    post_kernel/pre_bsd: kinput-1, n-1, m-1, passed_kernel-1, time_kernel-1
            wecp:  passed_bsd-1, reconstructs-1, found_cliq_frac-1, time_bsd-1
    
    - true_dist input: 
    post_kernel/pre_bsd: kinput-2, n-2, m-2, passed_kernel-2, time_kernel-2
            lp:    passed_bsd-2-1, reconstructs-2-1, found_cliq_frac-2-1, time_bsd-2-1
            ipart: passed_bsd-2-2, reconstructs-2-2, found_cliq_frac-2-2, time_bsd-2-2
    
    - guess0 input: 
    post_kernel/pre_bsd: kinput-3, n-3, m-3, passed_kernel-3, time_kernel-3
            lp:    passed_bsd-3-1, reconstructs-3-1, found_cliq_frac-3-1, time_bsd-3-1
            ipart: passed_bsd-3-2, reconstructs-3-2, found_cliq_frac-3-2, time_bsd-3-2
    ....
    
    - guess5 input: 
    post_kernel/pre_bsd: kinput-8, n-8, m-8, passed_kernel-8, time_kernel-8
            lp:    passed_bsd-8-1, reconstructs-8-1, found_cliq_frac-8-1, time_bsd-8-1
            ipart: passed_bsd-8-2, reconstructs-8-2, found_cliq_frac-8-2, time_bsd-8-2
    
    '''
    colnames = {'filename' : None, 
                'datatype' : None,
                #pre-preprocessing 
                'n-init' : None, 
                'm-init' : None, 
                'kdistinct-init' : None, 
                'ktotal-init' : None, 
                'witer' : None, 
                'wscale' : None, 
                # post-preprocessing
                'n-postproc' : None, 
                'm-postproc' : None, 
                'kdistinct-postproc' : None, 
                'ktotal-postproc' : None, 
                'time_preproc' : None, 
                #----- true_total input
                'kinput-1' : None, 
                'n-1' : None, 
                'm-1' : None, 
                'max_edgeweight-1' : None,
                'passed_kernel-1' : None, 
                'time_kernel-1' : None,
                # wecp
                'passed_bsd-1' : None, 
                'reconstructs-1' : None, 
                'found_cliq_frac-1' : None,
                'time_bsd-1' : None}
            
    # kdistinct + guess input info
    for i in range(2, 8+1):
        # post kernel info
        colnames['kinput-'+str(i)]=None
        colnames['n-'+str(i)]=None
        colnames['m-'+str(i)]=None
        colnames['max_edgeweight-'+str(i)]=None
        colnames['passed_kernel-'+str(i)]=None
        colnames['time_kernel-'+str(i)]=None
        
        # lp
        colnames['passed_bsd-'+str(i)+'-1']=None
        colnames['reconstructs-'+str(i)+'-1']=None
        colnames['found_cliq_frac-'+str(i)+'-1']=None
        colnames['time_bsd-'+str(i)+'-1']=None
        colnames['mem_usage-'+str(i)+'-1']=None
        
        # ipart
        colnames['passed_bsd-'+str(i)+'-2']=None
        colnames['reconstructs-'+str(i)+'-2']=None
        colnames['found_cliq_frac-'+str(i)+'-2']=None
        colnames['time_bsd-'+str(i)+'-2']=None
        colnames['mem_usage-'+str(i)+'-2']=None
            
    if get_colnames:
        return list(colnames.keys())
    else:
        pre_preprocessing = preproc_output
        post_preprocessing = postproc_output['post_preprocessing']
        
        kernel_data = kernel_output
        true_total_kernel = kernel_data['true_total_kernel']
        true_distinct_kernel = kernel_data['true_distinct_kernel']
        guess_kernels = kernel_data['guess_kernels']
        
        decomp_data = final_output['decomp_data']
        bsd_dw = decomp_data['bsd_dw']
        bswd_dw_lp = decomp_data['bswd_dw_lp']
        bswd_dw_ip = decomp_data['bswd_dw_ip']
        
        upd_fname = fname.split('/')[-1]
        if datatype=='tf':
            # remove 'witer' from fname
            newfnm = ''
            comps = upd_fname.split('_')
            for u in comps:
                if 'witer' not in u:
                    newfnm+='_'+u
            upd_fname=newfnm[1:-1]
        
        #### fill in the data
        colnames['filename'] = upd_fname
        colnames['datatype'] = datatype
        colnames['n-init'] = pre_preprocessing['n']
        colnames['m-init'] = pre_preprocessing['m']
        colnames['kdistinct-init'] = pre_preprocessing['kdistinct']
        colnames['ktotal-init'] = pre_preprocessing['ktotal']
        colnames['witer'] = witer
        colnames['wscale'] = wscale
        
        colnames['n-postproc'] = post_preprocessing['n']
        colnames['m-postproc'] = post_preprocessing['m']
        colnames['kdistinct-postproc'] = post_preprocessing['kdistinct']
        colnames['ktotal-postproc'] = post_preprocessing['ktotal']
        colnames['time_preproc'] = post_preprocessing['preprocess_time']
        
        ##----- true_total input
        if true_total_kernel is not None:
            colnames['kinput-1'] = true_total_kernel['kinput']
            colnames['n-1'] = true_total_kernel['n']
            colnames['m-1'] = true_total_kernel['m']
            colnames['max_edgeweight-1'] = true_total_kernel['max_eweight']
            colnames['passed_kernel-1'] = true_total_kernel['passed_kernel']
            colnames['time_kernel-1'] = true_total_kernel['kernel_time']
            
            # wecp data
            truetot = bsd_dw['true_total']
            colnames['passed_bsd-1'] = truetot['passed_bsd']
            colnames['reconstructs-1'] = truetot['reconstructs']
            colnames['found_cliq_frac-1'] = truetot['found_cliq_fraction']
            colnames['time_bsd-1'] = truetot['time_bsd']
        
        ##----- true distinct input
        if true_distinct_kernel is not None:
            # post kernel info
            colnames['kinput-2'] = true_distinct_kernel['kinput']
            colnames['n-2'] = true_distinct_kernel['n']
            colnames['m-2'] = true_distinct_kernel['m']
            colnames['max_edgeweight-2'] = true_distinct_kernel['max_eweight']
            colnames['passed_kernel-2'] = true_distinct_kernel['passed_kernel']
            colnames['time_kernel-2'] = true_distinct_kernel['kernel_time']
            
            # lp
            truedist_lp = bswd_dw_lp['true_distinct']
            colnames['passed_bsd-2-1'] = truedist_lp['passed_bsd']
            colnames['reconstructs-2-1'] = truedist_lp['reconstructs']
            colnames['found_cliq_frac-2-1'] = truedist_lp['found_cliq_fraction']
            colnames['time_bsd-2-1'] = truedist_lp['time_bsd']
            
            if 'mem_usage' in truedist_lp.keys():
                colnames['mem_usage-2-1']=truedist_lp['mem_usage']
            
            # ipart
            truedist_ip = bswd_dw_ip['true_distinct']
            colnames['passed_bsd-2-2'] = truedist_ip['passed_bsd']
            colnames['reconstructs-2-2'] = truedist_ip['reconstructs']
            colnames['found_cliq_frac-2-2'] = truedist_ip['found_cliq_fraction']
            colnames['time_bsd-2-2'] = truedist_ip['time_bsd']
            
            if 'mem_usage' in truedist_ip.keys():
                colnames['mem_usage-2-2']=truedist_ip['mem_usage']
        
        i=3
        vals = [-0.6, -0.4, -0.2, 0.2, 0.4, 0.6]
        for val in vals:
            if guess_kernels is not None:
                guess_kern = guess_kernels[val]
                if guess_kern is not None:
                    guess_ip = bswd_dw_ip['guesses'][val]
                    guess_lp = bswd_dw_lp['guesses'][val]
                                        
                    # post kernel info
                    colnames['kinput-'+str(i)] = guess_kern['kinput']
                    colnames['n-'+str(i)] = guess_kern['n']
                    colnames['m-'+str(i)] = guess_kern['m']
                    colnames['max_edgeweight-'+str(i)] = guess_kern['max_eweight']
                    colnames['passed_kernel-'+str(i)] = guess_kern['passed_kernel']
                    colnames['time_kernel-'+str(i)] = guess_kern['kernel_time']
                    
                    if guess_lp is not None:
                        # lp
                        colnames['passed_bsd-'+str(i)+'-1'] = guess_lp['passed_bsd']
                        colnames['reconstructs-'+str(i)+'-1'] = guess_lp['reconstructs']
                        colnames['found_cliq_frac-'+str(i)+'-1'] = guess_lp['found_cliq_fraction']
                        colnames['time_bsd-'+str(i)+'-1'] = guess_lp['time_bsd']
                    
                    if guess_ip is not None:
                        # ipart
                        colnames['passed_bsd-'+str(i)+'-2'] = guess_ip['passed_bsd']
                        colnames['reconstructs-'+str(i)+'-2'] = guess_ip['reconstructs']
                        colnames['found_cliq_frac-'+str(i)+'-2'] = guess_ip['found_cliq_fraction']
                        colnames['time_bsd-'+str(i)+'-2'] = guess_ip['time_bsd']
            i+=1
        
        return colnames
    

def get_overlapping_nodes(fname, dat, final_output, kernel_output, postproc_output, preproc_output):
    '''
    For each clique compute/save:
        1. # of nodes that are in other cliques
        2. # of cliques that are overlapping w. current clique
        3. clique size
    '''
    #print('\n\n')
    #print(final_output['pre_preprocessing'])
    #clique_vertices = final_output['post_preprocessing']['clique_vertices']   
    #clique_vertices = final_output['pre_preprocessing']['clique_vertices']  
    
    #clique_vertices = postproc_output['post_preprocessing']['clique_vertices']
    clique_vertices = preproc_output['clique_vertices']
    total_cliques = len(clique_vertices)
    
    clique_i=0
    for cliquei in clique_vertices:
        cliquen = len(cliquei)
        dat['clique_n-'+str(clique_i)]=cliquen # clique size
        
        num_node_ovl=set()
        num_cliq_ovl=0
        clique_j=0
        for cliquej in clique_vertices:
            if clique_i!=clique_j:
                #nno = len([x for x in cliquei if x in cliquej])
                #num_node_ovl+=nno 
                
                nno=0
                for x in cliquei:
                    if x in cliquej:
                        nno+=1
                        num_node_ovl.add(x)
                
                if nno>0:   # # of cliques overlapping
                    num_cliq_ovl+=1
                    
            clique_j+=1
        
        dat['clique_nodeovl-'+str(clique_i)]=len(num_node_ovl)/cliquen
        dat['clique_clqovl-'+str(clique_i)]=num_cliq_ovl/total_cliques
        clique_i+=1
    

def run(final_datadir, out_datadir, kernel_dir, postproc_dir, preproc_dir, typ):
    print('preproc dirname:    ', preproc_dir)
    print('postproc dirname:   ', postproc_dir)
    print('kern dirname:       ', kernel_dir)
    print('final data dirname: ', final_datadir)
    print('out_dirname:        ', out_datadir)
    print()
    
    colnames = get_data(None, None, None, None, None, None, None, None, get_colnames=True)
    
    if not os.path.exists(out_datadir):
        os.makedirs(out_datadir)
                
    final_files = utils_misc.get_files(final_datadir, '.pkl')
    
    data = []
    for fname in final_files:        
        preproc_file = preproc_dir+fname.split('/')[-1]
        postproc_file = postproc_dir+fname.split('/')[-1]
        kern_file = kernel_dir+fname.split('/')[-1]
        
        witer=None
        wscale=None  # sml, med, lrg
        if typ=='tf':
            witer = int(utils_misc.get_fname_value(fname, 'witer'))
            ws = int(utils_misc.get_fname_value(fname, 'scalefac'))
            if ws==1:
                wscale='sml'
            elif ws==4:
                wscale='med'
            elif ws==16:
                wscale='lrg'
        elif typ=='lv':
            ws = int(utils_misc.get_fname_value(fname, 'scalefac'))
            if ws==1:
                wscale='sml'
            elif ws==2:
                wscale='med'
            elif ws==4:
                wscale='lrg'
        
        # get pkl file info
        with open(fname, 'rb') as infile: 
            final_output = pickle.load(infile)
        
        with open(kern_file, 'rb') as infile:
            kernel_output = pickle.load(infile)
            
        with open(postproc_file, 'rb') as infile:
            postproc_output = pickle.load(infile)
        
        with open(preproc_file, 'rb') as infile:
            preproc_output = pickle.load(infile)
            
        if final_output is None:
            print('Warning: final_output is None')
        
        if kernel_output is None:
            print('Warning: kernel_output is None')
        
        if postproc_output is None:
            print('Warning: postproc_output is None')
        
        if preproc_output is None:
            print('Warning: preproc_output is None')
        
        k_total = postproc_output['post_preprocessing']['ktotal']
        k_distinct = postproc_output['post_preprocessing']['kdistinct']
        
        dat=None
        try:                                                              #NOTE
            dat = get_data(witer, wscale, fname, typ, final_output, 
                       kernel_output, postproc_output, preproc_output)
        except:
            print("ERROR: couldnt get complete data for ", fname)
            check_for_missing_data(final_output, kernel_output, k_distinct, 
                                    wscale, witer, fname)

        if dat is not None:
            get_overlapping_nodes(fname, dat, final_output, 
                                  kernel_output, postproc_output, preproc_output)
            
            data.append(dat)
                
    df = pd.DataFrame(data)
    
    return df
            


def main():
    
    preproc_dir_tf = 'data/pre_preprocessing/tf/'
    preproc_dir_lv = 'data/pre_preprocessing/lv/'
    
    postproc_dir_tf = 'data/post_preprocessing/tf/'
    postproc_dir_lv = 'data/post_preprocessing/lv/'
    
    kernel_dir_tf = 'data/kernels/tf/'
    kernel_dir_lv = 'data/kernels/lv/'
    
    in_datadir_tf = 'data/finaldata/tf/'
    in_datadir_lv = 'data/finaldata/lv/'
    
    out_datadir_tf = 'data/csvfiles/'
    out_datadir_lv = 'data/csvfiles/'

    start = time.time()
    print('Creating tf csv')
    tf_df = run(in_datadir_tf, out_datadir_tf, kernel_dir_tf, postproc_dir_tf, preproc_dir_tf, 'tf')
    #print(tf_df['num_overlap'])
    
    print('\nCreating lv csv')
    lv_df = run(in_datadir_lv, out_datadir_lv, kernel_dir_lv, postproc_dir_lv, preproc_dir_lv, 'lv')
    #print(lv_df)
    end = time.time()
    
    print(tf_df.shape, lv_df.shape)
    df = tf_df.append(lv_df)
    print(df)
    
    now = datetime.datetime.now()
    date_time = now.strftime("%m-%d-%Y_%H-%M-%S")
    
    name = 'fulldata_'+date_time+'.csv'
    print(name, 'time: ', end-start)
    
    source_dir = 'data/csvfiles/mostrecent/'
    target_dir = 'data/csvfiles/'
    
    if not os.path.exists(source_dir):
        os.makedirs(source_dir)
        
    file_names = os.listdir(source_dir)
        
    #move files from mostrecent dir
    for file_name in file_names:
        shutil.move(os.path.join(source_dir, file_name), target_dir)
    
    #save csv to mostrecent dir
    df.to_csv(source_dir+name)


if __name__=="__main__":
    main()
    
    
    
    
    
    
    
    
