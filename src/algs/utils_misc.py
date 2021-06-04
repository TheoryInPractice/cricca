
import os, re
 

def get_files(dir_name, ftype):
    files = []
    if os.path.isfile(dir_name):
        if dir_name.endswith(ftype):
            files.append(dir_name)
    elif os.path.isdir(dir_name):
        for fn in os.listdir(dir_name):
            if fn.endswith(ftype):
                files.append(dir_name+fn)
    
    return files


def filter_files_seed(allfiles, first_seed, last_seed):
    '''
    Returns subset of allfiles list of files containing 
    [first_seed, last_seed] files
    '''
    outfiles=[]
    for filename in allfiles:
        fname = filename.split('/')[-1]        
        seed = int(get_fname_value(fname, 'seed'))
        
        if seed <= last_seed and seed >= first_seed:
            outfiles.append(filename)
    return outfiles


def get_fname_value(fname, val):
    '''
    
    '''
    fname_comps=fname.split('_')
    value=None
    
    for comp in fname_comps:
        match = re.match(r"([a-z]+)([0-9]+)", comp, re.I)
        if match:
            items = match.groups()
            
            if items[0]==val:
                value = int(items[1])
    return value


def get_kvalue(fname, typ):
    '''
    Pulls out the kdistinct value from a filename
    typ='kdistinct' or 'ktotal'
    '''
    k = get_fname_value(fname, typ)
    return k
 

class TimeOutException(Exception):
   pass
 
 
def handler(signum, frame):
    #raise TimeOutException("end of time")
    print("Timeout Exception thrown")
    raise TimeOutException()


def num_isolates(A):
    isolates=0
    for i in range(A.shape[0]):
        isolated_v=True
        for j in range(A.shape[0]):
            if i!=j and A[i, j] != 0:
                isolated_v=False
        if isolated_v:
            isolates+=1
    return isolates


def get_max_edgeweight(G):
    # finds the maximum edge weight of the given graph
    edges = G.edges(data=True)
    max_eweight = -1
    
    for e in edges:
        w = e[2]['weight']
        
        if w > max_eweight:
            max_eweight = w
        
    return max_eweight
