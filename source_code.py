# %%
# -*- coding: utf-8 -*-
# 
# %%
import numpy as np
import pandas as pd
from sklearn.utils import resample
import time
import re
import pickle
from tqdm import tqdm
import time
from scipy import stats
import gc
import csv
import jellyfish
import itertools

import ray
import psutil

from sklearnex import patch_sklearn 
patch_sklearn() 

from sklearn.linear_model import LogisticRegressionCV
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from xgboost import XGBClassifier, plot_importance
from sklearn.preprocessing import StandardScaler
sc = StandardScaler() 

from sklearn.model_selection import train_test_split, cross_validate, ShuffleSplit

import matplotlib.pyplot as plt
from matplotlib_venn import venn3, venn2

pd.set_option('display.max_rows', 20) 
pd.set_option('display.max_columns', None) 

pd.options.display.float_format = '{:.5f}'.format 
np.set_printoptions(precision=5, suppress=True) 

# %%
##########################
# Load Data 
##########################
#%%
def load_data(pkl_path, region_code=None, reset_index=False):
    with open(pkl_path, 'rb') as pkl:
        data = pickle.load(pkl)
    print(f"data.shape: {data.shape}")

    if region_code:
        data = data[data['kommnr'].str.startswith(region_code)]
        print(f"data.shape: {data.shape}, region code: {region_code}")

    if reset_index:
        data = data.reset_index(drop=True) 
        print(f"index reset complete")

    return data
#%%
region_code=None 
path_1875='data/sources/_1875.pkl'
path_1900='data/sources/_1900.pkl'
_1875_org = load_data(path_1875, region_code, reset_index=True) ## fixed
_1900_org = load_data(path_1900, region_code, reset_index=True) ## fixed


########################
########################
# Linking
########################
########################
#%%
##########################
# Data preprocessing
##########################
#%%
variables = ['id_i','id_d','fornavn','fornavns','etternavn','etternavns','kjonn','famst_ipums','sivst','fsted_kode','faar','bosted','kommnr','ts']

def subset(df, variables):
    print(variables)
    return df[variables]
    
def extract_4digits(str):
    if re.search('!!',str): 
        return "0"
    else:
        if not re.match('[0-9]{4}$',str): 
            tmp = re.search('[0-9]{4}',str)  
            if tmp:
                return tmp.group() 
            else:
                return "0"
        else:
            return str

def extract_maritalstatus(str):
    if re.search('!!',str): 
        return "!!"
    else:
        if re.match('^uk',str): 
            return '!!'
        elif re.match('^ub',str): 
            return '!!'
        elif re.match('^unkn',str): 
            return '!!'
        elif re.match('^g',str): 
            return 'g'
        elif re.match('^u',str):
            return 'ug'
        elif re.match('^e',str):
            return 'e'
        elif re.match('^s',str):
            return 's'
        elif re.match('^f',str):
            return 'f'
        else:
            return '!!'

def extract_sex(str):
    if re.search('!!',str):  
        return '!!'
    elif re.match('^k',str): 
        return 'k'
    elif re.match('^m',str): 
        return 'm'
    else:
        return '!!'

def cut_spaces(str):
    str = re.sub('^\s+', '', str)
    str = re.sub('\s+$', '', str)
    str = re.sub('\s{2,}', ' ', str)
    return str

def preprocess_birthyear(df): 
    print(f"[org] faar> # None: {df.faar.isnull().sum()}")
    start = time.time()
    df.faar = df.faar.fillna("0")
    print(f"[after fill_na] faar> # None: {df[df.faar==0].shape[0]}")
    df.faar = df.faar.apply(extract_4digits).apply(int)
    df.faar = np.where(df.faar>df.ts.apply(int)+5, 0, df.faar) 
    df.faar = np.where(df.faar<df.ts.apply(int)-150, 0, df.faar) 
    end = time.time()
    print(f"preprocess birthyear: {end-start:.5f} sec")
    print(f"[after preprocess] faar> # None: {df[df.faar==0].shape[0]}")    
    return df

def preprocess_birthplace(df): 
    print(f"[org] fsted_kode> # None: {df.fsted_kode.isnull().sum()}")
    start = time.time()
    df.fsted_kode = df.fsted_kode.fillna("0")
    print(f"[after fill_na] fsted_kode> # None: {df[df.fsted_kode=='0'].shape[0]}")
    df.fsted_kode = df.fsted_kode.apply(extract_4digits)
    end = time.time()
    print(f"preprocess birthplace: {end-start:.5f} sec")
    print(f"[after preprocess] fsted_kode> # None: {df[df.fsted_kode=='0'].shape[0]}")    
    return df

def preprocess_sex(df): 
    print(f"[org] kjonn> # None: {df.kjonn.isnull().sum()}")
    start = time.time()
    df.kjonn = df.kjonn.fillna("!!").apply(str.lower)
    print(f"[after fill_na] kjonn> # None: {df[df.kjonn=='!!'].shape[0]}")
    df.kjonn = df.kjonn.apply(extract_sex)
    end = time.time()
    print(f"preprocess sex: {end-start:.5f} sec")
    print(f"[after preprocess] kjonn> # None: {df[df.kjonn=='!!'].shape[0]}")    
    return df

def preprocess_maritalstatus(df): 
    print(f"[org] sivst> # None: {df.sivst.isnull().sum()}")
    start = time.time()
    df.sivst = df.sivst.fillna("!!").apply(str.lower)
    print(f"[after fill_na] sivst> # None: {df[df.sivst=='!!'].shape[0]}")
    df.sivst = df.sivst.apply(extract_maritalstatus)
    end = time.time()
    print(f"preprocess marital status: {end-start:.5f} sec")
    print(f"[after preprocess] sivst> # None: {df[df.sivst=='!!'].shape[0]}")    
    return df

def preprocess_famst_ipums(df): 
    print(f"[org] famst_ipums> # None: {df.famst_ipums.isnull().sum()}")
    start = time.time()
    df.famst_ipums = df.famst_ipums.fillna("9999").apply(str.lower)
    print(f"[after fill_na] famst_ipums> # None: {df[df.famst_ipums=='!!'].shape[0]}")
    df['famst_ipums_f2'] = df.famst_ipums.str[:2]

    end = time.time()
    print(f"preprocess famst_ipums: {end-start:.5f} sec")
    print(f"[after preprocess] famst_ipums> # None: {df[df.famst_ipums=='!!'].shape[0]}")    
    return df

def preprocess_names(df): 
    start = time.time()
    print(f"[org] fornavn> # None: {df.fornavn.isnull().sum()}")
    print(f"[org] fornavns> # None: {df.fornavns.isnull().sum()}")
    print(f"[org] etternavn> # None: {df.etternavn.isnull().sum()}")
    print(f"[org] etternavns> # None: {df.etternavns.isnull().sum()}")
    df.fornavns = df.fornavns.fillna("!!") 
    df.fornavn = df.fornavn.fillna("!!") 
    df.etternavns = df.etternavns.fillna("!!") 
    df.etternavn = df.etternavn.fillna("!!") 
    df.fornavn = df.fornavn.apply(cut_spaces)
    df.fornavns = df.fornavns.apply(cut_spaces)
    df.etternavn = df.etternavn.apply(cut_spaces)
    df.etternavns = df.etternavns.apply(cut_spaces)
    end = time.time()
    print(f"preprocess names: {end-start:.5f} sec")
    print(f"[after fill_na/preprocess] fornavn> # None: {df[df.fornavn=='!!'].shape[0]}")
    print(f"[after fill_na/preprocess] fornavns> # None: {df[df.fornavns=='!!'].shape[0]}")
    print(f"[after fill_na/preprocess] etternavn> # None: {df[df.etternavn=='!!'].shape[0]}")
    print(f"[after fill_na/preprocess] etternavns> # None: {df[df.etternavns=='!!'].shape[0]}")
    return df

def preprocess_livingaddress(df): 
    print(f"[org] bosted> # None: {df.bosted.isnull().sum()}")
    start = time.time()
    df.bosted = df.bosted.fillna("!!") # 
    end = time.time()
    print(f"preprocess bosted: {end-start:.5f} sec")
    print(f"[after fill_na/preprocess] bosted> # None: {df[df.bosted=='!!'].shape[0]}")
    return df

def get_comm_names(df):
    df['comm_fornavns'] = -np.log(df.fornavns.value_counts().loc[df.fornavns].values/df.shape[0]) 
    df['comm_etternavns'] = -np.log(df.etternavns.value_counts().loc[df.etternavns].values/df.shape[0]) 
    return df

def get_mid_names(df):
    df[['fornavns_first','fornavns_mid']] = df.fornavns.str.split(pat=' ',n=1,expand=True) 
    df.fornavns_mid=df.fornavns_mid.fillna("") 
    df[['etternavns_first','etternavns_mid']] = df.etternavns.str.split(pat=' ',n=1,expand=True) 
    df.etternavns_mid=df.etternavns_mid.fillna("") 
    return df

def get_name_initials(df):
    df['fornavns_init'] = df.fornavns.str[:1]
    df['fornavns_mid_init'] = df.fornavns_mid.str[:1]
    df['etternavns_init'] = df.etternavns.str[:1]
    df['etternavns_mid_init'] = df.etternavns_mid.str[:1]
    return df

def assign_adjusted_komm_kode(df):
    with open('data/sources/kommnr_change.csv', mode='r') as infile:
        reader = csv.reader(infile)
        fsted_dict = {rows[0]:rows[1] for rows in reader if (rows[1]!='')} 
        df['adj_fsted_kode'] = np.where(df.fsted_kode.isin(fsted_dict.keys()), df.fsted_kode.map(fsted_dict), df.fsted_kode)
        df['adj_kommnr'] = np.where(df.kommnr.isin(fsted_dict.keys()), df.kommnr.map(fsted_dict), df.kommnr)        
        return df

def get_comm_birthplace(df):
    df['comm_fsted'] = -np.log(df.adj_fsted_kode.value_counts().loc[df.adj_fsted_kode].values/df.shape[0]) 
    return df

def get_householdID(df):
    df_id_d_split  = df.id_d.str.rsplit(pat='<',n=1,expand=True) 
    df['house_id']= df.ts+"<"+df_id_d_split[0] 
    df['i_in_house'] = df_id_d_split[1] 
    famst_ipums_sivst = df.famst_ipums + df.sivst
    woman_househead = np.logical_and(df.famst_ipums.shift(1)!='0101', np.logical_or(famst_ipums_sivst=='0201e',famst_ipums_sivst=='0201ug'))
    new_household = np.logical_or(df.famst_ipums=='0101', woman_househead==True).cumsum()
    df['householdID'] = df.house_id+'<<'+new_household.apply(str)
    df['household_num'] = (df.householdID != df.householdID.shift(1)).cumsum()
    return df

def family_candidates(df):
    a = (df.fornavns.str[:3]+df.faar.apply(str)+df.kjonn+df.fsted_kode).values 
    b = (df.fornavns.str[:3]+(df.faar+1).apply(str)+df.kjonn+df.fsted_kode).values
    c = (df.fornavns.str[:3]+(df.faar-1).apply(str)+df.kjonn+df.fsted_kode).values
    abc = list(itertools.chain.from_iterable([a,b,c])) 
    return set(abc)

def get_familyinfo(df):
    family_size_dic = df.groupby('household_num').household_num.size()
    tqdm.pandas()
    household_num_dic = df.groupby('household_num').progress_apply(family_candidates) 
    df['family_size'] = df.household_num.map(family_size_dic)
    df['family_info'] = df.household_num.map(household_num_dic)
    return df

def assign_blockkey(df):
    df['block_key']=df.fornavns.str[:1]+df.adj_fsted_kode.str[:3]+df.kjonn 
    df['midN_block_key']=df.fornavns_mid.str[:1]+df.adj_fsted_kode.str[:3]+df.kjonn
    return df

def make_index_column(df): 
    df['idx']=df.index
    return df

#%%
def preprocess(df, variables):
    df = subset(df, variables)
    df = preprocess_birthyear(df)
    df = preprocess_birthplace(df)
    df = preprocess_sex(df)
    df = preprocess_maritalstatus(df)
    df = preprocess_famst_ipums(df)
    df = preprocess_names(df)
    df = preprocess_livingaddress(df)
    df = get_comm_names(df)
    df = get_mid_names(df)
    df = get_name_initials(df)
    df = assign_adjusted_komm_kode(df)
    df = get_comm_birthplace(df)
    df = get_householdID(df)
    df = get_familyinfo(df)
    df = assign_blockkey(df)
    df = make_index_column(df) 
    return df
#%%
start = time.time()
_1875 = preprocess(_1875_org, variables)
_1900 = preprocess(_1900_org, variables)
end = time.time()
print(f"preprocessed time: {end-start:.5f} sec")
#%%
_1900_20 = _1900[np.logical_and((_1900.faar<1881), (_1900.faar!=0))]



#%%
##############################
# Selecting training data
##############################
#%%
def get_datasets(df1875, df1900, region_code):
    _75 = df1875[df1875['adj_kommnr'].str.startswith(region_code)].reset_index(drop=True) 
    _00 = df1900[df1900['adj_kommnr'].str.startswith(region_code)].reset_index(drop=True) 
    _00_20 = _00[np.logical_and((_00.faar<1881), (_00.faar!=0))]
    print(f"_1875_{region_code}.shape:{_75.shape}\n_1900_{region_code}.shape: {_00.shape}\n_1900_{region_code}_20.shape: {_00_20.shape}")
    return (_75,_00,_00_20)

# 0432 (0432,0433)
region_code='0432'
(_1875_432,_1900_432,_1900_432_20)= get_datasets(_1875,_1900,region_code)

# 1922 (1922)
region_code='1922'
(_1875_1922,_1900_1922,_1900_1922_20)= get_datasets(_1875,_1900,region_code)

# 1924 (1924)
region_code='1924'
(_1875_1924,_1900_1924,_1900_1924_20)= get_datasets(_1875,_1900,region_code)

# 1931 (1930,1931)
region_code='1931'
(_1875_1931,_1900_1931,_1900_1931_20)= get_datasets(_1875,_1900,region_code)

# 1933 (1932,1933)
region_code='1933'
(_1875_1933,_1900_1933,_1900_1933_20)= get_datasets(_1875,_1900,region_code)

# 1936 (1935,1936)
region_code='1936'
(_1875_1936,_1900_1936,_1900_1936_20)= get_datasets(_1875,_1900,region_code)

# 1938 (1938)
region_code='1938'
(_1875_1938,_1900_1938,_1900_1938_20)= get_datasets(_1875,_1900,region_code)

# 1941 (1941,1942)
region_code='1941'
(_1875_1941,_1900_1941,_1900_1941_20)= get_datasets(_1875,_1900,region_code)

# 1943
region_code='1943'
(_1875_1943,_1900_1943,_1900_1943_20)= get_datasets(_1875,_1900,region_code)

# troms
region_code='19'
(_1875_19,_1900_19,_1900_19_20)= get_datasets(_1875,_1900,region_code)

_1875_gt = pd.concat([_1875_432,_1875_1922,_1875_1924,_1875_1931,_1875_1933,_1875_1936,_1875_1938,_1875_1941,_1875_1943])
_1875_19gt = pd.concat([_1875_1922,_1875_1924,_1875_1931,_1875_1933,_1875_1936,_1875_1938,_1875_1941,_1875_1943])

_1900_gt_20 = pd.concat([_1900_432_20,_1900_1922_20,_1900_1924_20,_1900_1931_20,_1900_1933_20,_1900_1936_20,_1900_1938_20,_1900_1941_20,_1900_1943_20])
_1900_19gt_20 = pd.concat([_1900_1922_20,_1900_1924_20,_1900_1931_20,_1900_1933_20,_1900_1936_20,_1900_1938_20,_1900_1941_20,_1900_1943_20])
#%%



##############################
## Indexing (blocking)
##############################
#%%
def auto_garbage_collect(pct=80.0):
    if psutil.virtual_memory().percent >= pct:
        gc.collect()

@ray.remote ##
def get_candidates(df1,df2):    
    return df1.apply((lambda df1: (df1.name, df2[np.logical_and(
                                                    np.logical_or(
                                                        np.logical_or(df1['block_key']==df2['block_key'],df1['midN_block_key']==df2['block_key']),
                                                        df1['block_key']==df2['midN_block_key']),
                                                    abs(df1['faar']-df2['faar'])<6)].index)),axis=1)

def candidates_ray(df1, df2, cpus):
    df_list = np.array_split(df1,cpus)
    df2_ref = ray.put(df2) ##
    results_df_list = [get_candidates.remote(df_list[i],df2_ref) for i,x in enumerate(tqdm(df_list))]
    auto_garbage_collect()
    len_results_df_list = len(results_df_list)
    results_list = []
    pbar = tqdm(total=100)
    while len(results_df_list):
        done_id, results_df_list = ray.wait(results_df_list)
        results_list.append(ray.get(done_id[0]))
        pbar.update(100//len_results_df_list)
        auto_garbage_collect()
    pbar.close()
    cand = pd.concat(results_list).sort_index() 
    return cand

def get_candidates_ray(df1, df2, mid_match=False):
    cpus=psutil.cpu_count()
    print(cpus)
    ray.init(num_cpus=cpus)
    cand = candidates_ray(df1,df2,cpus)
    ray.shutdown()
    return cand

#%%
for direction in ('7500','0075'):
    print(f"direction = {direction}")
    if direction=='7500':
        start = time.time()
        candidates_all = get_candidates_ray(_1875, _1900_20, mid_match=False) 
        print(f'{time.time() - start:.5} seconds')
    elif direction=='0075': 
        start = time.time()
        candidates_all = get_candidates_ray(_1900_20, _1875, mid_match=False) 
        print(f'{time.time() - start:.5} seconds')
    
    all_cand_list = np.array_split(candidates_all,20) ## fixed
    
    for i in range(len(all_cand_list)):
        all_cand_list[i].to_pickle(f"data/tempresults/{direction}/norge_candidates_all_N{i}.pkl") 

#%%
## run before rerunning ray
# ray.shutdown() 
#%%

##############################
## Pair comparison
##############################
#%%
def auto_garbage_collect(pct=80.0):
    if psutil.virtual_memory().percent >= pct:
        gc.collect()

@ray.remote ##
def compare_pairs_ray(candidates, df1, df2): 
    res = [[None for _ in range(0)] for _ in range(len(df1))]

    for idx, potential_matches in enumerate(candidates):
        for order, df2_idx in enumerate(potential_matches[1]): 
            firstN_jw = 1 - jellyfish.jaro_winkler(df1.fornavn[potential_matches[0]], df2.fornavn[df2_idx]) 
            firstNs_jw = 1 - jellyfish.jaro_winkler(df1.fornavns[potential_matches[0]], df2.fornavns[df2_idx]) 
            lastN_jw = 1 - jellyfish.jaro_winkler(df1.etternavn[potential_matches[0]], df2.etternavn[df2_idx]) 
            lastNs_jw = 1 - jellyfish.jaro_winkler(df1.etternavns[potential_matches[0]], df2.etternavns[df2_idx]) 
            midNfirstN_jw = 1 - jellyfish.jaro_winkler(df1.fornavns_mid[potential_matches[0]], df2.fornavns[df2_idx]) 
            firstNmidN_jw = 1 - jellyfish.jaro_winkler(df1.fornavns[potential_matches[0]], df2.fornavns_mid[df2_idx]) 
            firstfNfirstN_jw = 1 - jellyfish.jaro_winkler(df1.fornavns_first[potential_matches[0]], df2.fornavns[df2_idx]) 
            firstNfirstfN_jw = 1 - jellyfish.jaro_winkler(df1.fornavns[potential_matches[0]], df2.fornavns_first[df2_idx]) 
            firstNI_diff = int((df1.fornavns_init[potential_matches[0]]!=df2.fornavns_init[df2_idx])) 
            lastNI_diff = int((df1.etternavns_init[potential_matches[0]]!=df2.etternavns_init[df2_idx]))  
            firstmNIfirstNI_diff = int((df1.fornavns_mid_init[potential_matches[0]]!=df2.fornavns_init[df2_idx])) 
            firstNIfirstmNI_diff = int((df1.fornavns_init[potential_matches[0]]!=df2.fornavns_mid_init[df2_idx])) 
            firstmNIfirstNI_diff = int((df1.etternavns_mid_init[potential_matches[0]]!=df2.etternavns_init[df2_idx])) 
            firstNIfirstmNI_diff = int((df1.etternavns_init[potential_matches[0]]!=df2.etternavns_mid_init[df2_idx])) 
            birthY_diff = abs(df1.faar[potential_matches[0]] - df2.faar[df2_idx]) 
            birthP_diff = abs(int(df1.fsted_kode[potential_matches[0]]) - int(df2.fsted_kode[df2_idx])) 
            adj_birthP_diff = abs(int(df1.adj_fsted_kode[potential_matches[0]]) - int(df2.adj_fsted_kode[df2_idx])) 
            firstN_comm = df1.comm_fornavns[potential_matches[0]]+df2.comm_fornavns[df2_idx]
            lastN_comm = df1.comm_etternavns[potential_matches[0]]+df2.comm_etternavns[df2_idx] 
            birthP_comm = df1.comm_fsted[potential_matches[0]]+df2.comm_fsted[df2_idx] 
            municp_diff = abs(int(df1.kommnr[potential_matches[0]]) - int(df2.kommnr[df2_idx])) 
            adj_municp_diff = abs(int(df1.adj_kommnr[potential_matches[0]]) - int(df2.adj_kommnr[df2_idx])) 
            residence_diff = jellyfish.jaro_winkler(df1.bosted[potential_matches[0]], df2.bosted[df2_idx])
            comm_family_num = len(df1.family_info[potential_matches[0]].intersection(df2.family_info[df2_idx])) 
            marital_diff = int((df1.sivst[potential_matches[0]]!=df2.sivst[df2_idx])) 
            fm_relation_diff = int((df1.famst_ipums[potential_matches[0]]!=df2.famst_ipums[df2_idx])) 
            # 75-00 
            marital_illogic = int((df1.sivst[potential_matches[0]]+df2.sivst[df2_idx]) in (['gug','eug','fug','sug'])) 
            fm_relation_illogic = int((df1.famst_ipums_f2[potential_matches[0]]+df2.famst_ipums_f2[df2_idx]) in (['0103','0203'])) 

            res[idx].append([potential_matches[0], df2_idx, firstN_jw, firstNs_jw, lastN_jw, lastNs_jw, birthY_diff, birthP_diff, adj_birthP_diff, 
                                     firstN_comm, lastN_comm, birthP_comm, 
                                     midNfirstN_jw, firstNmidN_jw, firstfNfirstN_jw, firstNfirstfN_jw, 
                                     firstmNIfirstNI_diff, firstNIfirstmNI_diff, firstNI_diff, lastNI_diff,
                                     municp_diff, adj_municp_diff, residence_diff, comm_family_num, marital_diff, fm_relation_diff,
                                     marital_illogic, fm_relation_illogic]) 
            
    return res                

def get_comparison_vectors_ray(candidates, df1, df2):
    cpus=psutil.cpu_count()
    print(cpus)
    ray.init(num_cpus=cpus)
    arr_list = np.array_split(candidates,cpus) 
    df1_ref = ray.put(df1)
    df2_ref = ray.put(df2)
    results_arr_list = [compare_pairs_ray.remote(arr_list[i],df1_ref, df2_ref) for i,x in enumerate(tqdm(arr_list))]
    auto_garbage_collect()
    len_results_df_list = len(results_arr_list)
    results_list = []
    pbar = tqdm(total=100)
    while len(results_arr_list):
        done_id, results_arr_list = ray.wait(results_arr_list)
        results_list.extend(ray.get(done_id[0]))
        pbar.update(100//len_results_df_list)
        auto_garbage_collect()
    pbar.close()
    ray.shutdown()
    return results_list

def get_comparison_vector_df_ray(df1,df2,features,i,way):
    
    with open(f"data/tempresults/{way}/norge_candidates_all_N{i}.pkl", "rb") as fh:
        cand = pickle.load(fh)
        start = time.time()
        res_df = pd.DataFrame(list(itertools.chain.from_iterable(get_comparison_vectors_ray(cand, df1, df2))), columns=features).sort_values(by=['idx_x','idx_y']).reset_index(drop=True)
        res_df['idx_xy'] = list(zip(res_df.idx_x, res_df.idx_y))
        print(f'{time.time() - start:.5} seconds')
        res_df.to_pickle(f"data/tempresults/{way}/compared_all_df_{i}.pkl") 

#%%
## create subsets consisting of variables used for linking
_1875s = _1875.loc[:,['id_i', 'fornavn', 'fornavns', 'etternavn', 'etternavns',
       'famst_ipums', 'sivst', 'fsted_kode', 'faar', 'bosted',
       'kommnr', 'comm_fornavns', 'comm_etternavns', 'fornavns_first',
       'fornavns_mid', 'fornavns_init',
       'fornavns_mid_init', 'etternavns_init', 'etternavns_mid_init',
       'adj_fsted_kode', 'comm_fsted', 'family_info','adj_kommnr','famst_ipums_f2']]
_1900s = _1900.loc[:,['id_i', 'fornavn', 'fornavns', 'etternavn', 'etternavns',
       'famst_ipums', 'sivst', 'fsted_kode', 'faar', 'bosted',
       'kommnr', 'comm_fornavns', 'comm_etternavns', 'fornavns_first',
       'fornavns_mid', 'fornavns_init',
       'fornavns_mid_init', 'etternavns_init', 'etternavns_mid_init',
       'adj_fsted_kode', 'comm_fsted', 'family_info','adj_kommnr','famst_ipums_f2']]
_1900_20s = _1900s[np.logical_and((_1900s.faar<1881), (_1900s.faar!=0))]
#%%
used_columns = ['idx_x','idx_y','firstN_JW','firstNs_JW','lastN_JW','lastNs_JW','birthY_diff','birthP_diff','adj_birthP_diff', 
                'firstN_comm','lastN_comm','birthP_comm', 
                'midNfirstN_jw', 'firstNmidN_jw', 'firstfNfirstN_jw', 'firstNfirstfN_jw',
                'firstmNIfirstNI_diff', 'firstNIfirstmNI_diff','firstNI_diff', 'lastNI_diff',
                'municp_diff','adj_municp_diff','residence_diff','comm_family_num', 'marital_diff', 'fm_relation_diff',
                'marital_illogic', 'fm_relation_illogic'] 
#%%
for direction in ('7500','0075'):
    print(f"direction = {direction}")
    if direction=='7500':
        for i in range(20):
            get_comparison_vector_df_ray(_1875s,_1900_20s,used_columns,i, direction)
    elif direction=='0075':
        for i in range(20):
            get_comparison_vector_df_ray(_1900_20s,_1875s,used_columns,i, direction)
#%%

##############################
## training ML models
##############################

## potential pairs
#%%
def get_potential_pairs_from_big_compared(df,region_code):
    if df.ts[0]=='1875':
        direction='7500'
        if region_code.startswith('19'):
            path = f"data/tempresults/{direction}/compared_all_df_19.pkl"
        elif region_code.startswith('04'):
            path = f"data/tempresults/{direction}/compared_all_df_4.pkl"

    elif df.ts[0]=='1900':
        direction='0075'
        if region_code.startswith('19'):
            path = f"data/tempresults/{direction}/compared_all_df_19.pkl"
        elif region_code.startswith('04'):
            path = f"data/tempresults/{direction}/compared_all_df_5.pkl"

    print(direction, path)

    with open(path, "rb") as fh: #
            compared_all = pickle.load(fh)

    region_shape = df[df.adj_kommnr==region_code].shape
    print(f"{df.ts[0]} {region_code} shape: {region_shape}")
    first_resident_idx = df[df.adj_kommnr==region_code].head(1).index.item()
    last_resident_idx = df[df.adj_kommnr==region_code].tail(1).index.item()
    print(first_resident_idx)
    print(last_resident_idx)
    first_candidate_idx = compared_all[compared_all.idx_x==first_resident_idx].head(1).index.item() 
    last_candidate_idx = compared_all[compared_all.idx_x==last_resident_idx].tail(1).index.item() 
    print(first_candidate_idx)
    print(last_candidate_idx)
    potential_pairs_tmp = compared_all.iloc[first_candidate_idx:last_candidate_idx+1]
    print(potential_pairs_tmp.tail(1).index.item())
    print(f"{df.ts[0]} {region_code} candidate shape: {potential_pairs_tmp.shape}")
    print(f"{region_shape[0]}-{len(potential_pairs_tmp.idx_x.value_counts())}={region_shape[0]-len(potential_pairs_tmp.idx_x.value_counts())}")
    potential_pairs_tmp.to_pickle(f"data/tempresults/{direction}/{region_code}_potential_pairs.pkl")
    print(f"data/tempresults/{direction}/{region_code}_potential_pairs.pkl",'Done!')


#%%
def get_idi_matches_idx(all_idi_matches, region_code, way):
    # 75-00 
    if way=='7500':
        res = all_idi_matches[all_idi_matches.adj_kommnr_x.str.startswith(region_code)] 
        res['idx_xy'] = list(zip(res.idx_x, res.idx_y))
    # 00-75 
    elif way=='0075':
        res = all_idi_matches[all_idi_matches.adj_kommnr_y.str.startswith(region_code)] 
        res['idx_xy'] = list(zip(res.idx_y, res.idx_x))
        
    true_pairs = res[['id_i','idx_xy']]
     
    dup_idi = pd.DataFrame(res.id_i.value_counts())
    dup_idi = dup_idi[dup_idi.id_i>1]
    dup_idi_records = res[res.id_i.isin(dup_idi.index)]
    
    print(f'{region_code}\nidi_matches:{res.shape[0]}\nduplicated idi:{dup_idi.shape[0]}\nrecords with duplicated idi:{dup_idi_records.shape[0]}\n')
    return true_pairs 

#%%
## cross validation 
#%%
extended_var = ['firstN_JW','firstNs_JW','lastN_JW','lastNs_JW','birthY_diff','birthP_diff','adj_birthP_diff',
                'firstN_comm','lastN_comm','birthP_comm', 
                'midNfirstN_jw', 'firstNmidN_jw', 'firstfNfirstN_jw', 'firstNfirstfN_jw',
                'firstmNIfirstNI_diff', 'firstNIfirstmNI_diff','firstNI_diff', 'lastNI_diff',
                'municp_diff','adj_municp_diff','residence_diff','comm_family_num', 'marital_diff', 'fm_relation_diff',
                'marital_illogic', 'fm_relation_illogic']
              
limited_var = ['firstN_JW','firstNs_JW','lastN_JW','lastNs_JW','birthY_diff','birthP_diff','adj_birthP_diff',
                'firstN_comm','lastN_comm','birthP_comm', 
                'midNfirstN_jw', 'firstNmidN_jw', 'firstfNfirstN_jw', 'firstNfirstfN_jw',
                'firstmNIfirstNI_diff', 'firstNIfirstmNI_diff','firstNI_diff', 'lastNI_diff']
#%%
def cross_val_test(potential_pairs, true_pairs, variables, test_size=0.1):
    potential_pairs.loc[:,['label']] = np.where(potential_pairs.idx_xy.isin(true_pairs.idx_xy),1,0)
    print(f"true in potential pairs: {potential_pairs[potential_pairs.label==1].shape[0]}, all true: {true_pairs.shape[0]}")
    X = potential_pairs.loc[:,variables].values
    y = np.ravel(potential_pairs.loc[:,['label']].values)
    print(np.ravel(y.shape))
    
    X = sc.fit_transform(X) #  feature scaling

    #lg_clf = LogisticRegressionCV(random_state=0)
    #svm_clf = SVC(kernel='rbf', random_state=0)
    #rf_clf = RandomForestClassifier(random_state=0)
    xgb_clf = XGBClassifier(random_state=0)

    cv = ShuffleSplit(n_splits=5, test_size=0.1, random_state=0)

    scores = cross_validate(xgb_clf, X, y, scoring=('precision','recall','f1'),cv=cv)
    print(scores) 
    
    return (xgb_clf, scores)



## train models 
#%%
def train_model(potential_pairs, true_pairs, variables, test_size=0.1):
    potential_pairs.loc[:,['label']] = np.where(potential_pairs.idx_xy.isin(true_pairs.idx_xy),1,0)
    print(f"true in potential pairs: {potential_pairs[potential_pairs.label==1].shape[0]}, all true: {true_pairs.shape[0]}")
    X = potential_pairs.loc[:,variables].values
    y = np.ravel(potential_pairs.loc[:,['label']].values)
    print(np.ravel(y.shape))
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.1, random_state=0) 
    sc = StandardScaler() 
    X_train = sc.fit_transform(X_train) 
    X_test = sc.transform(X_test) 
    lg_clf, svm_clf, rf_clf, xgb_clf = train_classifiers(X_train, X_test, y_train, y_test)
    
    return (lg_clf, svm_clf, rf_clf, xgb_clf)

def train_classifiers(X_train, X_test, y_train, y_test):
    lg_clf = LogisticRegressionCV(random_state=0)
    svm_clf = SVC(kernel='rbf', random_state=0)
    rf_clf = RandomForestClassifier(random_state=0)
    xgb_clf = XGBClassifier(random_state=0)

    start = time.time()
    lg_clf.fit(X_train, y_train)
    print(f'<<LR>> fit done: {time.time() - start:.5} seconds')

    start = time.time()
    svm_clf.fit(X_train, y_train)
    print(f'<<SVM>> fit done: {time.time() - start:.5} seconds')

    start = time.time()
    rf_clf.fit(X_train, y_train)
    print(f'<<RF>> fit done: {time.time() - start:.5} seconds')

    start = time.time()
    xgb_clf.fit(X_train, y_train)
    print(f'<<XGB>> fit done: {time.time() - start:.5} seconds')


    return (lg_clf, svm_clf, rf_clf, xgb_clf)

#%%
## feature importance
ext_features = [
                'JW distance between first names (original)',
                'JW distance between first names (standardized)',
                'JW distance between last names (original)',
                'JW distance between last names (standardized)',
                'Difference in birth years',
                'Difference in birth place codes (original)',
                'Difference in birth place codes (adjusted)',
                'Commonality of first names',
                'Commonality of last names',
                'Commonality of birth place codes (adjusted)', 
                'JW distance between middle name and first name', 
                'JW distance between first name and middle name', 
                'JW distance between first name w/o middle name and first name', 
                'JW distance between first name and first name w/o middle name',
                'Difference in middle name initial and first name initial', 
                'Difference in first name initial and middle name initial',
                'Difference in first name initials',
                'Difference in last name initials',
                'Difference in municipalities (original)',
                'Difference in municipalities (adjusted)',
                'Difference in residences_diff',
                'Number of family members in common', 
                'Difference in marital status', 
                'Difference in family relations',
                'Illogical change in marital status', 
                'Illogical change in family relations']
lim_features = [
                'JW distance between first names (original)',
                'JW distance between first names (standardized)',
                'JW distance between last names (original)',
                'JW distance between last names (standardized)',
                'Difference in birth years',
                'Difference in birth place codes (original)',
                'Difference in birth place codes (adjusted)',
                'Commonality of first names',
                'Commonality of last names',
                'Commonality of birth place codes (adjusted)', 
                'JW distance between middle name and first name', 
                'JW distance between first name and middle name', 
                'JW distance between first name w/o middle name and first name', 
                'JW distance between first name and first name w/o middle name',
                'Difference in middle name initial and first name initial', 
                'Difference in first name initial and middle name initial',
                'Difference in first name initials',
                'Difference in last name initials',]
#%%
def my_plot_importance(booster, figsize, features, **kwrgs):
    booster.get_booster().feature_names = features
    importance = booster.get_booster().get_score(importance_type='gain')
    for key in importance.keys():
        importance[key] = round(importance[key],2)
    fig, ax = plt.subplots(1,1,figsize=figsize)
    ax.set_title('Feature importance')
    ax.title.set_size(14)
    ax.set_xlabel("",fontsize=14)
    ax.set_yticklabels(features, fontsize=14)
    return plot_importance(importance, ax=ax, importance_type='gain', xlim=(0,400), 
                            xlabel='Average gain', ylabel=None, title=None,**kwrgs)

#%%
gt_regions = ['0432','1922','1924','1931','1936','1938','1941','1943','1933'] #
#%%
for ts in range(2):
    if ts==0:
        for i,v in enumerate(gt_regions):
            print(v)
            get_potential_pairs_from_big_compared(_1875,v)
    elif ts==1:
        for i,v in enumerate(gt_regions):
            print(v)
            get_potential_pairs_from_big_compared(_1900_20,v)
#%% 
for direction in ("7500","0075"):
    print(f"direction = {direction}")
    with open(f"data/tempresults/{direction}/0432_potential_pairs.pkl", "rb") as fh:
        potential_pairs_432 = pickle.load(fh)
    with open(f"data/tempresults/{direction}/1931_potential_pairs.pkl", "rb") as fh:
        potential_pairs_1931 = pickle.load(fh)
    with open(f"data/tempresults/{direction}/1936_potential_pairs.pkl", "rb") as fh:
        potential_pairs_1936 = pickle.load(fh)
    with open(f"data/tempresults/{direction}/1941_potential_pairs.pkl", "rb") as fh:
        potential_pairs_1941 = pickle.load(fh)
    with open(f"data/tempresults/{direction}/1943_potential_pairs.pkl", "rb") as fh:
        potential_pairs_1943 = pickle.load(fh)
    with open(f"data/tempresults/{direction}/1922_potential_pairs.pkl", "rb") as fh:
        potential_pairs_1922 = pickle.load(fh)
    with open(f"data/tempresults/{direction}/1924_potential_pairs.pkl", "rb") as fh:
        potential_pairs_1924 = pickle.load(fh)
    with open(f"data/tempresults/{direction}/1933_potential_pairs.pkl", "rb") as fh:
        potential_pairs_1933 = pickle.load(fh)
    with open(f"data/tempresults/{direction}/1938_potential_pairs.pkl", "rb") as fh:
        potential_pairs_1938 = pickle.load(fh)
    #%%
    potential_pairs_gt = pd.concat([potential_pairs_432,
                                potential_pairs_1922,potential_pairs_1924,potential_pairs_1931,potential_pairs_1933,
                                potential_pairs_1936,potential_pairs_1938,potential_pairs_1941,potential_pairs_1943])

    potential_pairs_gt.to_pickle(f"data/tempresults/{direction}/potential_pairs_gt.pkl")
    #%%
    ## split training set and test set
    potential_pairs_gt_train, potential_pairs_gt_test = train_test_split(potential_pairs_gt, test_size=0.1, random_state=0)

    #%%
    idi_matches_all = pd.merge(_1875, _1900, how='inner', on=['id_i'])
    #%%
    idi_matches_432 = get_idi_matches_idx(idi_matches_all,'0432', direction) 
    idi_matches_1922 = get_idi_matches_idx(idi_matches_all,'1922', direction)
    idi_matches_1924 = get_idi_matches_idx(idi_matches_all,'1924', direction)
    idi_matches_1931 = get_idi_matches_idx(idi_matches_all,'1931', direction)
    idi_matches_1933 = get_idi_matches_idx(idi_matches_all,'1933', direction)
    idi_matches_1936 = get_idi_matches_idx(idi_matches_all,'1936', direction)
    idi_matches_1938 = get_idi_matches_idx(idi_matches_all,'1938', direction)
    idi_matches_1941 = get_idi_matches_idx(idi_matches_all,'1941', direction)
    idi_matches_1943 = get_idi_matches_idx(idi_matches_all,'1943', direction)
    idi_matches_gt = pd.concat([idi_matches_432,idi_matches_1922,idi_matches_1924,idi_matches_1931,idi_matches_1933,idi_matches_1936,idi_matches_1938,idi_matches_1941,idi_matches_1943])

    #%%x
    lg_clf_gt_default, scores_lg_default = cross_val_test(potential_pairs_gt_train,idi_matches_gt,extended_var)
    #%%
    lg_clf_gt_default_l, scores_lg_default_l = cross_val_test(potential_pairs_gt_train,idi_matches_gt,limited_var)
    #%%
    rf_clf_gt_default, scores_rf_default = cross_val_test(potential_pairs_gt_train,idi_matches_gt,extended_var)
    #%%
    rf_clf_gt_default_l, scores_rf_default_l = cross_val_test(potential_pairs_gt_train,idi_matches_gt,limited_var)
    #%%
    xgb_clf_gt_default, scores_xgb_default = cross_val_test(potential_pairs_gt_train,idi_matches_gt,extended_var)
    #%%
    xgb_clf_gt_default_l, scores_xgb_default_l = cross_val_test(potential_pairs_gt_train,idi_matches_gt,limited_var)
    #%%
    svm_clf_gt_default, scores_svm_default = cross_val_test(potential_pairs_gt_train,idi_matches_gt,extended_var)
    #%%
    svm_clf_gt_default_l, scores_svm_default_l = cross_val_test(potential_pairs_gt_train,idi_matches_gt,limited_var)
    #%%
    print(f"precision_mean\t{scores_lg_default['test_precision'].mean():.2f}\tprecision_std\t{scores_lg_default['test_precision'].std():.2f}")
    print(f"recall_mean\t{scores_lg_default['test_recall'].mean():.2f}\trecall_std\t{scores_lg_default['test_recall'].std():.2f}")
    print(f"f1score_mean\t{scores_lg_default['test_f1'].mean():.2f}\tf1score_std\t{scores_lg_default['test_f1'].std():.2f}")
    #%%
    print(f"precision_mean\t{scores_rf_default['test_precision'].mean():.2f}\tprecision_std\t{scores_rf_default['test_precision'].std():.2f}")
    print(f"recall_mean\t{scores_rf_default['test_recall'].mean():.2f}\trecall_std\t{scores_rf_default['test_recall'].std():.2f}")
    print(f"f1score_mean\t{scores_rf_default['test_f1'].mean():.2f}\tf1score_std\t{scores_rf_default['test_f1'].std():.2f}")
    #%%
    print(f"precision_mean\t{scores_xgb_default['test_precision'].mean():.2f}\tprecision_std\t{scores_xgb_default['test_precision'].std():.2f}")
    print(f"recall_mean\t{scores_xgb_default['test_recall'].mean():.2f}\trecall_std\t{scores_xgb_default['test_recall'].std():.2f}")
    print(f"f1score_mean\t{scores_xgb_default['test_f1'].mean():.2f}\tf1score_std\t{scores_xgb_default['test_f1'].std():.2f}")
    #%%
    print(f"precision_mean\t{scores_svm_default['test_precision'].mean():.2f}\tprecision_std\t{scores_svm_default['test_precision'].std():.2f}")
    print(f"recall_mean\t{scores_svm_default['test_recall'].mean():.2f}\trecall_std\t{scores_svm_default['test_recall'].std():.2f}")
    print(f"f1score_mean\t{scores_svm_default['test_f1'].mean():.2f}\tf1score_std\t{scores_svm_default['test_f1'].std():.2f}")
    #%%
    # Time-invariant
    xlg_clf_gtl, xsvm_clf_gtl, xrf_clf_gtl, xgb_clf_gtl = train_model(potential_pairs_gt,idi_matches_gt,limited_var) 
    #%%
    xlg_clf_gtl.to_pickle(f"data/tempresults/{direction}/xlg_clf_gtl.pkl")
    xsvm_clf_gtl.to_pickle(f"data/tempresults/{direction}/xsvm_clf_gtl.pkl")
    xrf_clf_gtl.to_pickle(f"data/tempresults/{direction}/xrf_clf_gtl.pkl")
    xgb_clf_gtl.to_pickle(f"data/tempresults/{direction}/xgb_clf_gtl.pkl")

    #%%
    # Extended
    xlg_clf_gt, xsvm_clf_gt, xrf_clf_gt, xgb_clf_gt = train_model(potential_pairs_gt,idi_matches_gt,extended_var) 
    #%%
    xlg_clf_gt.to_pickle(f"data/tempresults/{direction}/xlg_clf_gt.pkl")
    xsvm_clf_gt.to_pickle(f"data/tempresults/{direction}/xsvm_clf_gt.pkl")
    xrf_clf_gt.to_pickle(f"data/tempresults/{direction}/xrf_clf_gt.pkl")
    xgb_clf_gt.to_pickle(f"data/tempresults/{direction}/xgb_clf_gt.pkl")
    #%%
    my_plot_importance(xgb_clf_gt, (7,8), ext_features)
    #%%
    my_plot_importance(xgb_clf_gtl, (7,6), lim_features)
    #%%
    plot_importance(xgb_clf_gt.get_booster(),importance_type='gain', xlabel='Average gain')
    #%%
    plot_importance(xgb_clf_gtl.get_booster(),importance_type='gain', xlabel='Average gain')



##############################
# Classification
##############################
#%%
def real_predict(potential_pairs, classifier, variables):
    X = potential_pairs.loc[:,variables].values
    X = sc.transform(X) 
    y_pred = (classifier.predict_proba(X)[:,1]>=0.1).astype(bool) ## 
    y_prob = classifier.predict_proba(X)[:,1]
    print(f"predicted: {y_pred.sum()}")
    return y_pred, y_prob

def get_predicted_matches(classifier,variables, way):
    matched_idx_list=[]
    for i in range(20):
        with open(f"data/tempresults/{way}/compared_all_df_{i}.pkl", "rb") as fh:
            cand = pickle.load(fh)
            print(f"df {i} shape: {cand.shape}")
            start = time.time()
            preds, probs = real_predict(cand, classifier, variables)
            print(f'{time.time() - start:.5} seconds')
            cand['pred_label']=preds
            cand['pred_prob']=probs
            matched_idx = cand[cand.pred_label==1]
            print(f"df {i} matched shape: {matched_idx.shape}")
            matched_idx_list.append(matched_idx)
    return matched_idx_list

def get_predicted_matches_comp_vectors(comp_vectors,classifier,variables):
    preds, probs = real_predict(comp_vectors, classifier, variables)
    comp_vectors['pred_label']=preds
    comp_vectors['pred_prob']=probs
    matched_idx = comp_vectors[comp_vectors.pred_label==1]
    print(f"matched shape: {matched_idx.shape}")
    return matched_idx

#%%
def check_dup_gap_cutoff(comp_vectors_x_on_prob, comp_vectors_y_on_prob, dup_gap):
    bestx_from_dup_idxx = comp_vectors_x_on_prob[comp_vectors_x_on_prob.prob_gap>=dup_gap] 
    uniqy_on_bestx = bestx_from_dup_idxx[bestx_from_dup_idxx.idx_y.map(bestx_from_dup_idxx.idx_y.value_counts())==1]
    dupy_on_bestx = bestx_from_dup_idxx[bestx_from_dup_idxx.idx_y.map(bestx_from_dup_idxx.idx_y.value_counts())>1]
    y_idxx_grouped = dupy_on_bestx.groupby('idx_y')['pred_prob']

    dupy_on_bestx['prob_gap_y'] = y_idxx_grouped.transform(lambda x: x.nlargest(1)/x.nlargest(2).min())  
    
    besty_on_bestx = dupy_on_bestx[dupy_on_bestx.prob_gap_y>=dup_gap] 
    
    bestx_from_dup_idxx = pd.concat([uniqy_on_bestx, besty_on_bestx], axis=0, join='inner')
    print(f"best x from duplicated x:\t{bestx_from_dup_idxx.shape[0]}")
    
    besty_from_dup_idxy = comp_vectors_y_on_prob[comp_vectors_y_on_prob.prob_gap>=dup_gap] 
    uniqx_on_besty = besty_from_dup_idxy[besty_from_dup_idxy.idx_x.map(besty_from_dup_idxy.idx_x.value_counts())==1]
    
    dupx_on_besty = besty_from_dup_idxy[besty_from_dup_idxy.idx_x.map(besty_from_dup_idxy.idx_x.value_counts())>1]
    x_idxy_grouped = dupx_on_besty.groupby('idx_x')['pred_prob']
    
    dupx_on_besty['prob_gap_x'] = x_idxy_grouped.transform(lambda x: x.nlargest(1)/x.nlargest(2).min()) 
    
    bestx_on_besty = dupx_on_besty[dupx_on_besty.prob_gap_x>=dup_gap] 

    besty_from_dup_idxy = pd.concat([uniqx_on_besty, bestx_on_besty], axis=0, join='inner')
    print(f"best y from duplicated y:\t{besty_from_dup_idxy.shape[0]}")

    res_from_dup = pd.concat([bestx_from_dup_idxx, besty_from_dup_idxy], axis=0).drop_duplicates('idx_xy') 
    resx_from_dup = res_from_dup.sort_values('pred_prob',ascending=False).drop_duplicates(['idx_x']) 
    resy_from_dup = res_from_dup.sort_values('pred_prob',ascending=False).drop_duplicates(['idx_y']) 
    bestxy_from_dup = resx_from_dup[resx_from_dup.idx_xy.isin(resy_from_dup.idx_xy)]
    print(f"best x,y from duplicated x,y:\t{bestxy_from_dup.shape[0]}")

    return bestxy_from_dup

def check_prob_dup_gap_cutoff(comp_vectors, way, prob, dup_gap=None):
    print(f"\nprob, dup_gap:\t{prob}\t{dup_gap}")
    
    if dup_gap is not None:
        print(f'given gap is {dup_gap}')

    prob_changed = comp_vectors[comp_vectors.pred_prob>=prob] 
    print(f"predicted matches:\t{prob_changed.shape[0]}")

    dedup_idx = prob_changed.sort_values('pred_prob',ascending=False).drop_duplicates('idx_x',keep=False) 
    print(f"x unique indexes:\t{dedup_idx.shape[0]}")
    dedup_idy = prob_changed.sort_values('pred_prob',ascending=False).drop_duplicates('idx_y',keep=False) 
    print(f"y unique indexes:\t{dedup_idy.shape[0]}")
    dedup_idxy = dedup_idx[dedup_idx.idx_xy.isin(dedup_idy.idx_xy)]
    print(f"xy unique indexes:\t{dedup_idxy.shape[0]}") 

    dup_idxx = prob_changed[prob_changed.idx_x.map(prob_changed.idx_x.value_counts())>1]
    print(f"x duplicated indexes:\t{dup_idxx.shape[0]}")
    sorted_prob_idxx = dup_idxx.sort_values('pred_prob',ascending=False) 
    idxx_grouped = sorted_prob_idxx.groupby('idx_x')['pred_prob'] 
    sorted_prob_idxx['prob_gap'] = idxx_grouped.transform(lambda x: x.nlargest(1)/x.nlargest(2).min()) 

    dup_idxy = prob_changed[prob_changed.idx_y.map(prob_changed.idx_y.value_counts())>1]
    print(f"y duplicated indexes:\t{dup_idxy.shape[0]}")
    sorted_prob_idxy = dup_idxy.sort_values('pred_prob',ascending=False) 
    idxy_grouped = sorted_prob_idxy.groupby('idx_y')['pred_prob'] 
    sorted_prob_idxy['prob_gap'] = idxy_grouped.transform(lambda x: x.nlargest(1)/x.nlargest(2).min()) 
    
    if dup_gap is not None:
        print(f"\nInner\ngiven prob, given dup_gap:\t{prob}\t{dup_gap}")
        bestxy_from_dup = check_dup_gap_cutoff(sorted_prob_idxx, sorted_prob_idxy, dup_gap)
        print(f"best x,y from duplicated x,y:\t{bestxy_from_dup.shape[0]}") 
        dedup_bestxy = pd.concat([dedup_idxy, bestxy_from_dup], axis=0, join='inner') 
        print(f"dedup + best x,y:\t{dedup_bestxy.shape[0]}")
        matched_result = get_matched_from_comp_vectors(dedup_bestxy, way)

    else:
        for gap in np.arange(1,2.05,0.1): ##
            print(f"\nMax dup_gap:\t{sorted_prob_idxx.prob_gap.max()}")
            print(f"\nInner\nProb, dup_gap:\t{prob}\t{gap}")
            bestxy_from_dup = check_dup_gap_cutoff(sorted_prob_idxx, sorted_prob_idxy, gap)
            print(f"best x,y from duplicated x,y:\t{bestxy_from_dup.shape[0]}")    
            dedup_bestxy = pd.concat([dedup_idxy, bestxy_from_dup], axis=0, join='inner') 
            print(f"dedup + best x,y:\t{dedup_bestxy.shape[0]}")

            matched_result = get_matched_from_comp_vectors(dedup_bestxy, way)

    return matched_result

def get_matched_from_comp_vectors(comp_vectors, way):
    print(f"dedup + best probs:\t{comp_vectors.shape[0]}")

    if way=='7500':
        uniq_1875 =_1875.loc[comp_vectors.idx_x] 
        uniq_1900 =_1900.loc[comp_vectors.idx_y] 
        uniq_1875.insert(0,'matched_id',range(0,0+len(uniq_1875))) 
        uniq_1900.insert(0,'matched_id',range(0,0+len(uniq_1900)))
        result = pd.merge(uniq_1875, uniq_1900, how='inner', on=['matched_id'])

    elif way=='0075':
        uniq_1900 =_1900.loc[comp_vectors.idx_x] 
        uniq_1875 =_1875.loc[comp_vectors.idx_y] 
        uniq_1875.insert(0,'matched_id',range(0,0+len(uniq_1875))) 
        uniq_1900.insert(0,'matched_id',range(0,0+len(uniq_1900)))
        result = pd.merge(uniq_1900, uniq_1875, how='inner', on=['matched_id']) 
    
    result['idx_xy']=comp_vectors['idx_xy'].values
    result['pred_prob']=comp_vectors['pred_prob'].values
    print(f"xy matches:\t{result.shape[0]}")

    TPFP = result.shape[0]
    TP = result[result.idx_xy.isin(idi_matches_gt.idx_xy)].shape[0]
    # TPFN = potential_pairs_gt_train[potential_pairs_gt_train.idx_xy.isin(idi_matches_gt.idx_xy)].shape[0] ##
    TPFN = potential_pairs_gt_test[potential_pairs_gt_test.idx_xy.isin(idi_matches_gt.idx_xy)].shape[0] ##

    print(f"GT> TP+FP:\t{TPFP}")
    print(f"GT> TP:\t{TP}")
    print(f"GT> TP+FN:\t{TPFN}") 

    if np.logical_and(TP!=0,TPFP!=0):
        print(f"Precision:\t{TP/TPFP}")
        print(f"Recall:\t{TP/TPFN}")
        print(f"F1score:\t{2*TP/(TPFP+TPFN)}") 
    
    return result
#%%
def get_ABEJW_unique_matched_from_comp_vectors(comp_vectors, way):
    dedup_idx = comp_vectors.drop_duplicates('idx_x',keep=False) 
    print(f"x unique indexes:\t{dedup_idx.shape[0]}")
    dedup_idy = comp_vectors.drop_duplicates('idx_y',keep=False) 
    print(f"y unique indexes:\t{dedup_idy.shape[0]}")
    dedup_idxy = dedup_idx[dedup_idx.idx_xy.isin(dedup_idy.idx_xy)]
    print(f"xy unique indexes:\t{dedup_idxy.shape[0]}") 
    gt_list = ['0432','1922','1924','1931','1933','1936','1938','1941','1943']

    if way=='7500':
        uniq_1875 =_1875.loc[dedup_idxy.idx_x] 
        uniq_1900 =_1900.loc[dedup_idxy.idx_y] 
        uniq_1875.insert(0,'matched_id',range(0,0+len(uniq_1875))) 
        uniq_1900.insert(0,'matched_id',range(0,0+len(uniq_1900)))
        result = pd.merge(uniq_1875, uniq_1900, how='inner', on=['matched_id']) 
        print(f'merged: {result.shape}') 
    elif way=='0075':
        uniq_1900 =_1900.loc[dedup_idxy.idx_x] 
        uniq_1875 =_1875.loc[dedup_idxy.idx_y] 
        uniq_1875.insert(0,'matched_id',range(0,0+len(uniq_1875))) 
        uniq_1900.insert(0,'matched_id',range(0,0+len(uniq_1900)))
        result = pd.merge(uniq_1900, uniq_1875, how='inner', on=['matched_id']) 
        print(f'merged: {result.shape}')

    result['idx_xy']=dedup_idxy['idx_xy'].values
    print(f"xy matches:\t{result.shape[0]}")

    
    TPFP = result.shape[0]
    TP = result[result.idx_xy.isin(idi_matches_gt.idx_xy)].shape[0]
    # TPFN = potential_pairs_gt_train[potential_pairs_gt_train.idx_xy.isin(idi_matches_gt.idx_xy)].shape[0] ##
    TPFN = potential_pairs_gt_test[potential_pairs_gt_test.idx_xy.isin(idi_matches_gt.idx_xy)].shape[0] ##
  
    print(f"GT> TP+FP:\t{TPFP}")
    print(f"GT> TP:\t{TP}")
    print(f"GT> TP+FN:\t{TPFN}") 

    if np.logical_and(TP!=0,TPFP!=0):
        print(f"Precision:\t{TP/TPFP}")
        print(f"Recall:\t{TP/TPFN}")
        print(f"F1score:\t{2*TP/(TPFP+TPFN)}") 
    
    return result

def ABEJW(df, jw_diff, birthP_diff, birthY_diff):
    return df[np.logical_and(
            np.logical_and(
                np.logical_and(
                    df.adj_birthP_diff<=birthP_diff, df.birthY_diff<=birthY_diff), 
                df.firstNs_JW<=jw_diff), 
            df.lastNs_JW<=jw_diff) 
            ]

def get_all_ABEJW(jw_diff=0.15, birthP_diff=0, birthY_diff=1):
    results = []
    for i in range(20):
        with open(f"data/tempresults/7500/compared_all_df_{i}.pkl", "rb") as fh: ## 75-00
            df = pickle.load(fh)
            print(i, df.shape)
            abe_df = ABEJW(df, jw_diff, birthP_diff, birthY_diff)
            print(i, abe_df.shape)
            results.append(abe_df)
            
    return pd.concat(results)
#%%

## check with match selection parameters
#%%
for direction in ("7500","0075"):
    with open(f"data/tempresults/{direction}/potential_pairs_gt.pkl", "rb") as fh:
        potential_pairs_gt = pickle.load(fh)
        potential_pairs_gt_train, potential_pairs_gt_test = train_test_split(potential_pairs_gt, test_size=0.1, random_state=0)
    with open(f"data/tempresults/{direction}/xlg_clf_gtl.pkl", "rb") as fh:
        xgb_clf_gtl = pickle.load(fh)
    with open(f"data/tempresults/{direction}/xlg_clf_gt.pkl", "rb") as fh:
        xgb_clf_gt = pickle.load(fh)        

    for var_scope in ('lim','ext'):
        if var_scope=='lim':
            sc = StandardScaler() 
            gtX = potential_pairs_gt.loc[:,limited_var].values
            gtX = sc.fit_transform(gtX)
            predicted_gt_train_lim = get_predicted_matches_comp_vectors(potential_pairs_gt_train,xgb_clf_gtl,limited_var)
            for i in np.arange(0.1,0.95,0.1): 
                check_prob_dup_gap_cutoff(predicted_gt_train_lim, direction, i) 
        elif var_scope=='ext':
            sc = StandardScaler() 
            gtX = potential_pairs_gt.loc[:,extended_var].values 
            gtX = sc.fit_transform(gtX)
            predicted_gt_train_ext = get_predicted_matches_comp_vectors(potential_pairs_gt_train, xgb_clf_gt, extended_var)
            for i in np.arange(0.1,0.95,0.1): 
                check_prob_dup_gap_cutoff(predicted_gt_train_ext, direction, i)

#%%

## Rule based model parameter selection
#%%
for i in np.arange(0.05, 0.22, 0.05): 
    for j in np.arange(0, 1.2, 1): 
        for k in np.arange(0, 3.2, 1): 
            print(f"jw_diff: {i}, birthP_diff: {j}, birthY_diff: {k}")
            abe = ABEJW(potential_pairs_gt_train, i, j, k)
            get_ABEJW_unique_matched_from_comp_vectors(abe, '7500')




##############################
# Linking nationwide
##############################
#%%
#%%
for direction in ("7500","0075"):
    with open(f"data/tempresults/{direction}/potential_pairs_gt.pkl", "rb") as fh:
        potential_pairs_gt = pickle.load(fh)
        potential_pairs_gt_train, potential_pairs_gt_test = train_test_split(potential_pairs_gt, test_size=0.1, random_state=0)
    with open(f"data/tempresults/{direction}/xlg_clf_gtl.pkl", "rb") as fh:
        xgb_clf_gtl = pickle.load(fh)
    with open(f"data/tempresults/{direction}/xlg_clf_gt.pkl", "rb") as fh:
        xgb_clf_gt = pickle.load(fh)        

    for var_scope in ('lim','ext'):
        if var_scope=='lim':
            sc = StandardScaler() 
            gtX = potential_pairs_gt.loc[:,limited_var].values
            gtX = sc.fit_transform(gtX)
            testall_lim = get_predicted_matches(xgb_clf_gtl, limited_var, direction) 
            matched_idx_xgb_lim = pd.concat(testall_lim)
            print(matched_idx_xgb_lim.shape)
            # Time invariant, Unique+Best
            matched_lim = check_prob_dup_gap_cutoff(matched_idx_xgb_lim, direction, 0.1, 1)
            matched_lim.to_pickle(f"data/tempresults/{direction}/matched_lim.pkl")
            # Time invariant, Unique
            matched_lim_03 = check_prob_dup_gap_cutoff(matched_idx_xgb_lim, direction, 0.3, 100)
            matched_lim_03.to_pickle(f"data/tempresults/{direction}/matched_lim_03.pkl") 
        elif var_scope=='ext':
            sc = StandardScaler() 
            gtX = potential_pairs_gt.loc[:,extended_var].values 
            gtX = sc.fit_transform(gtX)
            testall_ext = get_predicted_matches(xgb_clf_gt, extended_var, direction)
            matched_idx_xgb_ext = pd.concat(testall_ext)
            # Extended, Unique+Best
            matched_ext = check_prob_dup_gap_cutoff(matched_idx_xgb_ext, direction, 0.1, 1) 
            matched_ext.to_pickle(f"data/tempresults/{direction}/matched_ext.pkl")
            # Extended, Unique
            matched_ext_04 = check_prob_dup_gap_cutoff(matched_idx_xgb_ext, direction, 0.4, 100)
            matched_ext_04.to_pickle(f"data/tempresults/{direction}/matched_ext_04.pkl") 


## Rule-based
all_ABEJW = get_all_ABEJW(0.15,0,1)
matched_ABEJW = get_ABEJW_unique_matched_from_comp_vectors(all_ABEJW, '7500') # same as 00-75
print(matched_ABEJW.shape)
matched_ABEJW.to_pickle("data/tempresults/norge_matched_ABEJW_jw015p0y1.pkl")
#%%





#%%
#################################
# Two-way check
#################################
#%%
with open(f"data/tempresults/7500/matched_lim.pkl", "rb") as fh:
        matched_7500_lim = pickle.load(fh)
with open(f"data/tempresults/7500/matched_lim_03.pkl", "rb") as fh:
        matched_7500_lim_03 = pickle.load(fh)
with open(f"data/tempresults/7500/matched_ext.pkl", "rb") as fh:
        matched_7500_ext = pickle.load(fh)
with open(f"data/tempresults/7500/matched_ext_04.pkl", "rb") as fh:
        matched_7500_ext_04 = pickle.load(fh)
with open(f"data/tempresults/0075/matched_lim.pkl", "rb") as fh:
        matched_0075_lim = pickle.load(fh)
with open(f"data/tempresults/0075/matched_lim_03.pkl", "rb") as fh:
        matched_0075_lim_03 = pickle.load(fh)
with open(f"data/tempresults/0075/matched_ext.pkl", "rb") as fh:
        matched_0075_ext = pickle.load(fh)
with open(f"data/tempresults/0075/matched_ext_04.pkl", "rb") as fh:
        matched_0075_ext_04 = pickle.load(fh)
with open(f"data/tempresults/norge_matched_ABEJW_jw015p0y1.pkl", "rb") as fh:
        matched_ABEJW = pickle.load(fh)

#%%
matched_7500_lim['idxy_7500'] = list(zip(matched_7500_lim.idx_x, matched_7500_lim.idx_y)) 
matched_0075_lim['idxy_7500'] = list(zip(matched_0075_lim.idx_y, matched_0075_lim.idx_x)) 
matched_both_lim = matched_7500_lim[matched_7500_lim.idxy_7500.isin(matched_0075_lim.idxy_7500)]
matched_both_lim.shape
#%%
matched_7500_lim_03['idxy_7500'] = list(zip(matched_7500_lim_03.idx_x, matched_7500_lim_03.idx_y)) 
matched_0075_lim_03['idxy_7500'] = list(zip(matched_0075_lim_03.idx_y, matched_0075_lim_03.idx_x)) 
matched_both_lim_03 = matched_7500_lim_03[matched_7500_lim_03.idxy_7500.isin(matched_0075_lim_03.idxy_7500)]
matched_both_lim_03.shape
#%%
matched_7500_ext['idxy_7500'] = list(zip(matched_7500_ext.idx_x, matched_7500_ext.idx_y)) 
matched_0075_ext['idxy_7500'] = list(zip(matched_0075_ext.idx_y, matched_0075_ext.idx_x)) 
matched_both_ext = matched_7500_ext[matched_7500_ext.idxy_7500.isin(matched_0075_ext.idxy_7500)]
matched_both_ext.shape
#%%
matched_7500_ext_04['idxy_7500'] = list(zip(matched_7500_ext_04.idx_x, matched_7500_ext_04.idx_y)) 
matched_0075_ext_04['idxy_7500'] = list(zip(matched_0075_ext_04.idx_y, matched_0075_ext_04.idx_x)) 
matched_both_ext_04 = matched_7500_ext_04[matched_7500_ext_04.idxy_7500.isin(matched_0075_ext_04.idxy_7500)]
matched_both_ext_04.shape

#%%
matched_both_lim.to_pickle("data/tempresults/norge_matched_both_lim.pkl")
matched_both_lim_03.to_pickle("data/tempresults/norge_matched_both_lim_03.pkl")
matched_both_ext.to_pickle("data/tempresults/norge_matched_both_ext.pkl")
matched_both_ext_04.to_pickle("data/tempresults/norge_matched_both_ext_04.pkl")
#%%
## to load saved results
with open(f"data/tempresults/norge_matched_both_lim.pkl", "rb") as fh:
        matched_both_lim = pickle.load(fh)
with open(f"data/tempresults/norge_matched_both_lim_03", "rb") as fh:
        matched_both_lim_03 = pickle.load(fh)
with open(f"data/tempresults/norge_matched_both_ext.pkl", "rb") as fh:
        matched_both_ext = pickle.load(fh)
with open(f"data/tempresults/norge_matched_both_ext_04.pkl", "rb") as fh:
        matched_both_ext_04 = pickle.load(fh)
with open(f"data/tempresults/norge_matched_ABEJW_jw015p0y1.pkl", "rb") as fh:
        matched_ABEJW = pickle.load(fh)
        
#%%

# relationships
#%%
set1 = set(matched_both_lim_03.idx_xy)
set2 = set(matched_both_ext_04.idx_xy)
set3 = set(matched_ABEJW.idx_xy)

venn3([set1, set2, set3], ('Time invariant (Unique)', 'Extended (Unique)', 'Rule-based (ABE-JW)'))
plt.show()
#%%
set4 = set(matched_both_lim.idxy_7500)
set5 = set(matched_both_ext.idxy_7500)

venn2([set4, set5], ('Time invariant (Unique + Best)', 'Extended (Unique + Best)'))
plt.show()




########################
########################
# Evaluation
########################
########################
with open(f"data/tempresults/7500/xlg_clf_gtl.pkl", "rb") as fh:
    xgb_clf_gtl = pickle.load(fh)
with open(f"data/tempresults/7500/xlg_clf_gt.pkl", "rb") as fh:
    xgb_clf_gt = pickle.load(fh) 
########################
# 1. testset split from training data
########################
#%%
sc = StandardScaler() 
gtX = potential_pairs_gt.loc[:,extended_var].values 
gtX = sc.fit_transform(gtX)
predicted_gt_test_ext = get_predicted_matches_comp_vectors(potential_pairs_gt_test, xgb_clf_gt,extended_var)
#%%
sc = StandardScaler() 
gtX = potential_pairs_gt.loc[:,limited_var].values
gtX = sc.fit_transform(gtX)
predicted_gt_test_lim = get_predicted_matches_comp_vectors(potential_pairs_gt_test,xgb_clf_gtl,limited_var)
#%%
gt_test_lim_uniq_best = check_prob_dup_gap_cutoff(predicted_gt_test_lim, '7500', 0.2, 1) 
gt_test_lim_uniq = check_prob_dup_gap_cutoff(predicted_gt_test_lim, '7500', 0.3, 3) 
gt_test_ext_uniq_best = check_prob_dup_gap_cutoff(predicted_gt_test_ext, '7500', 0.2, 1) 
gt_test_ext_uniq = check_prob_dup_gap_cutoff(predicted_gt_test_ext, '7500', 0.4, 3) 
gt_test_abe = get_ABEJW_unique_matched_from_comp_vectors(ABEJW(potential_pairs_gt_test, 0.15, 0, 1), '7500') 
#%%
for i in np.arange(0.1,0.95,0.1): 
    # check_prob_dup_gap_cutoff(predicted_gt_test_lim, '7500', i) 
    check_prob_dup_gap_cutoff(predicted_gt_test_ext, '7500', i) 
#%%
gt_test_true = potential_pairs_gt_test[potential_pairs_gt_test.idx_xy.isin(idi_matches_gt.idx_xy)]
#%%
set1 = set(gt_test_lim_uniq_best.idx_xy)
set2 = set(gt_test_ext_uniq_best.idx_xy)
set3 = set(gt_test_true.idx_xy)

venn3([set1, set2, set3], ('Unique + Best (time-invariant)', 'Unique + Best (extended)', 'Testset'))
plt.show()
#%%
set1 = set(gt_test_lim_uniq.idx_xy)
set2 = set(gt_test_ext_uniq.idx_xy)
set3 = set(gt_test_true.idx_xy)

venn3([set1, set2, set3], ('Unique (time-invariant)', 'Unique (extended)', 'Testset'))
plt.show()
#%%
set1 = set(gt_test_lim_uniq.idx_xy)
set2 = set(gt_test_abe.idx_xy)
set3 = set(gt_test_true.idx_xy)

venn3([set1, set2, set3], ('Unique (time-invariant)', ' Rule-Based (ABE-JW)', 'Testset'))
plt.show()
#%%
set1 = set(gt_test_lim_uniq.idx_xy)
set2 = set(gt_test_ext_uniq.idx_xy)
set3 = set(gt_test_abe.idx_xy)

venn3([set1, set2, set3], ('Unique (time-invariant)', 'Unique (extended)', 'Rule-Based (ABE-JW)'))
plt.show()
#%%
########################
# 2.testset provided by the NHDC
########################
#%%
testset = pd.read_csv("data/sources/LinkToVerify.txt", delimiter='\t')
#%%
testset.id_d1.value_counts() 
#%%
testset.id_d2.value_counts() 
#%%
dedup_d1_test = testset.drop_duplicates('id_d1',keep=False) 
dedup_d1_test.shape 
#%%
dedup_d2_test = testset.drop_duplicates('id_d2',keep=False) 
dedup_d2_test.shape 
#%%
dedup_d1_test['id_d1_d2'] = list(zip(dedup_d1_test['id_d1'], dedup_d1_test['id_d2']))
dedup_d1_test 
#%%
dedup_d2_test['id_d1_d2'] = list(zip(dedup_d2_test['id_d1'], dedup_d2_test['id_d2']))
dedup_d2_test 
#%%
dedup_test = dedup_d1_test[dedup_d1_test.id_d1_d2.isin(dedup_d2_test.id_d1_d2)]
dedup_test.shape
#%%
_75_gt_list = ['0432','1922','1924','1931','1933','1936','1938','1941','1943']
#%%
_00_gt_list = ['0432','0433','1922','1924','1930','1931','1932','1933','1935','1936','1937','1938','1941','1942','1943'] 
#%%
_75_testset = dedup_test[~dedup_test.id_d1.str[:4].isin(_75_gt_list)]
_75_testset.shape 
#%%
_00_testset = dedup_test[~dedup_test.id_d2.str[:4].isin(_00_gt_list)]
_00_testset.shape 
#%%
dedup_deGT_test = _75_testset[_75_testset.id_d1_d2.isin(_00_testset.id_d1_d2)]
dedup_deGT_test.shape 
#%%
final_testset = dedup_deGT_test[dedup_deGT_test.id_d2.isin(_1900_20.id_d)]
final_testset.shape 
#%%
final_testset.to_pickle("data/sources/final_testset.pkl")
#%%
with open(f"data/sources/final_testset.pkl", "rb") as fh:
    final_testset = pickle.load(fh)

#%%
kommnr2_list = ['01','02','03','04','05','06','07','08','09','10','11','12','13','14','15','16','17','18','19','20']
#%%
tmp_list = []
for i in (kommnr2_list):
    print(i)
    tmp_list.append(final_testset[final_testset.id_d1.str[:2]==i].sample(500, random_state=0))
#%%
region_dist_final_testset = pd.concat(tmp_list, axis=0)
region_dist_final_testset.shape 
#%%
region_dist_final_testset.to_pickle("data/sources/region_dist_final_testset.pkl")
#%%
with open(f"data/sources/region_dist_final_testset.pkl", "rb") as fh:
    region_dist_final_testset = pickle.load(fh)
#%%
def get_testset(df, testset):
    dataset = []
    testset_df_idx = df[df.id_d.isin(testset.id_d1)].index # 75-00
    # testset_df_idx = df[df.id_d.isin(testset.id_d2)].index # 00-75
    for i in range(20):
        with open(f"data/tempresults/7500/compared_all_df_{i}.pkl", "rb") as fh: # 75-00
        # with open(f"data/tempresults/0075/compared_all_df_{i}.pkl", "rb") as fh: # 00-75
            cand = pickle.load(fh)
            dataset.append(cand[cand.idx_x.isin(testset_df_idx)])
    return dataset
#%%
def test_dataset(df, testset):
    df['id_d_xy'] = list(zip(df['id_d_x'], df['id_d_y'])) # 75-00, both
    #df['id_d_xy'] = list(zip(df['id_d_y'], df['id_d_x'])) # 00-75

    TPFP = df.shape[0]
    TP = testset[testset.id_d1_d2.isin(df.id_d_xy)].shape[0]
    TPFN = testset.shape[0]
    print(f'TP+FP:\t{TPFP}')
    print(f'TP:\t{TP}')
    print(f'TP+FN:\t{TPFN}')
    print(f'Precision:\t{TP/TPFP}')
    print(f'Recall\t{TP/TPFN}')
    print(f'f1-score\t{2*TP/(TPFP+TPFN)}')
#%%
full_testset_list_75 = get_testset(_1875, final_testset)
full_testset_75 = pd.concat(full_testset_list_75, axis=0)
full_testset_75.shape 
#%%
full_testset_list_00 = get_testset(_1900, final_testset)
full_testset_00 = pd.concat(full_testset_list_00, axis=0)
full_testset_00.shape 
#%%
full_region_dist_testset_list_75 = get_testset(_1875, region_dist_final_testset)
full_region_dist_testset_75 = pd.concat(full_region_dist_testset_list_75, axis=0)
full_region_dist_testset_75.shape 
#%%
full_region_dist_testset_list_00 = get_testset(_1900, region_dist_final_testset)
full_region_dist_testset_00 = pd.concat(full_region_dist_testset_list_00, axis=0)
full_region_dist_testset_00.shape 
#%%
########################
# (1) full data set
########################
#%%
full_testset_75.shape 
#%%
final_testset.shape 
#%%
sc = StandardScaler() 
gtX = potential_pairs_gt.loc[:,extended_var].values 
gtX = sc.fit_transform(gtX)
predicted_rhd_test_ext = get_predicted_matches_comp_vectors(full_testset_75, xgb_clf_gt,extended_var)
#%%
sc = StandardScaler() 
gtX = potential_pairs_gt.loc[:,limited_var].values
gtX = sc.fit_transform(gtX)
predicted_rhd_test_lim = get_predicted_matches_comp_vectors(full_testset_75,xgb_clf_gtl,limited_var)
#%%
rhd_test_lim_uniq_best = check_prob_dup_gap_cutoff(predicted_rhd_test_lim, '7500', 0.2, 1) 
rhd_test_lim_uniq = check_prob_dup_gap_cutoff(predicted_rhd_test_lim, '7500', 0.3, 3) 
rhd_test_ext_uniq_best = check_prob_dup_gap_cutoff(predicted_rhd_test_ext, '7500', 0.2, 1) 
rhd_test_ext_uniq = check_prob_dup_gap_cutoff(predicted_rhd_test_ext, '7500', 0.4, 3) 
rhd_test_abe = get_ABEJW_unique_matched_from_comp_vectors(ABEJW(full_testset_75, 0.15, 0, 1), '7500') 
#%%
for i in np.arange(0.1,0.95,0.1): 
    for j in np.arange(1,2.05,0.1):
        tmp = check_prob_dup_gap_cutoff(predicted_rhd_test_lim, '7500', i, j)
        # tmp = check_prob_dup_gap_cutoff(predicted_rhd_test_ext, '7500', i, j)
        test_dataset(tmp, final_testset)
#%%
for i in np.arange(0.1,0.95,0.1): 
    # tmp = check_prob_dup_gap_cutoff(predicted_rhd_test_lim, '7500', i, 100)
    tmp = check_prob_dup_gap_cutoff(predicted_rhd_test_ext, '7500', i, 100)
    test_dataset(tmp, final_testset)
#%%
test_dataset(rhd_test_lim_uniq_best, final_testset)
#%%
test_dataset(rhd_test_lim_uniq, final_testset)
#%%
test_dataset(rhd_test_ext_uniq_best, final_testset)
#%%
test_dataset(rhd_test_ext_uniq, final_testset)
#%%
test_dataset(rhd_test_abe, final_testset)
#%%
set1 = set(rhd_test_lim_uniq_best.id_d_xy)
set2 = set(rhd_test_ext_uniq_best.id_d_xy)
set3 = set(final_testset.id_d1_d2)

venn3([set1, set2, set3], ('Unique + Best (time-invariant)', 'Unique + Best (extended)', 'Testset'))
plt.show()
#%%
set1 = set(rhd_test_lim_uniq.id_d_xy)
set2 = set(rhd_test_ext_uniq.id_d_xy)
set3 = set(final_testset.id_d1_d2)

venn3([set1, set2, set3], ('Unique (time-invariant)', 'Unique (extended)', 'Testset'))
plt.show()
#%%
set1 = set(rhd_test_lim_uniq.id_d_xy)
set2 = set(rhd_test_abe.id_d_xy)
set3 = set(final_testset.id_d1_d2)

venn3([set1, set2, set3], ('Unique (time-invariant)', ' Rule-Based (ABE-JW)', 'Testset'))
plt.show()
#%%
#%%
set1 = set(rhd_test_lim_uniq.idx_xy)
set2 = set(rhd_test_ext_uniq.idx_xy)
set3 = set(rhd_test_abe.idx_xy)

venn3([set1, set2, set3], ('Unique (time-invariant)', 'Unique (extended)', 'Rule-Based (ABE-JW)'))
plt.show()
#%%
########################
# (2) sub test set
########################
#%%
full_region_dist_testset_75.shape 
#%%
region_dist_final_testset.shape 
#%%
sc = StandardScaler() 
gtX = potential_pairs_gt.loc[:,extended_var].values 
gtX = sc.fit_transform(gtX)
predicted_rhd_rg_test_ext = get_predicted_matches_comp_vectors(full_region_dist_testset_75, xgb_clf_gt,extended_var)
#%%
sc = StandardScaler() 
gtX = potential_pairs_gt.loc[:,limited_var].values
gtX = sc.fit_transform(gtX)
predicted_rhd_rg_test_lim = get_predicted_matches_comp_vectors(full_region_dist_testset_75,xgb_clf_gtl,limited_var)
#%%
rhd_rg_test_lim_uniq_best = check_prob_dup_gap_cutoff(predicted_rhd_rg_test_lim, '7500', 0.2, 1) 
rhd_rg_test_lim_uniq = check_prob_dup_gap_cutoff(predicted_rhd_rg_test_lim, '7500', 0.3, 3) 
rhd_rg_test_ext_uniq_best = check_prob_dup_gap_cutoff(predicted_rhd_rg_test_ext, '7500', 0.2, 1) 
rhd_rg_test_ext_uniq = check_prob_dup_gap_cutoff(predicted_rhd_rg_test_ext, '7500', 0.4, 3) 
rhd_rg_test_abe = get_ABEJW_unique_matched_from_comp_vectors(ABEJW(full_region_dist_testset_75, 0.15, 0, 1), '7500') 
#%%
for i in np.arange(0.1,0.95,0.1): 
    for j in np.arange(1,2.05,0.1):
        # tmp = check_prob_dup_gap_cutoff(predicted_rhd_rg_test_lim, '7500', i, j)
        tmp = check_prob_dup_gap_cutoff(predicted_rhd_rg_test_ext, '7500', i, j)
        test_dataset(tmp, region_dist_final_testset)
#%%
for i in np.arange(0.1,0.95,0.1): 
    # tmp = check_prob_dup_gap_cutoff(predicted_rhd_rg_test_lim, '7500', i, 100)
    tmp = check_prob_dup_gap_cutoff(predicted_rhd_rg_test_ext, '7500', i, 100)
    test_dataset(tmp, region_dist_final_testset)
#%%
test_dataset(rhd_rg_test_lim_uniq_best, region_dist_final_testset)
#%%
test_dataset(rhd_rg_test_lim_uniq, region_dist_final_testset)
#%%
test_dataset(rhd_rg_test_ext_uniq_best, region_dist_final_testset)
#%%
test_dataset(rhd_rg_test_ext_uniq, region_dist_final_testset)
#%%
test_dataset(rhd_rg_test_abe, region_dist_final_testset)
#%%
set1 = set(rhd_rg_test_lim_uniq_best.id_d_xy)
set2 = set(rhd_rg_test_ext_uniq_best.id_d_xy)
set3 = set(region_dist_final_testset.id_d1_d2)

venn3([set1, set2, set3], ('Unique + Best (time-invariant)', 'Unique + Best (extended)', 'Testset'))
plt.show()
#%%
set1 = set(rhd_rg_test_lim_uniq.id_d_xy)
set2 = set(rhd_rg_test_ext_uniq.id_d_xy)
set3 = set(region_dist_final_testset.id_d1_d2)

venn3([set1, set2, set3], ('Unique (time-invariant)', 'Unique (extended)', 'Testset'))
plt.show()
#%%
set1 = set(rhd_rg_test_lim_uniq.id_d_xy)
set2 = set(rhd_rg_test_abe.id_d_xy)
set3 = set(region_dist_final_testset.id_d1_d2)

venn3([set1, set2, set3], ('Unique (time-invariant)', ' Rule-Based (ABE-JW)', 'Testset'))
plt.show()
#%%
set1 = set(rhd_rg_test_lim_uniq.idx_xy)
set2 = set(rhd_rg_test_ext_uniq.idx_xy)
set3 = set(rhd_rg_test_abe.idx_xy)

venn3([set1, set2, set3], ('Unique (time-invariant)', 'Unique (extended)', 'Rule-Based (ABE-JW)'))
plt.show()
#%%








########################
########################
# Representativeness
########################
########################

#####################
# Changes in characteristics over time
#####################
#%%
with open("data/tempresults/_1875_all_preprocessedN.pkl", "rb") as fh:
    _1875 = pickle.load(fh)
with open("data/tempresults/_1900_all_preprocessedN.pkl", "rb") as fh:
    _1900 = pickle.load(fh)
_1900_20 = _1900[np.logical_and((_1900.faar<1881), (_1900.faar!=0))]
_1900_25 = _1900[np.logical_and((_1900.faar<1876), (_1900.faar!=0))]
#%%
with open(f"data/tempresults/norge_matched_both_lim.pkl", "rb") as fh:
    lim = pickle.load(fh)
with open(f"data/tempresults/norge_matched_both_lim_03.pkl", "rb") as fh:
    lim_03 = pickle.load(fh)
with open(f"data/tempresults/norge_matched_both_ext.pkl", "rb") as fh:
    ext = pickle.load(fh)
with open(f"data/tempresults/norge_matched_both_ext_04.pkl", "rb") as fh:
    ext_04 = pickle.load(fh)
with open(f"data/tempresults/7500/norge_7500_matched_ABEJW_jw015p0y1.pkl", "rb") as fh:
    abe = pickle.load(fh)
#%%
data = [_1900_20, _1900_25, lim, lim_03, ext, ext_04, abe]
#%%
######## 
# Size
########
for i,v in enumerate(data):
    print(v.shape[0])
#%%
######## 
# Age
######## 
_1900_20m = (1900-_1900_20.faar).mean()
_1900_25m = (1900-_1900_25.faar).mean()
print(_1900_20m)
print(_1900_25m)

for i,v in enumerate(data):
    if i<2:
        print(f'Data {i}:\t{(1900-v.faar).mean()}\t{(1900-v.faar).std()}')
        print(stats.ttest_1samp((1900-v.faar), _1900_25m))
    else:
        print(f'Data {i}:\t{(1900-v.faar_y).mean()}\t{(1900-v.faar_y).std()}')
        print(stats.ttest_1samp((1900-v.faar_y), _1900_25m))
#%%
#%%
# 25-45y (1875: 0-20y)
for i,v in enumerate(data):
    if i<2:
        print(f'Data {i}:\t{v[np.logical_and(25<=(1900-v.faar),(1900-v.faar)<45)].shape[0]}\t{v[np.logical_and(25<=(1900-v.faar),(1900-v.faar)<45)].shape[0]/v.shape[0]}')
    else:
        print(f'Data {i}:\t{v[np.logical_and(25<=(1900-v.faar_y),(1900-v.faar_y)<45)].shape[0]}\t{v[np.logical_and(25<=(1900-v.faar_y),(1900-v.faar_y)<45)].shape[0]/v.shape[0]}')
#%%
# 45-60y (1875: 20-35y)
for i,v in enumerate(data):
    if i<2:
        print(f'Data {i}:\t{v[np.logical_and(45<=(1900-v.faar),(1900-v.faar)<60)].shape[0]}\t{v[np.logical_and(45<=(1900-v.faar),(1900-v.faar)<60)].shape[0]/v.shape[0]}')
    else:
        print(f'Data {i}:\t{v[np.logical_and(45<=(1900-v.faar_y),(1900-v.faar_y)<60)].shape[0]}\t{v[np.logical_and(45<=(1900-v.faar_y),(1900-v.faar_y)<60)].shape[0]/v.shape[0]}')
#%%
# over 60y (1875: over 35y)
for i,v in enumerate(data):
    if i<2:
        print(f'Data {i}:\t{v[60<=(1900-v.faar)].shape[0]}\t{v[60<=(1900-v.faar)].shape[0]/v.shape[0]}')
    else:
        print(f'Data {i}:\t{v[60<=(1900-v.faar_y)].shape[0]}\t{v[60<=(1900-v.faar_y)].shape[0]/v.shape[0]}')
#%%
######## 
# Family size
######## 
_1900_20m = _1900_20.family_size.mean()
_1900_25m = _1900_25.family_size.mean()
print(_1900_20m)
print(_1900_25m)

for i,v in enumerate(data):
    if i<2:
        print(f'Data {i}:\t{v.family_size.mean()}\t{v.family_size.std()}')
        print(stats.ttest_1samp(v.family_size, _1900_25m))
    else:
        print(f'Data {i}:\t{v.family_size_y.mean()}\t{v.family_size_y.std()}')
        print(stats.ttest_1samp(v.family_size_y, _1900_25m))
#%%
# size 1
for i,v in enumerate(data):
    if i<2:
        print(f'Data {i}:\t{v[v.family_size<=1].shape[0]}\t{v[v.family_size<=1].shape[0]/v.shape[0]}')
    else:
        print(f'Data {i}:\t{v[v.family_size_y<=1].shape[0]}\t{v[v.family_size_y<=1].shape[0]/v.shape[0]}')
#%%
# size 2-4
for i,v in enumerate(data):
    if i<2:
        print(f'Data {i}:\t{v[np.logical_and(1<v.family_size,v.family_size<=4)].shape[0]}\t{v[np.logical_and(1<v.family_size,v.family_size<=4)].shape[0]/v.shape[0]}')
    else:
        print(f'Data {i}:\t{v[np.logical_and(1<v.family_size_y,v.family_size_y<=4)].shape[0]}\t{v[np.logical_and(1<v.family_size_y,v.family_size_y<=4)].shape[0]/v.shape[0]}')
#%%
# size 5-9
for i,v in enumerate(data):
    if i<2:
        print(f'Data {i}:\t{v[np.logical_and(4<v.family_size,v.family_size<=9)].shape[0]}\t{v[np.logical_and(4<v.family_size,v.family_size<=9)].shape[0]/v.shape[0]}')
    else:
        print(f'Data {i}:\t{v[np.logical_and(4<v.family_size_y,v.family_size_y<=9)].shape[0]}\t{v[np.logical_and(4<v.family_size_y,v.family_size_y<=9)].shape[0]/v.shape[0]}')
#%%
# size 7-10
for i,v in enumerate(data):
    if i<2:
        print(f'Data {i}:\t{v[np.logical_and(7<v.family_size,v.family_size<=10)].shape[0]}\t{v[np.logical_and(7<v.family_size,v.family_size<=10)].shape[0]/v.shape[0]}')
    else:
        print(f'Data {i}:\t{v[np.logical_and(7<v.family_size_y,v.family_size_y<=10)].shape[0]}\t{v[np.logical_and(7<v.family_size_y,v.family_size_y<=10)].shape[0]/v.shape[0]}')
#%%
# size over 10
for i,v in enumerate(data):
    if i<2:
        print(f'Data {i}:\t{v[10<=v.family_size].shape[0]}\t{v[10<=v.family_size].shape[0]/v.shape[0]}')
    else:
        print(f'Data {i}:\t{v[10<=v.family_size_y].shape[0]}\t{v[10<=v.family_size_y].shape[0]/v.shape[0]}')
#%%
######## 
# Sex
######## 
_1900_20.kjonn.value_counts(normalize=True)
#%%
_1900_25.kjonn.value_counts(normalize=True)
#%%
for i,v in enumerate(data):
    if i<2:
        print(f'Data {i}:\n{v.kjonn.value_counts(normalize=True)}')
    else:
        print(f'Data {i}:\n{v.kjonn_y.value_counts(normalize=True)}')

base_df = pd.DataFrame(_1900_25.kjonn.value_counts(normalize=True))
base = (base_df.loc['k'].values.item(),base_df.loc['m'].values.item())
base = np.array(base)+(base_df.loc['!!'].values.item()/2)
base
#%%
for i,v in enumerate(data):
    if i<2:
        print(f'Data {i}:\n{v.kjonn.value_counts(normalize=True)}')
    else:
        expected = [v.shape[0]*base[0], v.shape[0]*base[1]] 
        print(f"Data {i}> expected: {expected}")
        v_df = pd.DataFrame(v.kjonn_y.value_counts())
        observed = [v_df.loc['k'].values.item(),v_df.loc['m'].values.item()]
        print(f"Data {i}> observed: {observed}")
        print(stats.chisquare(observed, expected))
#%%
######## 
# marital status
######## 
#%%
for i,v in enumerate(data):
    if i<2:
        print(f'Data {i}:\n{v.sivst.value_counts(normalize=True)}')
    else:
        print(f'Data {i}:\n{v.sivst_y.value_counts(normalize=True)}')

    
base_df = pd.DataFrame(_1900_25.sivst.value_counts(normalize=True))
base = (base_df.loc['ug'].values.item(),base_df.loc['g'].values.item(),base_df.loc['e'].values.item(),
        base_df.loc['s'].values.item(),base_df.loc['f'].values.item(),base_df.loc['!!'].values.item())
base
#%%
for i,v in enumerate(data):
    if i<2:
        print(f'Data {i}:\n{v.sivst.value_counts(normalize=True)}')
    else:
        expected = [v.shape[0]*base[0], v.shape[0]*base[1], v.shape[0]*base[2], v.shape[0]*base[3], v.shape[0]*base[4], v.shape[0]*base[5]]
        print(f"Data {i}> expected: {expected}")
        v_df = pd.DataFrame(v.sivst_y.value_counts())
        observed = (v_df.loc['ug'].values.item(),v_df.loc['g'].values.item(),v_df.loc['e'].values.item(),
                    v_df.loc['s'].values.item(),v_df.loc['f'].values.item(),v_df.loc['!!'].values.item())
        print(f"Data {i}> observed: {observed}")
        print(stats.chisquare(observed, expected))
#%%
########
# living municipality == birth municipality
########
#%%
for i,v in enumerate(data):
    if i<2:
        print(f'Data {i}:\t{v[v.adj_fsted_kode==v.adj_kommnr].shape}')
        print(f'Data {i}:\t{v[v.adj_fsted_kode==v.adj_kommnr].shape[0]/v.shape[0]*100}')
    else:
        print(f'Data {i}:\t{v[v.adj_fsted_kode_y==v.adj_kommnr_y].shape}')
        print(f'Data {i}:\t{v[v.adj_fsted_kode_y==v.adj_kommnr_y].shape[0]/v.shape[0]*100}')
#%%
########
# living municipality (1875) == birth municipality (1900)
########
#%%
for i,v in enumerate(data):
    if i<2:
        pass
    else:
        print(f'Data {i}:\t{v[v.adj_kommnr_x==v.adj_kommnr_y].shape}')
        print(f'Data {i}:\t{v[v.adj_kommnr_x==v.adj_kommnr_y].shape[0]/v.shape[0]*100}')

#%%
########
# living in urban areas
########
#%%
for i,v in enumerate(data):
    if i<2:
        print(f'Data {i}:\t{v[v.adj_kommnr.str[2]=="0"].shape[0]}\t{v[v.adj_kommnr.str[2]=="0"].shape[0]/v.shape[0]}')
    else:
        print(f'Data {i} 1900 :\t{v[v.adj_kommnr_y.str[2]=="0"].shape[0]}\t{v[v.adj_kommnr_y.str[2]=="0"].shape[0]/v.shape[0]}')

#%%
########
# living in rural areas
########
#%%
for i,v in enumerate(data):
    if i<2:
        print(f'Data {i}:\t{v[v.adj_kommnr.str[2]!="0"].shape[0]}\t{v[v.adj_kommnr.str[2]!="0"].shape[0]/v.shape[0]}')
    else:
        print(f'Data {i} 1900 :\t{v[v.adj_kommnr_y.str[2]!="0"].shape[0]}\t{v[v.adj_kommnr_y.str[2]!="0"].shape[0]/v.shape[0]}')
#%%
########
# born in rural areas
########
#%%
for i,v in enumerate(data):
    if i<2:
        print(f'Data {i}:\t{v[v.adj_fsted_kode.str[2]!="0"].shape[0]}\t{v[v.adj_fsted_kode.str[2]!="0"].shape[0]/v.shape[0]}')
    else:
        print(f'Data {i} 1900 :\t{v[v.adj_fsted_kode_y.str[2]!="0"].shape[0]}\t{v[v.adj_fsted_kode_y.str[2]!="0"].shape[0]/v.shape[0]}')
########
# birthplace region
########
#%%
east = ['01','02','03','04','05','06','07','08']
south = ['09','10']
west = ['11','12','13','14','15']
mid = ['16','17']
north = ['18','19','20']
regions = [east, south, west, mid, north]
#%%
for i,v in enumerate(data):
    if i<2:
        for j in range(len(regions)):

            print(f'Data {i}, region {j}:\t{v[v.adj_fsted_kode.str[:2].isin(regions[j])].shape[0]}\t{v[v.adj_fsted_kode.str[:2].isin(regions[j])].shape[0]/v.shape[0]}')
    else:
        for j in range(len(regions)):
            print(f'Data {i}, region {j}:\t{v[v.adj_fsted_kode_y.str[:2].isin(regions[j])].shape[0]}\t{v[v.adj_fsted_kode_y.str[:2].isin(regions[j])].shape[0]/v.shape[0]}')
#%%
########
# municipality region
########
#%%
for i,v in enumerate(data):
    if i<2:
        for j in range(len(regions)):
            print(f'Data {i}, region {j}:\t{v[v.adj_kommnr.str[:2].isin(regions[j])].shape[0]}\t{v[v.adj_kommnr.str[:2].isin(regions[j])].shape[0]/v.shape[0]}')
    else:
        for j in range(len(regions)):
            print(f'Data {i}, region {j}:\t{v[v.adj_kommnr_y.str[:2].isin(regions[j])].shape[0]}\t{v[v.adj_kommnr_y.str[:2].isin(regions[j])].shape[0]/v.shape[0]}')
#%%
########
# rates of cities by region
########
#%%
for i,v in enumerate(data):
    if i<2:
        for j in range(len(regions)):
            region = v[v.adj_kommnr.str[:2].isin(regions[j])]
            print(f'Data {i}, region {j}:\t{region[region.adj_kommnr.str[2]=="0"].shape[0]}\t{region.shape[0]}\t{region[region.adj_kommnr.str[2]=="0"].shape[0]/region.shape[0]}')
    else:
        for j in range(len(regions)):
            region = v[v.adj_kommnr_y.str[:2].isin(regions[j])]
            print(f'Data {i}, region {j}:\t{region[region.adj_kommnr_y.str[2]=="0"].shape[0]}\t{region.shape[0]}\t{region[region.adj_kommnr_y.str[2]=="0"].shape[0]/region.shape[0]}')
#%%
########
# family relationship
########
#%%
for i,v in enumerate(data):
    if i<2:
        print(f'Data {i}:\n{v.famst_ipums_f2.value_counts(normalize=True)}')
        
    else:
        print(f'Data {i}:\n{v.famst_ipums_f2_y.value_counts(normalize=True)}')
#%%
base_df = pd.DataFrame(_1900_25.famst_ipums_f2.value_counts(normalize=True))
base_df
base = (base_df.loc['01'].values.item(),base_df.loc['02'].values.item(),base_df.loc['03'].values.item(),base_df.loc['04'].values.item())
base
#%%
for i,v in enumerate(data):
    if i<2:
        print(f'Data {i}:\n{v.famst_ipums_f2.value_counts(normalize=True)}')
    else:
        expected = [v.shape[0]*base[0], v.shape[0]*base[1], v.shape[0]*base[2], v.shape[0]*base[3]]
        print(f"Data {i}> expected: {expected}")
        v_df = pd.DataFrame(v.famst_ipums_f2_y.value_counts())
        observed = (v_df.loc['01'].values.item(),v_df.loc['02'].values.item(),v_df.loc['03'].values.item(),v_df.loc['04'].values.item())
        print(f"Data {i}> observed: {observed}")
        print(stats.chisquare(observed, expected))
#%%


#####################
# Changes in characteristics over time
#####################
#%%
pd.options.display.float_format = '{:,.2f}'.format 
np.set_printoptions(precision=2, suppress=True) 
pd.set_option('display.precision', 2) 
#%%
def heatmap_matched_migration(df):
    dfcopy = df.copy()
    dfcopy['kommnr_x2'] = df.kommnr_x.str[:2]
    dfcopy['kommnr_y2'] = df.kommnr_y.str[:2]
    dfcopy_gr = dfcopy.groupby(['kommnr_x2','kommnr_y2'])
    dfcopy_gr_count = dfcopy_gr.size()
    dfcopy_gr_count_unstack = dfcopy_gr_count.unstack()
    print(dfcopy_gr_count_unstack.shape)
    
    return dfcopy_gr_count_unstack.div(dfcopy_gr_count_unstack.sum(axis=1), axis=0).mul(100).round(2).fillna(0) # rate
    #return dfcopy_gr_count_unstack.fillna(0) # number
#%%
def heatmap_matched_famst_ipums(df):
    dfcopy = df.copy()
    dfcopy_gr = dfcopy.groupby(['famst_ipums_f2_x','famst_ipums_f2_y'])
    dfcopy_gr_count = dfcopy_gr.size()
    dfcopy_gr_count_unstack = dfcopy_gr_count.unstack()
    print(dfcopy_gr_count_unstack.shape)
    
    return dfcopy_gr_count_unstack.div(dfcopy_gr_count_unstack.sum(axis=1), axis=0).mul(100).round(2).fillna(0) # rate 
    # return dfcopy_gr_count_unstack.fillna(0) # number
#%%
def heatmap_matched_marital_status(df):
    dfcopy = df.copy()
    dfcopy_gr = dfcopy.groupby(['sivst_x','sivst_y'])
    dfcopy_gr_count = dfcopy_gr.size()
    dfcopy_gr_count_unstack = dfcopy_gr_count.unstack()
    print(dfcopy_gr_count_unstack.shape)
    
    return dfcopy_gr_count_unstack.div(dfcopy_gr_count_unstack.sum(axis=1), axis=0).mul(100).round(2).fillna(0) # rate
    #return dfcopy_gr_count_unstack.fillna(0) # number
#%%
######## 
# kommnr 
######## 
#%%
data = [_1900_20, _1900_25, lim, lim_03, ext, ext_04, abe]
#%%
# hm = heatmap_matched_migration(data[2])
# hm = heatmap_matched_migration(data[4])
hm = heatmap_matched_migration(data[6])
hm = hm.rename_axis('1875')
hm = hm.rename_axis('1900',axis='columns')
hm.style.background_gradient(cmap='Blues',axis=0,vmin=0,vmax=10) 
#%%
######## 
# FAMST_IPUMS 
######## 
#%%
# hm = heatmap_matched_famst_ipums(data[2])
# hm = heatmap_matched_famst_ipums(data[4])
hm = heatmap_matched_famst_ipums(data[6])
hm.style.background_gradient(cmap='Blues',axis=1) 
#%%
######## 
# Marital status 
######## 
#%%
# hm = heatmap_matched_marital_status(data[2])
# hm = heatmap_matched_marital_status(data[4])
hm = heatmap_matched_marital_status(data[6])
hm = hm.loc[['!!','ug','g','e','f'],['!!','ug','g','e','f','s']]
hm.style.background_gradient(cmap='Blues',axis=1) 
#%%

