#%matplotlib inline
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter, defaultdict
import networkx as nx
from scipy import sparse
from pymc3.math import logsumexp
import pymc3 as pm
import theano.tensor as tt

#global parameters
##
#To select high usage BTSs, we consider BTSs that serve a minimum number of users
min_users_per_bts = 2000
#We are interested in cluster that share a given minimum number of users in common
#This number should be relative to the number of served users by BTS and its neighbors: *research point!
min_com_users = 500
#input file
#Network data
#IN_FILE = 'D:/Master Project/dataset/location/DPA_cellid_date_20170126.csv'
#data type 
float_type = np.float32

#read data file
df = pd.read_csv(IN_FILE,header = 0)

#process the list of BTS for each user
df_ = df['cellid_set_count_1d_MOD'].str.split(',')
def get_bs(x):
    bs = []
    num = []
    try:
        for i in x:
            try:
                i_ = i.split(':')
                if (i_[0] != ''):
                    if (np.int(i_[1]) > 0):
                        bs.append(i_[0])
                        num.append(np.int(i_[1]))
            except:
                0
    except:
        0
    return bs
df_ = pd.DataFrame(df_)
#the column 'bss' contain the list of BTS for each user 
df_['bss'] = df_['cellid_set_count_1d_MOD'].apply(get_bs)
#filter out users with no BTSs
df_ = df_[df_.bss.apply(len) != 0]

#if we want to work with a fraction of the data, then set frac to less than 1
frac = 1
if frac < 1:
    df_sample  = df_.sample(frac=frac)
else:
    df_sample = df_
#%%
#create a data frame with BTS as index and their users in a column
#this gives the BTS view of the data
# columns: 'ucount' the count of users, 'users': list of users
def df_bts_(df_):
    bs_list = []
    for i in df_.bss.values:
        bs_list.extend(i)
    l = Counter(bs_list)
    df = pd.DataFrame.from_dict(l, orient='index')
    df.columns = ['ucount']
    bs_us_dict = defaultdict(list)
    for i in df_.index:
        for j in df_.loc[i,'bss']:
            bs_us_dict[j].append(i)
    us_ = pd.Series(bs_us_dict)
    us_.name = 'users'
    df = df.join(us_)
    return df

df_bts = df_bts_(df_sample)

#create list of BTS with at least the minimum number of users
large_bts = list(df_bts[df_bts['ucount']>=min_users_per_bts].index)
#compute some statistics of BTS usage
print('Number of BTSs (ignoreing null):', len(df_bts['ucount']))
print('minimum users in a BTS:', df_bts['ucount'].min())
print('maximum users in a BTS:', df_bts['ucount'].max())
print('average users in BTSs:', np.round(df_bts['ucount'].mean()))
print('median of users in BTSs:', df_bts['ucount'].median())
print('mode(s) of users in BTSs:', df_bts['ucount'].mode().values)
print('Number of BTS with 1 user:', len(df_bts[df_bts['ucount'] == 1]))
print('Number of large BTSs (more than', min_users_per_bts,'users):', len(large_bts))
#%%

#ax = df_bts.ucount.hist(bins = 20)
#plt.title('Distribution of number of users among BTSs')
#plt.xlabel('number of users')
#plt.ylabel('number of BTSs')	

#create the graph with large_bts as nodes
#the weight of an edge is the number of users in common
#this is a time costly step, for total data set takes around 20 min in a PC with 32GB RAM and i7 
def nx_graph(df, large_bts):
    B = nx.Graph()
    l = df.index
        
    for i in l:
        large_i = list(set(large_bts) & set(df.loc[i,'bss']))
        A = nx.complete_graph(large_i)
        Ae = A.edges()
        for e in Ae:
            if not B.has_edge(e[0],e[1]):
                B.add_edge(e[0],e[1])
                B[e[0]][e[1]]['weight'] = 1
            else:
                B[e[0]][e[1]]['weight'] += 1
    return B
B = nx_graph(df_sample, large_bts)
print('Number of nodes and edges:',len(B.nodes), len(B.edges))

#remove edges with users less than min_com_users
def remove_small_edges(B,min_com_users):
    edges = list(B.edges())
    for e in edges:
        if B[e[0]][e[1]]['weight'] < min_com_users:
            B.remove_edge(*e)
small_B = B.copy()
remove_small_edges(small_B,min_com_users)
print('Number of nodes and edges after removing low weight edges:',len(small_B.nodes), len(small_B.edges))

#get the 0-1 adjency matrix of the graph
Adj = nx.adj_matrix(small_B, nodelist=large_bts, weight=None)
print('shape of Adjacency matrix: ',Adj.shape)
#np.savetxt('/raid60/anil.gaddam/adjacency_BTS.csv',Adj, delimiter=',')

K= 100 #50
B= Adj.shape[0]

#%%
dir_model = pm.Model()
with dir_model:
    
    pi = pm.Dirichlet('pi', a = np.ones(K), shape = K)

    dri = pm.Dirichlet('dri', a = np.ones((K, B)), shape = (K, B))
    
    category_U = pm.Categorical('category_u', p = pi, shape = B)
    
    vector_U = pm.Bernoulli('vec_u', p = dri[category_U], observed = Adj.toarray())
    
#%%
#with dir_model:
#    inference = pm.ADVI()
#    approx = pm.fit(1000,method = inference)
#    tr = approx.sample(1000)               
#%%
#Use MCMC sampling
with dir_model:
    tr = pm.sample(200, chains = 1)
np.savetxt('/raid60/anil.gaddam/50clusters.csv',np.argmax(tr['dri'].mean(axis=0), axis = 0))

   



