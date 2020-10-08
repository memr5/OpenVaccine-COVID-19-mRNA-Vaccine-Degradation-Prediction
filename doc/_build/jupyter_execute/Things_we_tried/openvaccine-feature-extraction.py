#!/usr/bin/env python
# coding: utf-8

# # Feature extraction

# 
# ## Install Packages

# In[1]:


get_ipython().run_cell_magic('capture', '', '!pip install forgi\n!yes Y |conda install -c bioconda viennarna')


# ## Library Imports

# In[2]:


import os, math, random
from collections import Counter

import RNA
import subprocess
from forgi.graph import bulge_graph
import forgi.visual.mplotlib as fvm
from IPython.display import Image, SVG

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from colorama import Fore, Back, Style


# ## Data Import

# In[3]:


path = '/kaggle/input/stanford-covid-vaccine'
train_df = pd.read_json(f'{path}/train.json',lines=True)
test_df = pd.read_json(f'{path}/test.json', lines=True)
sub_df = pd.read_csv(f'{path}/sample_submission.csv')

print('Train set sequences: ', train_df.shape)
print('Test set sequences: ', test_df.shape)


# ## Train Data Overview

# In[4]:


train_df.head()


# ## Test Data Overview

# In[5]:


test_df.head()


# ## New Feature Overview

# In[6]:


Select_id = "id_001f94081"


# In[7]:


Sequence = train_df[train_df['id'] == Select_id]["sequence"].values[0]
structure = train_df[train_df['id'] == Select_id]["structure"].values[0]
predicted_loop_type = train_df[train_df['id'] == Select_id]["predicted_loop_type"].values[0]
bg, = bulge_graph.BulgeGraph.from_fasta_text('>seq\n' + Sequence + '\n' + structure)
print("Sequence :",Sequence)
print("Structure :",structure)
print("Predicted Loop type :",predicted_loop_type)
print("Generated Loop type :", bg.to_element_string())


# In[8]:


plt.figure(figsize=(10,10))
fvm.plot_rna(bg, text_kwargs={"fontweight":"black"}, lighten=0.7,
             backbone_kwargs={"linewidth":3})
plt.show()


# ## Generating Graph Matrices from the Structures
#    * [Referance](https://www.kaggle.com/theoviel/generating-graph-matrices-from-the-structures)

# In[9]:


def build_matrix(couples, size):
    mat = np.zeros((size, size))
    
    for i in range(size):  # neigbouring bases are linked as well
        if i < size - 1:
            mat[i, i + 1] = 1
        if i > 0:
            mat[i, i - 1] = 1
    
    for i, j in couples:
        mat[i, j] = 1
        mat[j, i] = 1
        
    return mat


# In[10]:


def get_couples(structure):
    """
    For each closing parenthesis, I find the matching opening one and store their index in the couples list.
    The assigned list is used to keep track of the assigned opening parenthesis
    """
    opened = [idx for idx, i in enumerate(structure) if i == '(']
    closed = [idx for idx, i in enumerate(structure) if i == ')']

    assert len(opened) == len(closed)


    assigned = []
    couples = []

    for close_idx in closed:
        for open_idx in opened:
            if open_idx < close_idx:
                if open_idx not in assigned:
                    candidate = open_idx
            else:
                break
        assigned.append(candidate)
        couples.append([candidate, close_idx])
        
    assert len(couples) == len(opened)
    
    return couples


# ## 1. Abs-Difference between Graph Matrix and BPPS

# In[11]:


couples = get_couples(structure)
mat = build_matrix(couples, len(structure))

fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(11.35, 5))

im = axes[0].imshow(mat, interpolation='none',cmap='gray')
axes[0].set_title('Graph of the structure')
# axes[0].pcolormesh(adj_mat, )

bpp = np.load(path +f"/bpps/{Select_id}.npy")

im = axes[1].imshow(bpp, interpolation='none',cmap='gray')
axes[1].set_title('BPP Matrix')

im = axes[2].imshow(abs(mat-bpp), interpolation='none',cmap='gray')
axes[2].set_title('Abs-Difference(mat-bpps) Matrix')

plt.show()

print("Abs-Difference Value : ",abs(mat-bpp).sum())


# ## 2. Loop-Count

# In[12]:


print(f"{len(list(bg.stem_iterator()))} stems")
print(f"{len(list(bg.iloop_iterator()))} interior loops")
print(f"{len(list(bg.mloop_iterator()))} multiloops")
print(f"{len(list(bg.hloop_iterator()))} hairpin loops")
print(f"{len(list(bg.floop_iterator()))} fiveprimes")
print(f"{len(list(bg.tloop_iterator()))} threeprimes")


# ## 3. Iterate over all pairs of connected stems.

# In[13]:


list(bg.connected_stem_iterator())


# ## 4. Elements Present in RNA

# In[14]:


stem_elements = list(bg.stem_iterator())
iloop_elements = list(bg.iloop_iterator())
mloop_elements = list(bg.mloop_iterator())
hloop_elements = list(bg.hloop_iterator())
floop_elements = list(bg.floop_iterator())
tloop_elements = list(bg.tloop_iterator())

elements = stem_elements + iloop_elements + mloop_elements + hloop_elements + floop_elements + tloop_elements
print(list(elements))


# ## 5. Element_Length of each element.

# In[15]:


list(map(bg.element_length,elements))


# ## 6. A list containing the sequence(s) corresponding to the defines

# In[16]:


list(map(bg.get_define_seq_str,elements))


# ## 7. Create a minimum spanning tree from this BulgeGraph. This is useful

# In[17]:


print(bg.get_mst())


# ## 8. Get the minimum and maximum base pair distance between 2 elements

# In[18]:


# Example
bg.min_max_bp_distance('s1','s0')


# In[19]:


min_sum = 0
max_sum = 0
for e1 in stem_elements:
    for e2 in iloop_elements:
        min_dis,max_dis = bg.min_max_bp_distance(e1,e2)
        min_sum += min_dis
        max_sum += max_dis
print("Sum of Min and Max Distances between stem and iloop elements:",min_sum,max_sum)
min_norm_distance = min_sum/(len(stem_elements)+len(iloop_elements))
max_norm_distance = max_sum/(len(stem_elements)+len(iloop_elements))
print("Normalised Min and Max Distances between stem and iloop elements:",min_norm_distance,max_norm_distance)


# In[20]:


distance = 0
for e1 in stem_elements:
    for e2 in iloop_elements:
        distance += bg.ss_distance(e1,e2)
norm_distance = distance/(len(stem_elements)+len(iloop_elements))
print("norm_distance between stem and iloop elements:",norm_distance)


# ## 9. Simple MFE prediction for a given sequence

# In[21]:


# compute minimum free energy (MFE) and corresponding structure
(structure, mfe) = RNA.fold(Sequence)

print('Sequence : ',Sequence)
print('structure : ', structure)
print("MFE(Minimum Free energy) : ",mfe)


# In[22]:


seq = Sequence
# create fold_compound data structure (required for all subsequently applied  algorithms)
fc = RNA.fold_compound(seq)
# compute MFE and MFE structure
(mfe_struct, mfe) = fc.mfe()
# rescale Boltzmann factors for partition function computation
fc.exp_params_rescale(mfe)
# compute partition function
(pp, pf) = fc.pf()
# compute centroid structure
(centroid_struct, dist) = fc.centroid()
# compute free energy of centroid structure
centroid_en = fc.eval_structure(centroid_struct)
# compute MEA structure
(MEA_struct, MEA) = fc.MEA()
# compute free energy of MEA structure
MEA_en = fc.eval_structure(MEA_struct)
# print everything like RNAfold -p --MEA
print("Original Sequence: ",Sequence)
print("Original structure: ",structure)
print(f"MFE Structure : {mfe_struct} ,\nMFE : [{mfe}] ")
print(f"centroid structure Structure : {centroid_struct} ,\nCFE : [{centroid_en}] ")
print(f"MEA Structure : {MEA_struct} ,\nMEA : [{MEA_en}] ")


# In[23]:


fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize = (22, 8))
bg1, = bulge_graph.BulgeGraph.from_fasta_text('>seq\n' + Sequence + '\n' + mfe_struct)
fvm.plot_rna(bg1, text_kwargs={"fontweight":"black"}, lighten=0.7,
             backbone_kwargs={"linewidth":0.5},ax=ax2)
ax1.set_title('MFE Structure : ', fontsize=16)


bg2, = bulge_graph.BulgeGraph.from_fasta_text('>seq\n' + Sequence + '\n' + structure)
fvm.plot_rna(bg2, text_kwargs={"fontweight":"black"}, lighten=0.7,
             backbone_kwargs={"linewidth":0.5},ax=ax1)
ax2.set_title('Original Structure ', fontsize=16)

bg3, = bulge_graph.BulgeGraph.from_fasta_text('>seq\n' + Sequence + '\n' + centroid_struct)
fvm.plot_rna(bg3, text_kwargs={"fontweight":"black"}, lighten=0.7,
             backbone_kwargs={"linewidth":0.5},ax=ax3)
ax3.set_title('Centroid Structure ', fontsize=16)

bg4, = bulge_graph.BulgeGraph.from_fasta_text('>seq\n' + Sequence + '\n' + MEA_struct)
fvm.plot_rna(bg4, text_kwargs={"fontweight":"black"}, lighten=0.7,
             backbone_kwargs={"linewidth":0.5},ax=ax4)
ax4.set_title('MEA Structure ', fontsize=16)

plt.show()


# #### [https://www.tbi.univie.ac.at/RNA/ViennaRNA/doc/html/helloworld_swig.html#helloworld_python](https://www.tbi.univie.ac.at/RNA/ViennaRNA/doc/html/helloworld_swig.html#helloworld_python)

# ## Loop-type (Non Sequencial) Probablity

# In[24]:


def loop_count(row):
    Sequence = row["sequence"]
    structure = row["structure"]
    len_ = row["seq_length"]
    bg, = bulge_graph.BulgeGraph.from_fasta_text('>seq\n' + Sequence + '\n' + structure)
    return [len(list(bg.stem_iterator()))/len_, len(list(bg.iloop_iterator()))/len_, 
            len(list(bg.mloop_iterator()))/len_, len(list(bg.hloop_iterator()))/len_, 
            len(list(bg.floop_iterator()))/len_, len(list(bg.tloop_iterator()))/len_]


# In[25]:


loops_name = ['stems','interior_loops', 'multiloops', 'hairpin loops', 'fiveprimes','threeprimes']
train_df[loops_name] = train_df.apply(loop_count, axis=1,result_type="expand")
test_df[loops_name] = test_df.apply(loop_count, axis=1,result_type="expand")
train_df[['id']+loops_name].head()


# In[26]:


fig, _ax = plt.subplots(nrows=2,ncols=3,figsize=(20,10))
fig.suptitle("Train Data New Features Histograms (Loop-Count)", fontsize=20,)
for i,_ax in enumerate(_ax.ravel()):
    mean_value = train_df[loops_name[i]].mean()
    max_value_index,max_value = Counter(train_df[loops_name[i]]).most_common(1)[0]
    _ax.hist(x=train_df[loops_name[i]],bins='auto', color='#0504aa', alpha=1, rwidth=1)
    _ax.set(ylabel=f"'{loops_name[i]}' Frequency", title= f"'{loops_name[i]}' Histogram")
    _ax.axvline(x=mean_value, color='r', label= 'Average',linewidth=2)
    _ax.axvline(x=max_value_index, color='y', label= 'Max',linewidth=2)
    _ax.legend([f"Average : {mean_value:0.2f}",f"Max Frequency : {max_value}", "Hist"], loc ="upper right")
plt.show()


# In[27]:


# Train Data New Features correlation(Loop Count)
corr = train_df[loops_name].corr()
sns.set_context("notebook", font_scale=1.0, rc={"lines.linewidth": 2.5})
plt.figure(figsize=(7,5))
plt.title("Train Data New Features correlation (Loop-Count): ")
mask = np.zeros_like(corr)
mask[np.triu_indices_from(mask, 1)] = True
a = sns.heatmap(corr,mask=mask, annot=True, fmt='.2f')
rotx = a.set_xticklabels(a.get_xticklabels(), rotation=90)
roty = a.set_yticklabels(a.get_yticklabels(), rotation=30)


# * Loop Count Features are not highly correlated so it is good for us

# ## process_inputs_2 : [here](https://www.kaggle.com/kank2130/covax-gru-lstm)

# In[28]:


def process_inputs_2(df):
    df1 = df.copy()
    df2 = df.copy()
    df3 = df.copy()
    df4 = df.copy()
    df5 = df.copy()
    from collections import Counter as count
    bases = []
    for j in range(len(df1)):
        counts = dict(count(df1.iloc[j]['sequence']))
        bases.append((
            counts['A'] / df1.iloc[j]['seq_length'],
            counts['G'] / df1.iloc[j]['seq_length'],
            counts['C'] / df1.iloc[j]['seq_length'],
            counts['U'] / df1.iloc[j]['seq_length']
        ))

    bases = pd.DataFrame(bases, columns=['A_percent', 'G_percent', 'C_percent', 'U_percent'])
    del df1
    print("Done : ['A_percent', 'G_percent', 'C_percent', 'U_percent']")
    
    pairs = []
    all_partners = []
    for j in range(len(df2)):
        partners = [-1 for i in range(130)]
        pairs_dict = {}
        queue = []
        for i in range(0, len(df2.iloc[j]['structure'])):
            if df2.iloc[j]['structure'][i] == '(':
                queue.append(i)
            if df2.iloc[j]['structure'][i] == ')':
                first = queue.pop()
                try:
                    pairs_dict[(df2.iloc[j]['sequence'][first], df2.iloc[j]['sequence'][i])] += 1
                except:
                    pairs_dict[(df2.iloc[j]['sequence'][first], df2.iloc[j]['sequence'][i])] = 1

                partners[first] = i
                partners[i] = first

        all_partners.append(partners)

        pairs_num = 0
        pairs_unique = [('U', 'G'), ('C', 'G'), ('U', 'A'), ('G', 'C'), ('A', 'U'), ('G', 'U')]
        for item in pairs_dict:
            pairs_num += pairs_dict[item]
        add_tuple = list()
        for item in pairs_unique:
            try:
                add_tuple.append(pairs_dict[item]/pairs_num)
            except:
                add_tuple.append(0)
        pairs.append(add_tuple)

    pairs = pd.DataFrame(pairs, columns=['U-G', 'C-G', 'U-A', 'G-C', 'A-U', 'G-U'])
    del df2
    print("Done : ['U-G', 'C-G', 'U-A', 'G-C', 'A-U', 'G-U']")
    
    pairs_rate = []
    for j in range(len(df3)):
        res = dict(count(df3.iloc[j]['structure']))
        pairs_rate.append(res['('] / (df3.iloc[j]['seq_length']/2))

    pairs_rate = pd.DataFrame(pairs_rate, columns=['pairs_rate'])
    del df3
    
    loops = []
    for j in range(len(df4)):
        counts = dict(count(df4.iloc[j]['predicted_loop_type']))
        available = ['E', 'S', 'H', 'B', 'X', 'I', 'M']
        row = []
        for item in available:
            try:
                row.append(counts[item] / df4.iloc[j]['seq_length'])
            except:
                row.append(0)
        loops.append(row)

    loops = pd.DataFrame(loops, columns=available)
    del df4
    print("Done : ['E', 'S', 'H', 'B', 'X', 'I', 'M']")
    
    return pd.concat([df5, bases, pairs, loops, pairs_rate], axis=1)


# In[29]:


# ['A_percent', 'G_percent', 'C_percent', 'U_percent']
# ['U-G', 'C-G', 'U-A', 'G-C', 'A-U', 'G-U']
# ['pairs_rate']
# ['E', 'S', 'H', 'B', 'X', 'I', 'M']

train_df = process_inputs_2(train_df)
train_df.head()


# In[30]:


test_df = process_inputs_2(test_df)
test_df.head()


# In[31]:


ns_columns = ['A_percent','G_percent', 'C_percent', 'U_percent', 
            'U-G', 'C-G', 'U-A', 'G-C','A-U', 'G-U', 'E', 'S', 
            'H', 'B', 'X', 'I', 'M', 'pairs_rate'] + loops_name

def add_axis(row):
    s_len = row[1]
    val = row[0]
    return np.array([val]*int(s_len)).ravel()

for c in ns_columns:
    print(c)
    train_df[c] = train_df[[c,"seq_length"]].apply(add_axis,axis=1)
    test_df[c] = test_df[[c,"seq_length"]].apply(add_axis,axis=1)


# In[32]:


train_df[ns_columns].head()


# ## Sequencial Feature Generation 

# ## Implementing Sliding Window Features

# In[33]:


seq_features = ['sequence','structure','predicted_loop_type']


# ### Sliding Window Pair Features

# In[34]:


def pair_feature(row):
    arr = list(row)
    its = [iter(['_']+arr[:]) ,iter(arr[1:]+['_'])]
    list_touple = list(zip(*its))
    return list(map("".join,list_touple))


# In[35]:


print("Sequence: ",Sequence)
print("pair_feature len: ",len(pair_feature(Sequence)))
print("pair_feature : ",(pair_feature(Sequence)))


# In[36]:


# for col in seq_features:
#     train_df['pair_AB_'+col] = train_df[col].apply(pair_feature)
#     test_df['pair_AB_'+col] = test_df[col].apply(pair_feature)
# train_df[seq_features].head()


# ## Compute MFE,MEA ,Centroid Structure

# In[37]:


def compute_structure(seq):
    # create fold_compound data structure (required for all subsequently applied  algorithms)
    fc = RNA.fold_compound(seq)
    # compute MFE and MFE structure
    (mfe_struct, mfe) = fc.mfe()
    # rescale Boltzmann factors for partition function computation
    fc.exp_params_rescale(mfe)
    # compute partition function
    (pp, pf) = fc.pf()
    # compute centroid structure
    (centroid_struct, dist) = fc.centroid()
    # compute MEA structure
    (MEA_struct, MEA) = fc.MEA()
    return mfe_struct,centroid_struct,MEA_struct


# In[38]:


for (id_,se,st) in zip(train_df['id'],train_df['sequence'], train_df['structure']):
    mfe_struct,centroid_struct,MEA_struct = compute_structure(se)
    train_df.loc[train_df['id'] == id_,'mfe_structure'] =  mfe_struct
    train_df.loc[train_df['id'] == id_,'centroid_structure'] =  centroid_struct
    train_df.loc[train_df['id'] == id_,'MEA_structure'] =  MEA_struct
    
for (id_,se,st) in zip(test_df['id'],test_df['sequence'], test_df['structure']):
    mfe_struct,centroid_struct,MEA_struct = compute_structure(se)
    test_df.loc[test_df['id'] == id_,'mfe_structure'] =  mfe_struct
    test_df.loc[test_df['id'] == id_,'centroid_structure'] =  centroid_struct
    test_df.loc[test_df['id'] == id_,'MEA_structure'] =  MEA_struct


# In[39]:


train_df[['structure','mfe_structure','centroid_structure','MEA_structure']].head()


# ## Loop TypeFeaure Generation (MFE,MEA ,Centroid Structure)

# In[40]:


bg, = bulge_graph.BulgeGraph.from_fasta_text('>seq\n' + Sequence + '\n' + structure)
print(bg.to_element_string(with_numbers=True).split('\n'))
print(predicted_loop_type)


# In[41]:


def create_element_string(row):
    
    Sequence = row["sequence"]
    structure = row["structure"]
    bg, = bulge_graph.BulgeGraph.from_fasta_text('>seq\n' + Sequence + '\n' + structure)
    return bg.to_element_string(with_numbers=True).split('\n')

#  |   Create a string similar to dotbracket notation that identifies what
#  |      type of element is present at each location.
#  |      
#  |      For example the following dotbracket:
#  |      
#  |      ..((..))..
#  |      
#  |      Should yield the following element string:
#  |      
#  |      ffsshhsstt
#  |      
#  |      Indicating that it begins with a fiveprime region, continues with a
#  |      stem, has a hairpin after the stem, the stem continues and it is terminated
#  |      by a threeprime region.
#  |      
#  |      :param with_numbers: show the last digit of the element id in a second line.::
#  |      
#  |                               (((.(((...))))))
#  |      
#  |                           Could result in::
#  |      
#  |                               sssissshhhssssss
#  |                               0000111000111000
#  |      
#  |                           Indicating that the first stem is named 's0', followed by 'i0','
#  |                           s1', 'h0', the second strand of 's1' and the second strand of 's0'


# In[42]:


# train_df[['LP_type_value','LP_type_index']] = train_df.apply(create_element_string, axis=1,result_type="expand").head()


# ### Alternate Method

# In[43]:


for (id_,se,st,mfe,cent,mea) in zip(train_df['id'],train_df['sequence'], train_df['structure'],train_df['mfe_structure'], train_df['centroid_structure'], train_df['MEA_structure']):
    
    bg, = bulge_graph.BulgeGraph.from_fasta_text('>seq\n' + se + '\n' + mfe)
    LP_type_value,LP_type_index = bg.to_element_string(with_numbers=True).split('\n')
    train_df.loc[train_df['id'] == id_,'mfe_predicted_loop_type'] =  LP_type_value
    train_df.loc[train_df['id'] == id_,'mfe_predicted_loop_type_inxex'] =  LP_type_index
    
    bg, = bulge_graph.BulgeGraph.from_fasta_text('>seq\n' + se + '\n' + cent)
    LP_type_value,LP_type_index = bg.to_element_string(with_numbers=True).split('\n')
    train_df.loc[train_df['id'] == id_,'centroid_predicted_loop_type'] =  LP_type_value
    train_df.loc[train_df['id'] == id_,'centroid_predicted_loop_type_inxex'] =  LP_type_index
    
    bg, = bulge_graph.BulgeGraph.from_fasta_text('>seq\n' + se + '\n' + mea)
    LP_type_value,LP_type_index = bg.to_element_string(with_numbers=True).split('\n')
    train_df.loc[train_df['id'] == id_,'mea_predicted_loop_type'] =  LP_type_value
    train_df.loc[train_df['id'] == id_,'mea_predicted_loop_type_inxex'] =  LP_type_index
    


# In[44]:


train_df[['mfe_predicted_loop_type','mfe_predicted_loop_type_inxex','centroid_predicted_loop_type','centroid_predicted_loop_type_inxex','mea_predicted_loop_type','mea_predicted_loop_type_inxex']].tail()


# In[45]:


for (id_,se,st,mfe,cent,mea) in zip(test_df['id'],test_df['sequence'], test_df['structure'],test_df['mfe_structure'], test_df['centroid_structure'], test_df['MEA_structure']):
    
    bg, = bulge_graph.BulgeGraph.from_fasta_text('>seq\n' + se + '\n' + mfe)
    LP_type_value,LP_type_index = bg.to_element_string(with_numbers=True).split('\n')
    test_df.loc[test_df['id'] == id_,'mfe_predicted_loop_type'] =  LP_type_value
    test_df.loc[test_df['id'] == id_,'mfe_predicted_loop_type_inxex'] =  LP_type_index
    
    bg, = bulge_graph.BulgeGraph.from_fasta_text('>seq\n' + se + '\n' + cent)
    LP_type_value,LP_type_index = bg.to_element_string(with_numbers=True).split('\n')
    test_df.loc[test_df['id'] == id_,'centroid_predicted_loop_type'] =  LP_type_value
    test_df.loc[test_df['id'] == id_,'centroid_predicted_loop_type_inxex'] =  LP_type_index
    
    bg, = bulge_graph.BulgeGraph.from_fasta_text('>seq\n' + se + '\n' + mea)
    LP_type_value,LP_type_index = bg.to_element_string(with_numbers=True).split('\n')
    test_df.loc[test_df['id'] == id_,'mea_predicted_loop_type'] =  LP_type_value
    test_df.loc[test_df['id'] == id_,'mea_predicted_loop_type_inxex'] =  LP_type_index


# In[46]:


test_df[['mfe_predicted_loop_type','mfe_predicted_loop_type_inxex','centroid_predicted_loop_type','centroid_predicted_loop_type_inxex','mea_predicted_loop_type','mea_predicted_loop_type_inxex']].tail()


# ## BPPS Sequence Feature

# In[47]:


def build_matrix(couples, size):
    mat = np.zeros((size, size))
    
    for i in range(size):  # neigbouring bases are linked as well
        if i < size - 1:
            mat[i, i + 1] = 1
        if i > 0:
            mat[i, i - 1] = 1
    
    for i, j in couples:
        mat[i, j] = 1
        mat[j, i] = 1
        
    return mat

def get_couples(structure):
    """
    For each closing parenthesis, I find the matching opening one and store their index in the couples list.
    The assigned list is used to keep track of the assigned opening parenthesis
    """
    opened = [idx for idx, i in enumerate(structure) if i == '(']
    closed = [idx for idx, i in enumerate(structure) if i == ')']

    assert len(opened) == len(closed)


    assigned = []
    couples = []

    for close_idx in closed:
        for open_idx in opened:
            if open_idx < close_idx:
                if open_idx not in assigned:
                    candidate = open_idx
            else:
                break
        assigned.append(candidate)
        couples.append([candidate, close_idx])
        
    assert len(couples) == len(opened)
    
    return couples


# In[48]:


def BPPS_Mat(row):
    structure = row["structure"]
    couples = get_couples(structure)
    mat = build_matrix(couples, len(structure))
    Select_id = row['id']
    bpp = np.load(path +f"/bpps/{Select_id}.npy")
    BPPS_Mat_min = np.min(mat-bpp,axis=0)
    BPPS_Mat_average = np.average(mat-bpp,axis=0)
    BPPS_Max = np.max(bpp,axis=0)
    
    return [list(BPPS_Mat_min),list(BPPS_Mat_average), 
            list(BPPS_Max)]


# In[49]:


list_cols = ['BPPS_Mat_min','BPPS_Mat_average','BPPS_Max']


# In[50]:


# train_df[list_cols] = train_df.apply(BPPS_Mat, axis=1,result_type="expand")
# test_df[list_cols] = test_df.apply(BPPS_Mat, axis=1,result_type="expand")


# ### checking

# In[51]:


# exploded = [train_df[col].explode() for col in list_cols]
# train_expanded = pd.DataFrame(dict(zip(list_cols, exploded)))
# train_expanded.reset_index(drop=True, inplace=True)
# train_expanded[list_cols]  = train_expanded[list_cols].astype(str).astype(float)
# train_expanded.head()


# In[52]:


# # Train Data New Features correlation (Loop Count)
# corr = train_expanded.corr()
# sns.set_context("notebook", font_scale=1.0, rc={"lines.linewidth": 2.5})
# plt.figure(figsize=(7,5))
# plt.title("Train Data New Features correlation (BPPS_Mat_min_row, BPPS_Mat_min_col): ")
# mask = np.zeros_like(corr)
# mask[np.triu_indices_from(mask, 1)] = True
# a = sns.heatmap(corr,mask=mask, annot=True, fmt='.4f')
# rotx = a.set_xticklabels(a.get_xticklabels(), rotation=90)
# roty = a.set_yticklabels(a.get_yticklabels(), rotation=30)


# ## Pair Map

# In[53]:


def mk_pair_map(structure, type='pm'):
    pm = np.full(len(structure), -1, dtype=float)
    pd = np.full(len(structure), -1, dtype=float)
    queue = []
    for i, s in enumerate(structure):
        if s == "(":
            queue.append(i)
        elif s == ")":
            j = queue.pop()
            pm[i] = j/len(structure)
            pm[j] = i/len(structure)
            pd[i] = (i-j)/len(structure)
            pd[j] = (i-j)/len(structure)
    if type == 'pm':
        return pm
    elif type == 'pd':
        return pd


# In[54]:


train_df['pair_map'] = train_df.structure.apply(mk_pair_map, type='pm')
test_df['pair_map'] = test_df.structure.apply(mk_pair_map, type='pm')

train_df['pair_distance'] = train_df.structure.apply(mk_pair_map, type='pd')
test_df['pair_distance'] = test_df.structure.apply(mk_pair_map, type='pd')


# ## Final Train And Test Data

# In[55]:


train_df.to_csv('train.csv', index=False)
train_df.head()


# In[56]:


train_df.columns


# In[57]:


test_df.columns


# In[58]:


test_df.to_csv('test.csv', index=False)
test_df.head()


# In[59]:


print(train_df.shape)
print(test_df.shape)

