#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 15:34:08 2020

@author: Aisling
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

data = np.load('gan_generation1.npy')
listg = data.tolist()

data1 = np.load('d_n.npy')
list1 = data1.tolist()

c0 = [list1[i][0] for i in range(0,len(list1))]
c1 = [list1[i][1] for i in range(0,len(list1))]
c2 = [list1[i][2] for i in range(0,len(list1))]
c3 = [list1[i][3] for i in range(0,len(list1))]
c4 = [list1[i][4] for i in range(0,len(list1))]
c5 = [list1[i][5] for i in range(0,len(list1))]

g0 = [listg[i][0] for i in range(0,len(listg))]
g1 = [listg[i][1] for i in range(0,len(listg))]
g2 = [listg[i][2] for i in range(0,len(listg))]
g3 = [listg[i][3] for i in range(0,len(listg))]
g4 = [listg[i][4] for i in range(0,len(listg))]
g5 = [listg[i][5] for i in range(0,len(listg))]


sns.distplot(c0, hist=False, rug=False,label='dataset')
sns.distplot(g0, hist=False, rug=False,label='dataset')

plt.title('Length 1')
plt.ylabel('frequency')
plt.show()
sns.distplot(c1, hist=False, rug=False,label='dataset')
sns.distplot(g1, hist=False, rug=False,label='dataset')

plt.title('Length 2')
plt.ylabel('frequency')
plt.show()
sns.distplot(c2, hist=False, rug=False,label='dataset')
sns.distplot(g2, hist=False, rug=False,label='dataset')

plt.title('Length 3')
plt.ylabel('frequency')
plt.show()
sns.distplot(c3, hist=False, rug=False,label='dataset')
sns.distplot(g3, hist=False, rug=False,label='dataset')

plt.title('Length 4')
plt.ylabel('frequency')
plt.show()
sns.distplot(c4, hist=False, rug=False,label='dataset')
sns.distplot(g4, hist=False, rug=False,label='dataset')

plt.title('Length 5')
plt.ylabel('frequency')
plt.show()
sns.distplot(c5, hist=False, rug=False,label='dataset')
sns.distplot(g5, hist=False, rug=False,label='dataset')

plt.title('Length 6')
plt.ylabel('frequency')
plt.show()

plt.title('Distribution of GAN Generations')
plt.ylabel('frequency')


