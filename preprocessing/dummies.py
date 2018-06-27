# encoding: utf-8
""" 特征工程
"""

import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv('log.csv', sep="\t")
print(df)

df = df.drop(['id'], axis=1)
print(df)
'''
    name  age     edu
0    jim   15  little
1  lilei   16  middle
2    han   18    high
'''

# 哑变量
edu = pd.get_dummies(df['edu'])
edu = edu.rename(columns=lambda x: "edu_" + str(x))
df = pd.concat([df, edu], axis=1)
print(df)


# plt.hist(df['age'].values, alpha=0.2)
# plt.show()

# plt.hist(mean_np, normed=False, alpha=0.2)
# plt.show()


