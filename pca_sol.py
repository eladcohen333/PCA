from scipy import stats
import numpy
import math
import itertools
from sklearn.decomposition import PCA
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
from mpl_toolkits.mplot3d import Axes3D
import plotly.express as px
import plotly.io as pio


#prepions and data_normalized

data = pd.read_excel(r"C:\Users\DELL\Desktop\Computational_learning\pca\bc_task_pca.xlsx")
# print(data)
delete = ['id','B.C.']
save_1 = data['id']
save_2 = data['B.C.']
save_delete1 = pd.DataFrame(save_1)
save_delete2 = pd.DataFrame(save_2)
# print(save_delete)
data.drop(delete ,inplace=True, axis=1)
df = pd.DataFrame(data)
# print(data)
new_scalar = stats.zscore(data)
data_normalized = pd.DataFrame(new_scalar)

# data_normalized.to_excel("try_out.xlsx")
# print(data_normalized)

# 1.pca_first time

pca = PCA(n_components=3)
scores = pca.fit_transform(data_normalized)
pca_scores = pd.DataFrame(data=scores,columns=['PC1', 'PC2', 'PC3'])
pca_scores.insert(0,'B.C.',save_delete2)
# print(pca_scores)

# 2. visualization pca on 3D



var = pca.explained_variance_ratio_.sum()
fig = px.scatter_3d(pca_scores, x='PC1', y='PC2', z='PC3', color=pca_scores['B.C.'],title=f'Total Explained Variance: {var}',
labels={'0':'PC1', '1':'PC2', '2':'PC3'})
fig.write_image("pca_1111111","jpg")
# fig.show()

# 3
eigenvalue = pca.explained_variance_
pca_eigenvalue ={"eigenvalue PCA1 ":eigenvalue[0],"eigenvalue PCA2 ":eigenvalue[1],"eigenvalue PCA3 ":eigenvalue[2]}
print("pca_eigenvalue:",pca_eigenvalue)
var_of_pca = pca.explained_variance_ratio_
table_var = pd.DataFrame(var_of_pca)
print("sum of variance :",var)
h = ["pca1","pca2","pca3"]
table_var.insert(0,"explained_variance",h)
print(table_var)







