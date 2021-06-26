from sklearn.preprocessing import StandardScaler
import pickle
import pandas as pd
from sklearn.decomposition import PCA

COMPONENTS = 50

def create_components_names(num_of_components):
    columns = []
    for i in range(0, num_of_components):
        columns.append('component_' + str(i))
    return columns

#Prepare dataset
df = pd.read_pickle('local_features.pkl')
columns_to_drop = ['label']
X = df.drop(columns_to_drop, axis = 1)

#Apply PCA
X = StandardScaler().fit_transform(X)
pca = PCA(n_components = COMPONENTS)
principalComponents = pca.fit_transform(X)

component_names = create_components_names(COMPONENTS)

principalDf = pd.DataFrame(data = principalComponents
             , columns = component_names)
principalDf = principalDf.join(df['label'])
import pdb; pdb.set_trace()
#pickle.dump(pca, open('pca.pkl', 'wb'))
principalDf.to_pickle('pca_data.pkl')
