import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import random
import matplotlib.pyplot as plt

PCA_COMPONENTS = 20
def get_random_colors(num_of_colors):
    colors = []
    for i in range(0, num_of_colors):
        r = random.random()
        g = random.random()
        b = random.random()
        color = [r, g, b]
        colors.append(color)
    return colors

def plot_components(finalDf):
        #Code for plotting
        fig = plt.figure(figsize = (8,8))
        ax = fig.add_subplot(1,1,1)
        ax.set_xlabel('Principal Component 1', fontsize = 15)
        ax.set_ylabel('Principal Component 2', fontsize = 15)
        ax.set_title('2 component PCA', fontsize = 20)
        targets = finalDf.label.unique()
        colors = get_random_colors(len(targets))

        for target, color in zip(targets,colors):
            indicesToKeep = finalDf['label'] == target
            ax.scatter(finalDf.loc[indicesToKeep, 'pc1']
                   , finalDf.loc[indicesToKeep, 'pc2']
                   , finalDf.loc[indicesToKeep, 'pc3']
                   #, c = color
                   , s = 50)
        ax.legend(targets)
        plt.show()

def pca_component_names(num):
    component_names = []
    for i in range(1, num+1):
        name = 'pc' + str(i)
        component_names.append(name)
    return component_names

def main():
    df = pd.read_pickle('local_features.pkl')
    X = df.drop(columns = ['label'])

    #Standardize the features
    X = StandardScaler().fit_transform(X)

    pca = PCA(n_components = PCA_COMPONENTS)

    principal_components = pca.fit_transform(X)

    component_names = pca_component_names(PCA_COMPONENTS)

    principalDf = pd.DataFrame(data = principal_components, columns = component_names)

    finalDf = pd.concat([principalDf, df[['label']]], axis = 1)
    import pdb; pdb.set_trace()
    finalDf.to_pickle('pca_components_features.pkl')


if __name__ == "__main__":
    main()
