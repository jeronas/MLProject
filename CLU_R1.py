import numpy as np
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn import metrics
from sklearn import preprocessing
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import preprocessing
from sklearn import mixture
import seaborn as sns
import scipy.cluster.hierarchy as sch
from sklearn.utils import shuffle     
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.cluster import KMeans
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from IPython.display import display

from sklearn import tree


#Why do we need to standardize data?
#Data standardization is the process of rescaling the attributes so that they have mean as 0 and variance as 1. The ultimate goal to perform standardization is to bring down all the features to a common scale without distorting the differences in the range of the values

def k_means_clustering (data,maxclusters):
    inertia_all=[]
    silhouete_all=[]
    print("K means Clustering")
    for i in range(2,maxclusters):
        kmeans=KMeans(n_clusters=i)
        kmeans.fit(data)
        y_kmeans=kmeans.predict(data)
        silhouete_vals=silhouette_samples(data,y_kmeans)
        inertia_all.append(kmeans.inertia_)
        silhouete_all.append(np.mean(silhouete_vals))
        print(f'Number of Clusters = {i}, Silhouette = {round(np.mean(silhouete_vals),4)} Inertia = {round(kmeans.inertia_,2)}')
    # plot
    plt.figure(0)
    plt.title('K Means Silhouette')
    plt.plot(range(2, maxclusters), silhouete_all, 'r*-')
    plt.ylabel('Silhouette Score')
    plt.xlabel('Number of Clusters')
    plt.show()

    plt.figure(1)
    plt.title('K Means Inertia')
    plt.plot(range(2, maxclusters), inertia_all, 'g*-')
    plt.ylabel('Inertia Score')
    plt.xlabel('Number of Clusters')
    plt.show()
    return inertia_all,silhouete_all

def gaussianMixture (data,maxClusters):
    silhouettesAll=[]
    print("Gaussian Mixture")
    for i in range(2,maxClusters):
        gmm = mixture.GaussianMixture(n_components=i, covariance_type='full').fit(data)
        labels=gmm.predict(data)
        silhouetteScore=silhouette_score(data, labels)
      #  print 'mixtures, clusters', n, gmm.bic(x) 
        print (f'Number of Clusters = {i}, Silhouette={(silhouettesAll,4)}')
        silhouettesAll.append(round(silhouetteScore,3))
      #  print 'mixtures, clusters', n, gmm.bic(x)       
    plt.figure(5)
    plt.title('Gaussian Mixture Silhouette')
    plt.xlabel('Number of clusters')
    plt.ylabel('Silhouette Values')
    plt.plot(range(2,maxClusters), silhouettesAll,  'r*-')
    plt.show() 
    print(" ")
    return silhouettesAll
   

def Hclustering (data,maxClusters):
    silhouete_all=[]
    print("Hierarchical Clustering")
    for i in range(2,maxClusters):
        H_clust=AgglomerativeClustering(n_clusters=i,affinity='euclidean', linkage='ward', compute_full_tree=True)
        H_clust.fit(data)
        silhouette = silhouette_score(data, H_clust.labels_)
        print (f'Number of Clusters = {i}, Silhouette={round(silhouette,4)}')
        silhouete_all.append(round(silhouette,3))
    plt.figure(4)
    plt.title('Hierarchical Clustering Silhouette')
    plt.xlabel('Number of clusters')
    plt.ylabel('Silhouette Values')
    plt.plot(range(2,maxClusters), silhouete_all, '*-')
    plt.show()
    print(" ")
    return silhouete_all    


df = pd.read_csv('CUSTOMER.csv')

print(" ")
print(df.info())
#Descriptive statistics below shows on average clients spend the most on fresh groceries and the least on delicassen.
print(df.describe())

import matplotlib.pyplot as plt
import seaborn as sns 
colmns = ['Channel','Region']
for col in colmns:
    sns.set_style("whitegrid")
    plt.figure(figsize = (8,5))
    sns.countplot(x=df[col], data=df) 
    plt.show()
    
sns.set_style('whitegrid')
sns.countplot(x="Channel",hue='Region',data=df)



headers = df.iloc[0]
new_data  = pd.DataFrame(df.values[0:], columns=headers)
new_data
#print(headers)
new_data = new_data.replace('?',np.NaN)
new_data = new_data.replace(np.nan,0)
new_data
#print(new_data)
dr = new_data.drop(new_data.columns[0], axis=1)
dr
#print(dr)
x = df.drop([0,])
x
#print(x)
data = df.replace('?',np.NaN)
data = df.replace(np.nan,0)
#print(data)

x=df.iloc[1:,2:8]
x

x1 = x.astype(float)
print(x1)




maxClusters=15


#X=preprocessing.scale(X)|
R=preprocessing.scale(x1)

pca = PCA(n_components=5,random_state=1)
X=pca.fit_transform(R)




kInertia, kSilhouette = k_means_clustering (X, maxClusters)





# Covert data to numpy arrays
X=np.array(df)


print(f'PCA explained variance ratio (first {5} components): %s'
       % str(pca.explained_variance_ratio_))
print(" ")
print(f"Total percentage of components explaining the data {round(sum(pca.explained_variance_ratio_),2)*100} %")
print(" ")




PC_values = np.arange(pca.n_components_) + 1
plt.plot(PC_values, pca.explained_variance_ratio_, 'o-', linewidth=2, color='blue')
plt.title('Scree Plot')
plt.xlabel('Principal Component')
plt.ylabel('Variance Explained')
plt.show()

#cluster with two clusters
kmeans = KMeans(n_clusters=2)
kmeans.fit(X)
kmeans.cluster_centers_


# Print cluster size
print('\nSize of Cluster-1=', len(kmeans.labels_[kmeans.labels_==0]))
print('Size of Cluster-2=', len(kmeans.labels_[kmeans.labels_==1]))


# 1st approch to characterization/ analysis of clusters
#  print cluster centres
print('\nFeatures names ',x.columns)
print('\nCluster Centre 1',kmeans.cluster_centers_[0])
print('\nCluster Centre 2',kmeans.cluster_centers_[1])



# 2nd approch to characterization/ analysis of clusters
#  Box plot for each feature for each cluster
idxCluster1=np.where(kmeans.labels_==0)
idxCluster2=np.where(kmeans.labels_==1)

#1st cluster
plt.figure(1)
plt.xticks(rotation=90)
plt.boxplot(X[idxCluster1],labels=list(df.columns))
plt.title("Cluster 1")


#2nd cluster
plt.figure(2)
plt.xticks(rotation=90)
plt.boxplot(X[idxCluster2],labels=list(df.columns))
plt.title("Cluster 2")








# 3rd approch to characterization/ analysis of clusters
#  train a DT for each cluster
perc=0.3

X_train, X_test, Y_train, Y_test = train_test_split(X, kmeans.labels_, test_size=perc)


clfDT =  tree.DecisionTreeClassifier()
#clfDT =  tree.DecisionTreeClassifier( max_depth=None, min_samples_leaf=20)
#clfDT= tree.DecisionTreeClassifier( max_depth=5)

clfDT.fit(X_train, Y_train)

print('Tree rules=\n',tree.export_text(clfDT, feature_names=list(df.columns)))


#test the trained model on the training set
Y_train_pred_DT=clfDT.predict(X_train)
#test the trained model on the test set
Y_test_pred_DT=clfDT.predict(X_test)


confMatrixTrainDT=confusion_matrix(Y_train, Y_train_pred_DT, labels=None)
confMatrixTestDT=confusion_matrix(Y_test, Y_test_pred_DT, labels=None)


print ('train: Conf matrix Decision Tree')
print (confMatrixTrainDT)
print ()

print ('test: Conf matrix Decision Tree')
print (confMatrixTestDT)
print ()


pr_y_test_pred_DT=clfDT.predict_proba(X_test)

#ROC curve for the class encoded by 0, survived
fprDT, tprDT, thresholdsDT = roc_curve(Y_test, pr_y_test_pred_DT[:,1])


print(classification_report(Y_test, Y_test_pred_DT))


# Run The PCA
pca = PCA(n_components=3)
pca.fit(df)
f=pca.fit(df)


k_means_optimum = KMeans(n_clusters = 2, init = 'k-means++',  random_state=42)
y = k_means_optimum.fit_predict(X)

# Store results of PCA in a data frame
result=pd.DataFrame(pca.transform(df), columns=['PCA%i' % i for i in range(3)], index=df.index)
 
# Plot initialisation
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(result['PCA0'], result['PCA1'], result['PCA2'], c='green', cmap="Set2_r", s=60)
 
# make simple, bare axis lines through space:
xAxisLine = ((min(result['PCA0']), max(result['PCA0'])), (0, 0), (0,0))
ax.plot(xAxisLine[0], xAxisLine[1], xAxisLine[2], 'r')
yAxisLine = ((0, 0), (min(result['PCA1']), max(result['PCA1'])), (0,0))
ax.plot(yAxisLine[0], yAxisLine[1], yAxisLine[2], 'r')
zAxisLine = ((0, 0), (0,0), (min(result['PCA2']), max(result['PCA2'])))
ax.plot(zAxisLine[0], zAxisLine[1], zAxisLine[2], 'r')
 
# label the axes
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.set_zlabel("PC3")
ax.set_title("PCA on CUSTOMER data")
plt.show()



hSilhouete=Hclustering (X,maxClusters)



# Affinity = {“euclidean”, “l1”, “l2”, “manhattan”,
# “cosine”}
# Linkage = {“ward”, “complete”, “average”}
Hclustering = AgglomerativeClustering( n_clusters=2, 
                affinity='euclidean', linkage='ward', compute_full_tree=True)



#X = StandardScaler().fit_transform(X)
print ('Linkage,  Distance type')
print (Hclustering.linkage, Hclustering.affinity)
print ()



Hclustering.fit(X)
print ('Cluster labels=',Hclustering.labels_)


# Print cluster size
print('\nSize of Cluster-1=', len(Hclustering.labels_[kmeans.labels_==0]))
print('Size of Cluster-2=', len(Hclustering.labels_[kmeans.labels_==1]))



# 3rd approch to characterization/ analysis of clusters
#  train a DT for each cluster
perc=0.3

X_train, X_test, Y_train, Y_test = train_test_split(X, Hclustering.labels_, test_size=perc)


clfDT =  tree.DecisionTreeClassifier()
#clfDT =  tree.DecisionTreeClassifier( max_depth=None, min_samples_leaf=20)
#clfDT= tree.DecisionTreeClassifier( max_depth=5)

clfDT.fit(X_train, Y_train)

print('Tree rules=\n',tree.export_text(clfDT, feature_names=list(df.columns)))


#test the trained model on the training set
Y_train_pred_DT=clfDT.predict(X_train)
#test the trained model on the test set
Y_test_pred_DT=clfDT.predict(X_test)


confMatrixTrainDT=confusion_matrix(Y_train, Y_train_pred_DT, labels=None)
confMatrixTestDT=confusion_matrix(Y_test, Y_test_pred_DT, labels=None)


print ('train: Conf matrix Decision Tree')
print (confMatrixTrainDT)
print ()

print ('test: Conf matrix Decision Tree')
print (confMatrixTestDT)
print ()


pr_y_test_pred_DT=clfDT.predict_proba(X_test)

#ROC curve for the class encoded by 0, survived
fprDT, tprDT, thresholdsDT = roc_curve(Y_test, pr_y_test_pred_DT[:,1])


print(classification_report(Y_test, Y_test_pred_DT))





gSilhouette=gaussianMixture (X,maxClusters)
gmm = mixture.GaussianMixture(n_components=2, covariance_type='full').fit(X)

#cluster with two clusters
kmeans = KMeans(n_clusters=2)
kmeans.fit(X)
kmeans.cluster_centers_

# 3rd approch to characterization/ analysis of clusters
#  train a DT for each cluster
perc=0.3

X_train, X_test, Y_train, Y_test = train_test_split(X, kmeans.labels_, test_size=perc)


clfDT =  tree.DecisionTreeClassifier()
#clfDT =  tree.DecisionTreeClassifier( max_depth=None, min_samples_leaf=20)
#clfDT= tree.DecisionTreeClassifier( max_depth=5)

clfDT.fit(X_train, Y_train)

print('Tree rules=\n',tree.export_text(clfDT, feature_names=list(df.columns)))


#test the trained model on the training set
Y_train_pred_DT=clfDT.predict(X_train)
#test the trained model on the test set
Y_test_pred_DT=clfDT.predict(X_test)


confMatrixTrainDT=confusion_matrix(Y_train, Y_train_pred_DT, labels=None)
confMatrixTestDT=confusion_matrix(Y_test, Y_test_pred_DT, labels=None)


print ('train: Conf matrix Decision Tree')
print (confMatrixTrainDT)
print ()

print ('test: Conf matrix Decision Tree')
print (confMatrixTestDT)
print ()


pr_y_test_pred_DT=clfDT.predict_proba(X_test)

#ROC curve for the class encoded by 0, survived
fprDT, tprDT, thresholdsDT = roc_curve(Y_test, pr_y_test_pred_DT[:,1])


print(classification_report(Y_test, Y_test_pred_DT))

 

# Print cluster size
print('\nSize of Cluster-1=', len(kmeans.labels_[kmeans.labels_==0]))
print('Size of Cluster-2=', len(kmeans.labels_[kmeans.labels_==1]))


# 1st approch to characterization/ analysis of clusters
#  print cluster centres
print('\nFeatures names ',x.columns)
print('\nCluster Centre 1',kmeans.cluster_centers_[0])
print('\nCluster Centre 2',kmeans.cluster_centers_[1])


