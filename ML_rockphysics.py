import pandas as pd
import numpy as np
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn import clone
from scipy.spatial.distance import correlation
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.tree import export_graphviz
from sklearn.metrics import classification_report
import scipy.stats.mstats as mstats
import warnings

warnings.filterwarnings('ignore')

pd.set_option('display.max_columns', None)  

facies_names = ['SS', 'CSiS', 'FSiS', 'SiSh', 'MS', 'WS', 'D', 'PS', 'BS']
facies_colors = ['#F4D03F', '#F5B041','#DC7633','#6E2C00',
       '#1B4F72','#2E86C1', '#AED6F1', '#A569BD', '#196F3D']
features = ['Depth', 'GR', 'ILD_log10', 'DeltaPHI', 'PHIND', 'NM_M', 'RELPOS']

colnames = ['Facies', 'Depth', 'GR', 'ILD_log10',
            'DeltaPHI', 'PHIND', 'PE','NM_M', 'RELPOS']
training_set = pd.read_csv("training_data.csv", usecols=colnames,
                 skiprows= 0)
test_set = pd.read_csv("test_data.csv", usecols=colnames,
                       skiprows=0)
boxplot = plt.figure()
test_set.boxplot()
#Filling in missing values

X_train = training_set[['Depth','GR', 'ILD_log10', 'DeltaPHI', 'PHIND', 'NM_M', 'RELPOS']][training_set.PE.notnull()]
y_train = training_set['PE'][training_set.PE.notnull()]
X_fit = training_set[['Depth', 'GR', 'ILD_log10', 'DeltaPHI', 'PHIND', 'NM_M', 'RELPOS']][training_set.PE.isnull()]
Clf = RandomForestRegressor(n_estimators=100)
Clf.fit(X_train, y_train)
y_predict = Clf.predict(X_fit)
training_set['PE'][training_set.PE.isnull()] = y_predict


#Creating training and validation sets

X = training_set[['Depth', 'GR', 'ILD_log10', 'DeltaPHI', 'PHIND', 'NM_M', 'RELPOS']]
Y = training_set[['Facies']]
'''
scaler = StandardScaler().fit(X)
scaled_X = scaler.transform(X)
'''
X_train, X_cross, Y_train, Y_cross = train_test_split( X, Y, test_size = 0.1,
                                                     random_state = 1000)
X_test = test_set[['Depth', 'GR', 'ILD_log10', 'DeltaPHI', 'PHIND', 'NM_M', 'RELPOS']]
Y_test = test_set[['Facies']]

test_orig = test_set[['Facies', 'Depth', 'GR', 'ILD_log10', 'DeltaPHI', 'PHIND', 'NM_M', 'RELPOS']]
forest = RandomForestClassifier(n_estimators=100, class_weight="balanced",max_features=0.1, min_samples_leaf=25, min_samples_split=50)

forest.fit(X_train, Y_train)
Y_pred = forest.predict(X_cross)
Y_pred1 = forest.predict(X_test)
X_test['Facies'] = Y_pred1

get_mode = lambda x: mstats.mode(x, axis=None)[0]
X_test['mode'] = pd.rolling_apply(arg=X_test['Facies'], func=get_mode, window=3, center=True, freq=None)
X_test['mode'].fillna(X_test.Facies, inplace=True)
X_test['mode'] = X_test['mode'].astype(float)
pd.options.display.float_format = '{:,.0f}'.format
X_test.replace(X_test['Facies'], X_test['mode'])


def make_facies_log_plot(name, logs, facies_colors):
    #make sure logs are sorted by depth
    logs = logs.sort_values(by='Depth')
    cmap_facies = colors.ListedColormap(
            facies_colors[0:len(facies_colors)], 'indexed')
    
    ztop=logs.Depth.min(); zbot=logs.Depth.max()
    
    cluster=np.repeat(np.expand_dims(logs['Facies'].values,1), 100, 1)
    
    f, ax = plt.subplots(nrows=1, ncols=4, figsize=(8, 12))
    ax[0].plot(logs.GR, logs.Depth, markersize=0.2)
    ax[1].plot(logs.ILD_log10, logs.Depth)
    ax[2].plot(logs.PHIND, logs.Depth, color='r')
    im=ax[3].imshow(cluster, interpolation='none', aspect='auto',
                    cmap=cmap_facies,vmin=1,vmax=9)
    
    divider = make_axes_locatable(ax[3])
    cax = divider.append_axes("right", size="20%", pad=0.05)
    cbar=plt.colorbar(im, cax=cax)
    cbar.set_label((17*' ').join([' SS ', 'CSiS', 'FSiS', 
                                'SiSh', ' MS ', ' WS ', ' D  ', 
                                ' PS ', ' BS ']))
    cbar.set_ticks(range(0,1)); cbar.set_ticklabels('')
    
    for i in range(len(ax)-1):
        ax[i].set_ylim(ztop,zbot)
        ax[i].invert_yaxis()
        ax[i].grid()
        ax[i].locator_params(axis='x', nbins=3)
    
    ax[0].set_xlabel("GR")
    ax[0].set_xlim(logs.GR.min(),logs.GR.max())
    ax[1].set_xlabel("ILD")
    ax[1].set_xlim(logs.ILD_log10.min(),logs.ILD_log10.max())
    ax[2].set_xlabel("PHIND")
    ax[2].set_xlim(logs.PHIND.min(),logs.PHIND.max())
    ax[3].set_xlabel('Facies')
    
    ax[1].set_yticklabels([]); ax[2].set_yticklabels([]); ax[3].set_yticklabels([])
    ax[3].set_xticklabels([])
    f.suptitle(name, fontsize=14,y=0.94)

#Plotting figures and printing reports

#print(classification_report(Y_test, Y_pred1, target_names=facies_names))
print(classification_report(Y_test, Y_pred1, target_names=facies_names))
report = classification_report(Y_test, Y_pred1, target_names=facies_names).split()


confusion_matrix = confusion_matrix(Y_test, Y_pred1, labels=[1,2,3,4,5,6,7,8,9])

rankings = forest.feature_importances_
n = 7
ind = np.arange(n) 
width = 0.35
plt.bar(ind, rankings, width, color='r')
plt.xticks(ind, features)
plt.title("Feature Importances")
make_facies_log_plot('Shankle Predicted', X_test, facies_colors)
plt.savefig('Shankle_Predicted')
make_facies_log_plot('Shankle', test_orig, facies_colors)
plt.savefig('Shankle_actual')


fig, ax = plt.subplots()

ax.matshow(confusion_matrix, cmap=plt.cm.Blues)
ax.set_title('Classification Confusion Matrix')
for i in range(9):
    for j in range(9):
        c = confusion_matrix[j,i]
        ax.text(i, j, str(c), va='center', ha='center')

pd.options.display.mpl_style = 'default'



p = sns.pairplot(test_set[['Facies','Depth', 'GR', 'ILD_log10', 'DeltaPHI', 'PHIND']], hue='Facies')

plt.savefig("pairplot_matrix")




                 
