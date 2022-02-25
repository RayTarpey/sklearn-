"# sklearn-" 
"# sklearn-" 
from sklearn import datasets
import numpy as np
iris = datasets.load_iris()
x = iris.data[:, [2,3]]
y = iris.target
print('Class labels:', np.unique(y))
print(x[110:112])

#from sklearn.cross_validation import train_test_split #寫法改為 model_selection
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x, y,test_size=0.3, random_state=0 )

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
sc.fit(x_train)
sc.scale_


x_train_std = sc.transform(x_train)
x_test_std = sc.transform(x_test)

from sklearn.linear_model import Perceptron
#ppn = Perceptron()
#ppn = Perceptron(n_iter=40, eta0=0.1, random_state=0) #寫法改為n_iter_no_change
ppn = Perceptron(n_iter_no_change=40, eta0=0.1, random_state=0)
ppn.fit(x_train_std,y_train)

y_pred = ppn.predict(x_test_std)
print('Misclassified samples: %d' % (y_test != y_pred).sum())
from sklearn.metrics import accuracy_score
print('Accuracy: %.2f' % accuracy_score(y_test, y_pred))

from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
def plot_decision_regions(x, y, classifier, test_idx=None, resolution=0.02):
    markers = ('s', 'x', 'o', '^','v')
    colors = ('red','blue','green','gray','cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])
        # plot the decision surface
    x1_min, x1_max = x[:, 0].min() - 1, x[:, 0].max() + 1
    x2_min, x2_max = x[:, 1].min() - 1, x[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=x[y == cl, 0], y=x[y == cl, 1], alpha=0.8, c=cmap(idx), edgecolor='black', marker=markers[idx], label=cl)

    # highlight test samples
    if test_idx:
        x_test, y_test = x[test_idx, :], y[test_idx]
        plt.scatter(x_test[:, 0],x_test[:, 1],cmap=['gray'],alpha=1.0,edgecolor='black', \
                    linewidths=1,marker='o',s=55, label='test set')
        


x_combined_std = np.vstack((x_train_std,x_test_std))
y_combined = np.hstack((y_train, y_test))

print('x_combined_std.shape:{}'.format(x_combined_std.shape))
print('y_combined.shape:{}'.format(y_combined.shape))

plot_decision_regions(x=x_combined_std,y=y_combined,classifier=ppn,test_idx=range(105,150), resolution=0.02)

plt.xlabel('petal length [standardized]')
plt.ylabel('petal width [standardized]')
plt.legend(loc='upper left')
plt.show()

#logistic regression 線性邏輯斯回歸 (but無關回歸)
import matplotlib.pyplot as plt
import numpy as np

def sigmoid(z): #as an activation function where fi(z) = 1/1+exp^-z
    return 1.0/(1.0+np.exp(-z))

z = np.arange(-7,7,0.1)
phi_z = sigmoid(z)
plt.plot(z,phi_z)
plt.axvline(0.0,color='k') #中線
plt.axhspan(0.0,1.0,facecolor='1.0',alpha=1.0,ls='dotted')
plt.axhline(y=0.5,ls='dotted',color='k')#橫虛線
plt.yticks([0.0,0.5,1.0])
plt.ylim(-0.1,1.1)
plt.xlabel('z')
plt.ylabel('$\phi (z)$')
plt.show()

#logistic regression model training 
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression (C=1000.0,random_state=0) #C is to normalized the quantity of lombda (用L2 normalized做特徵縮放將權重縮小
lr.fit(x_train_std,y_train)
plot_decision_regions(x_combined_std,y_combined,classifier=lr,test_idx=range(105,150))
plt.xlabel('petal length [standardized]')
plt.ylabel('petal width [standardized]')
plt.legend(loc='upper left')
plt.show()

lr.predict_proba(x_test_std [0:2])#predict the testing data sets, where every column indicates the perception of each class



#support vector machine(SVM) 支援向量機
from sklearn.svm import SVC
svm = SVC(kernel='linear', C=1.0, random_state=0)
svm.fit(x_train_std, y_train)
plot_decision_regions(x_combined_std,y_combined, classifier=svm,test_idx=range(105,150))
plt.xlabel('petal length [standardized]')
plt.ylabel('petal width [standardized]')
plt.legend(loc='upper left')
plt.show()

#kernel support vector machine(kernel-SVM) 核支援向量機處理非線性分類
np.random.seed(0)
x_xor = np.random.randn(200,2)
y_xor = np.logical_xor(x_xor[:,0]>0, x_xor[:,1]>0)
print(y_xor[:17])
y_xor = np.where(y_xor,1,-1)
print("{}\n\n{}".format(x_xor[:17],y_xor[:17]))
plt.scatter(x_xor[y_xor==1,0], x_xor[y_xor==1,1], c='b', marker='x', label='1')
plt.scatter(x_xor[y_xor==-1,0], x_xor[y_xor==-1,1], c='r', marker='s', label='1')
plt.xlim(-3.0,3.0)
plt.ylim(-3.0,3.0)
plt.legend()
plt.show()

svm =SVC(kernel='rbf', random_state=0, gamma=0.30, C=10.0)
svm.fit(x_xor,y_xor)
plot_decision_regions(x_xor,y_xor,classifier=svm)
plt.legend(loc='upper left')
plt.show()


#the effection of gamma
svm =SVC(kernel='rbf', random_state=0, gamma=0.20, C=1.0)
svm.fit(x_train_std,y_train)
plot_decision_regions(x_combined_std,y_combined,classifier=svm,test_idx=range(105,150))
plt.legend(loc='upper left')
plt.show()

svm =SVC(kernel='rbf', random_state=0, gamma=100.0, C=1.0)
svm.fit(x_train_std,y_train)
plot_decision_regions(x_combined_std,y_combined,classifier=svm,test_idx=range(105,150))
plt.legend(loc='upper left')
plt.show()

#決策樹(橫軸決策邊界)
from sklearn.tree import DecisionTreeClassifier
tree = DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=0)
tree.fit(x_train,y_train)
x_combined = np.vstack((x_train,x_test))
y_combined = np.hstack((y_train,y_test))
plot_decision_regions(x_combined,y_combined,classifier=tree,test_idx=range(105,150))
plt.xlabel('petal length [standardized]')
plt.ylabel('petal width [standardized]')
plt.legend(loc='upper left')
plt.show()

from sklearn.tree import export_graphviz
import matplotlib.image as mpimg
#from scipy import misc
#from skimage.transform import resize
export_graphviz(tree,out_file='tree.dot',feature_names=['petal length','petal width'])
#dot -Tpng tree.dot -o tree.png #由CMD下指令將決策圖轉成PNG檔
lena = mpimg.imread('tree.png')
#my_image = resize(lena, output_shape=(18, 18)).reshape((1, 18 * 18 * 4)).T
#lena_new_sz = misc.imresize(lena, 0.5) #Scipy版本太高要額外再改
plt.imshow(lena)

