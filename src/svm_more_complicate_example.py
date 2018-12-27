import numpy as np #numpy是一个python里针对科学计算，支持矩阵运算的数据包，里面有很多方程
import pylab as pl #python 画图的一个包
from sklearn import svm

clf = svm.SVC(kernel='linear')

np.random.seed(0) #随机的取值，填入参数0，表示我们每次运行的时候，不要让这些随机值变化，就是每次都是这些随机值
X = np.r_[np.random.randn(20,2)-[2,2],np.random.randn(20,2)+[2,2]]#产生训练实例的点,成正太分布，减号表示在左边，加号表示在右边
Y = [0]*20 + [1]*20 #Y表示类别，前20个点属于0这个类，后20个点属于1这个类

clf.fit(X,Y)

w = clf.coef_[0] #获取w向量的值
a = -w[0]/w[1]
x = np.linspace(-5, 5) #从-5到5产生连续的值，例如：-5,-4，-3，-2，1，2,3，4
y = a*x-(clf.intercept_[0]/w[1]) 

b=clf.support_vectors_[0] 
y_down = a*x + (b[1]-a*b[0])

b = clf.support_vectors_[-1]
y_up = a*x + (b[1]-a*b[0])
print(x)
print(clf.n_support_) 
print(clf.predict([[2,-5]])) #预测[2,-5]这个点属于哪一类

#画图
pl.plot(x,y,'k-')
pl.plot(x,y_down,'k--')
pl.plot(x,y_up,'k--')

pl.scatter(clf.support_vectors_[:,0],clf.support_vectors_[:,1],s=80, facecolors='none') #把支持向量区分开
pl.scatter(X[:,0],X[:,1],c=Y) #X[:,0]就是取所有行的第0个数据, X[:,1] 就是取所有行的第1个数据
pl.axis('tight')
pl.show()
