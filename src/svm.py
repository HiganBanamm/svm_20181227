from sklearn import svm

X = [[2,0],[1,1],[2,3]]
y = [0,0,1] #分别代表这三个数据的类别
clf = svm.SVC(kernel='linear') #使用SVC来创建分类器
clf.fit(X,y)

print(clf)

print(clf.support_vectors_) #输出支持向量
print(clf.support_)  #输出支持向量的索引
print(clf.n_support_)  #每个类中分别找到了几个支持向量
                    #[1,1]表示第一个类别中找到了1个，第二个类别中找到了1个
                    
print(clf.predict([[2,0]])) #测试[2,0]这个点属于哪一类
