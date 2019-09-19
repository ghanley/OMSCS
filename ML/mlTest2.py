from sklearn import tree
from sklearn.datasets import load_iris
from sklearn.datasets import load_boston

from sklearn.model_selection import train_test_split
import numpy as np
from util import import_data


data = load_iris()

X_train, X_test, y_train, y_test = train_test_split(data.data,data.target, test_size=0.2, random_state=42)

c = tree.DecisionTreeClassifier()
c.fit(X_train, y_train)
accu_train = np.sum(c.predict(X_train) == y_train) / float(y_train.size)
accu_test = np.sum(c.predict(X_test) == y_test) / float(y_test.size)









from sklearn.externals.six import StringIO
import pydotplus
dot_data = StringIO()
tree.export_graphviz(c, out_file = dot_data, feature_names=data.feature_names,class_names = data.target_names, filled=True, rounded=True, impurity=False)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_pdf("iris.pdf")


print "Classication acc on train", accu_train
print "acc on test", accu_test
