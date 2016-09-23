import scipy
import numpy
import sklearn
import pprint
import pandas

from sklearn.datasets import load_svmlight_file
from sklearn import tree
from sklearn.externals.six import StringIO
import pydot
from IPython.display import Image

input_file = "/Users/luthermartin-pers/Documents/ML Assignment #1/creditcard_default.csv"
#f = open("/Users/luthermartin-pers/Desktop/creditcard_default.txt")
df = pandas.read_csv(input_file)
#f.readline()   skip the header
#data = numpy.loadtxt(f, delimiter=",")

# data = load_svmlight_file("url_svmlight/Day0training.svm", zero_based="False", dtype=numpy.float64)
# test_data = load_svmlight_file("url_svmlight/Day0testing.svm", zero_based="False", dtype=numpy.float64)
dot_data = StringIO()
numpy_array = df.as_matrix()

X = numpy_array[1:,:22]
Y = numpy_array[1:,23]
Xt = numpy_array[0,:22]
# Xy = test_data[1]

clf = tree.DecisionTreeClassifier()
clf = clf.fit(X,Y)
clf_test = clf.predict(Xt)

tree.export_graphviz(clf, out_file=dot_data, filled=True, rounded=True,
                    special_characters=True)
graph = pydot.graph_from_dot_data(dot_data.getvalue())
print(graph)
Image(graph.create_png())