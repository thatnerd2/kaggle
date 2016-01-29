from sklearn import svm
import numpy as np;
import pandas as pd;
import csv;

data = [];
ys = [];

with open('train.csv', 'rb') as f:
	lines = csv.reader(f.read().splitlines());
	data = [row for row in lines][1::];
	xs = [row[3::] for row in data][0:10000];
	ys = [row[2] for row in data][0:10000];

def isNumeric (s):
	try:
		float(s)
		return True
	except ValueError:
		return False

xs = map(lambda x: map(lambda y: "0" if not isNumeric(y) else y, x), xs);
print "Step 1 done";
#xs = map(lambda x: filter(isNumeric, x), xs);
xs = map(lambda x: map(float, x), xs);
print "Step 2 done"
ys = map(float, ys);

print "Step 3 done"

#X = [[0, 0, 99, 0, 0, 0], [1, 1, 3, 1, 1, 1]]
#y = [0, 1]
clf = svm.SVC()
clf.fit(xs, ys) 
for i in range(0, 50, 2):
	print "pred(xs[",i,"]):",clf.predict(xs[i]);
	print "\tsupposed to be",ys[i];

