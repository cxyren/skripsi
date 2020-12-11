from keras.models import load_model
from collections import deque
from sklearn.metrics import confusion_matrix, classification_report, multilabel_confusion_matrix
from tqdm import tqdm
from sklearn.preprocessing import LabelBinarizer
import matplotlib.pyplot as plt
from glob import glob
import pandas as pd
import numpy as np
import math
import pickle
import cv2
import os
import sys

#initialize
num_model = 59
count = len(glob('D:/user/Documents/Skripsi/Output/*')) + 1
model_path = 'D:/user/Documents/Skripsi/Model/'
temp_path = 'D:/user/Documents/Skripsi/Input/Temp/'
model_file = 'modelActivity%02i.h5' % num_model
label_file = 'lb%02i.pickle' % num_model
input_path = 'D:/user/Documents/Skripsi/Input/CSV/'
# input_skeleton = sys.argv[1]
output_path = 'D:/user/Documents/Skripsi/Output/'
output_video = 'Output%02i.avi' % count

size = 128

connecting_joint = [2, 1, 21, 3, 21, 5, 6, 7, 21, 9, 10, 11, 1, 13, 14, 15, 1, 17, 18, 19, 2, 8, 8, 12, 12]

# load the trained model and label from disk
print('[INFO] loading model and label ...')
print(os.path.join(model_path, model_file))
model = load_model(os.path.join(model_path, model_file))
lb = pickle.loads(open(os.path.join(model_path, label_file), "rb").read())

y_true = []

# print('Skeleton name: %s' % input_skeleton)
# load skeleton
print('[INFO] load skeleton ...')
#read skeleton
X = pickle.loads(open(os.path.join('C:/train/', 'new_testx5.pickle'), "rb").read())
y = pickle.loads(open(os.path.join('C:/train/', 'new_testy5.pickle'), "rb").read())

X = np.array(X)
y = np.array(y)

lb2 = LabelBinarizer()
y = lb2.fit_transform(y)

print('[INFO] Model Predicting ...')
preds = model.predict(x=X.astype('float32'))
report = classification_report(y.argmax(axis=1), preds.argmax(axis=1), target_names=lb.classes_, output_dict=True)
print("classification report"),
print(classification_report(y.argmax(axis=1), preds.argmax(axis=1), target_names=lb.classes_))
df = pd.DataFrame(report).transpose()
df.to_csv(os.path.join(output_path, "CLASSIFICATIONNTURGBD4.csv"), index = False)
scores = model.evaluate(X, y, verbose=0)
print("%s: %.2f%%" % (model.metrics_names[0], scores[0]))
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
f = open(os.path.join(output_path, 'reportONNTURGB+D4.txt'), 'w')
f.write("%s: %.2f%%\n" % (model.metrics_names[0], scores[0]))
f.write("%s: %.2f%%\n" % (model.metrics_names[1], scores[1]*100))
f.close()

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
	import itertools
	if normalize:
		cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
		print("Normalized confusion matrix")
	else:
		print('Confusion matrix, without normalization')
	
	print(cm)
	
	plt.imshow(cm, interpolation='nearest', cmap=cmap)
	plt.title(title)
	plt.colorbar()
	tick_marks = np.arange(len(classes))
	plt.xticks(tick_marks, classes, rotation=45)
	plt.yticks(tick_marks, classes)
	
	fmt = '.2f' if normalize else 'd'
	thresh = cm.max() / 2.
	for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
	    plt.text(j, i, format(cm[i, j], fmt),
	             horizontalalignment="center",
	             color="white" if cm[i, j] > thresh else "black")
				 
	plt.ylabel('True label')
	plt.xlabel('Predicted label')
	plt.savefig(os.path.join(output_path, "CONFUSIONMATRIX4.png"), bbox_inches='tight')

result = confusion_matrix(y_true=y.argmax(axis=1), y_pred=preds.argmax(axis=1)).ravel()
result = np.array(result)
result = np.reshape(result, (-1, 5))
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(result, lb.classes_, title='Confusion matrix, without normalization')

multilabel = multilabel_confusion_matrix(y_true=y.argmax(axis=1), y_pred=preds.argmax(axis=1))
f = open(os.path.join(output_path, 'reportONNTURGB+D4.txt'), 'a')
f.write('\n')
for i in range(len(multilabel)):
	f.write('Label: %s\n' %lb.classes_[i])
	f.write('TN: %i\n' % multilabel[i][0][0])
	f.write('FP: %i\n' % multilabel[i][0][1])
	f.write('FN: %i\n' % multilabel[i][1][0])
	f.write('TP: %i\n\n' % multilabel[i][1][1])
f.close()