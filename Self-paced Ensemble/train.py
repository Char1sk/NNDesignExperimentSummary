import time
RANDOM_STATE = 42

from self_paced_ensemble import SelfPacedEnsembleClassifier
from self_paced_ensemble.self_paced_ensemble.base import sort_dict_by_key
from self_paced_ensemble.utils._plot import plot_2Dprojection_and_cardinality

from collections import Counter
import matplotlib.pyplot as plt

from sklearn.decomposition import KernelPCA
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score

X, y = make_classification(n_classes=2, class_sep=1, # 3-class
    weights=[0.1, 0.9], n_informative=3, n_redundant=1, flip_y=0,
    n_features=20, n_clusters_per_class=1, n_samples=2000, random_state=0)

time_pre=time.perf_counter()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

clf = SelfPacedEnsembleClassifier(
    n_estimators=5,
    random_state=RANDOM_STATE,
).fit(X_train, y_train)

# Predict & Evaluate
y_pred = clf.predict(X_test)
score = average_precision_score(y_test, y_pred)
time_aft=time.perf_counter()
print ("SelfPacedEnsemble {} | AUPRC: {:.3f} | #Training Samples: {:d} | time:{:.3f}".format(
    len(clf.estimators_), score, sum(clf.estimators_n_training_samples_),time_aft-time_pre
    ))


import torch 
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
time_pre=time.perf_counter()
x_data=torch.Tensor(X_train)
y_data=torch.Tensor(y_train).view(1000,1)

class Model(nn.Module):
	def __init__(self,input_size,h1,output_size):
		super().__init__()
		self.linear1=nn.Linear(input_size,h1)
		self.linear2=nn.Linear(h1,output_size)
	def forward(self,x):
		pred1=torch.sigmoid(self.linear1(x))
		pred2=torch.sigmoid(self.linear2(pred1))
		return pred2

model=Model(20,10,1)


criterion=nn.BCELoss()
optimizer=torch.optim.Adam(model.parameters(),lr=0.01)

epochs=1000
for i in range(epochs):
    y_pred=model.forward(x_data)
    loss=criterion(y_pred,y_data)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# plt.plot(losses)
# plt.show()

y_pred=model.forward(torch.Tensor(X_test))
score = average_precision_score(y_test, y_pred.detach().numpy())
time_aft=time.perf_counter()
print ("DNN | AUPRC: {:.3f} | #Epoch: {:d} | time:{:.3f}".format(
     score, epochs, time_aft-time_pre
    ))



# Visualize the dataset
# projection = KernelPCA(n_components=2).fit(X, y)
# fig = plot_2Dprojection_and_cardinality(X, y, projection=projection)
# plt.show()
# plt.figure()
# plt.show()

# print(X_test.shape) 