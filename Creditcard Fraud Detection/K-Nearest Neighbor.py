import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pylab import rcParams
from sklearn.model_selection import train_test_split


sns.set(style='whitegrid', palette='muted', font_scale=1.5)
rcParams['figure.figsize'] = 14, 8
RANDOM_SEED = 42
LABELS = ["Normal", "Fraud"]

df= pd.read_csv('sample_data.csv')

df.isnull().values.any() #checking for missing values
frauds = df[df.Class == 1]
normal = df[df.Class == 0]
frauds.shape


from sklearn.preprocessing import StandardScaler
df = df.drop(['Time'], axis=1)
df['Amount'] = StandardScaler().fit_transform(df['Amount'].values.reshape(-1, 1))

X = df.iloc[:,:29].values
#Y=df["Class"].tolist()
#Y=np.array(Y)
Y = df.iloc[:,29].values


X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.4)


#### KNN#####

# Fitting K-NN to the Training set
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 2, metric = 'minkowski', p = 2)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)


plt.figure(figsize=(5, 5))
sns_plot=sns.heatmap(cm, xticklabels=LABELS, yticklabels=LABELS, annot=True, fmt="d");
plt.title("KNN train:test=40:60")
plt.xlabel('Predicted class')
plt.ylabel('True class')
plt.show()

from sklearn.metrics import precision_recall_fscore_support
precision_recall_fscore_support(y_test, y_pred, average='binary')
