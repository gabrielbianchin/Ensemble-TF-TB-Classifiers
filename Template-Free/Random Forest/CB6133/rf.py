import numpy as np
from sklearn.ensemble import RandomForestClassifier
import joblib
from sklearn.model_selection import GridSearchCV

base = np.load('/content/drive/My Drive/Mestrado - Experimentos/cullpdb+profile_6133.npy')
base = np.reshape(base, (-1, 700, 57))
a = np.arange(0,21)
b = np.arange(35,56)
c = np.hstack((a,b))

features = base[:, :, c]
classes = base[:, :, 22:30]

def timeseries_to_classification(sequence):
    datas = []
    for i in range(len(sequence)):
        for j in range(42):
            datas.append(sequence[i, j])
    return np.array(datas)

def dados(window, start, end, leng=8):
    
    data = []
    label = []
    
    for i in range(start, end):
        protein = features[i]
        ss = classes[i]
        
        x = protein[~np.all(protein == 0, axis=1)]
        y = ss[~np.all(ss == 0, axis=1)]
        
        padding = np.zeros(window * 42).reshape(window, 42)
        x = np.vstack((padding, x))
        x = np.vstack((x, padding))
        
        padding = np.zeros(window * leng).reshape(window, leng)
        y = np.vstack((padding, y))
        y = np.vstack((y, padding))
        
        cont = (window * 2) + 1
        
        for i in range(x.shape[0] - (window * 2)):
            data.append(timeseries_to_classification(x[i:cont]))
            label.append(np.argmax(y[i+window:i+window+1]))
            cont += 1
    
    data = np.array(data)
    label = np.array(label)
    
    return data, label

for i in range(4, 9):

	clf = RandomForestClassifier(n_estimators=500, max_depth=15, verbose=2)
	  
	X_train, y_train = dados(i, 0, 5600)
	X_val, y_val = dados(i, 5877, 6133)
	X_test, y_test = dados(i, 5605, 5877)
	  
	clf.fit(X_train, y_train)

	pred_val = clf.predict_proba(X_val)
	pred_test = clf.predict_proba(X_test)

	np.save('/content/pred-cb6133-RF-' + str(i) + '.npy', pred_test)
	np.save('/content/pred-cb6133-RF-' + str(i) + '-val.npy', pred_val)