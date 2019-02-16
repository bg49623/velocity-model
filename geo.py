import numpy as np
import matplotlib.pyplot as plt
import obspy.io.segy.core as segy
from obspy.io.segy.core import _read_segy
#np.set_printoptions(threshold=np.nan)
import pandas as pd
from sklearn.svm import SVR

def read_model(fname):
    data = _read_segy(fname)
    return np.array([tr.data for tr in data.traces])

###########DATA MANIPULATION############
x = read_model("MODEL_P-WAVE_VELOCITY_1.25m.segy")

subsections = []
print(x[1][1])

xslice = x[:-1 , :-1]

print(xslice.shape)
shapes = np.split(xslice, 8)
print(shapes[1].shape)
for i in range(0, 8):
    subsections.append(np.array(shapes[i].reshape(-1)))

print(shapes[1].shape)
arr = np.zeros(shape = (8, 4760000))
for i in range(0, 8):
    arr[i] = subsections[i]


df = pd.DataFrame(data = arr)
df = df.T
X = np.array([0, 1, 2, 3]).reshape(-1,1)
res1 = np.zeros(shape = (47600, 1))

######PREDICTIVE MODELING#########
for i in range (0,47600):
    Y = np.array(df.loc[[i], [0,1,2,3]]).flatten('F')
    svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)
    y_rbf = svr_rbf.fit(X, Y).predict(X)
    res1[i] = (y_rbf[1] + y_rbf[2] + y_rbf[3] + y_rbf[0])/2

res1 = res1.reshape(170, 280)

'''X1 = np.array([4, 5, 6, 7]).reshape(-1,1)
res2 = np.zeros(shape = (4760000, 1))

for i in range(0,500):
    Y = np.array(df.loc[[i], [4, 5, 6, 7]]).flatten('F')
    svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)
    y_rbf = svr_rbf.fit(X1, Y).predict(X1)
    res2[i] = (y_rbf[1] + y_rbf[2]) / 2'''
#plt.plot(X,Y)


#print(subsections[0].reshape(1700, 2800).shape)



#PLOTTING

#plt.scatter(X,Y, color = 'orange')
#plt.plot(X,y_rbf, color = 'purple', lw = 2, label = 'Prediction')
#print(shapes[1].reshape(-1)[10000])
plt.imshow(res1, cmap = "jet")
plt.show()
