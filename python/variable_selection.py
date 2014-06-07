

from __future__ import division
import numpy as np
from sklearn.linear_model import LogisticRegression
from scipy.io import loadmat
from sklearn.decomposition import TruncatedSVD,PCA
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
import matplotlib 
from sklearn.lda import LDA
from pylab import *




print "DecMeg2014: https://www.kaggle.com/c/decoding-the-human-brain"
print
subjects_train = range(1, 8) # use range(1, 17) for all subjects
print "Training on subjects", subjects_train 

# We throw away all the MEG data outside the first 0.5sec from when
# the visual stimulus start:
tmin = 0.0
tmax = 0.500
print "Restricting MEG data to the interval [%s, %s]sec." % (tmin, tmax)

X_train = []
y_train = []
Xval = []
yval = []

print
print "Creating the trainset."

subject = 1

filename = 'data/train_subject%02d.mat' % subject
print "Loading", filename
data = loadmat(filename, squeeze_me=True)
XX = data['X']
yy = data['y']
sfreq = data['sfreq']
tmin_original = data['tmin']
print "Dataset summary:"
print "XX:", XX.shape
print "yy:", yy.shape
print "sfreq:", sfreq

beginning = np.round((tmin - tmin_original) * sfreq).astype(np.int)
end = np.round((tmax - tmin_original) * sfreq).astype(np.int)
XX = XX[:, :, beginning:end].copy()

trials = XX.shape[0]
sensors = XX.shape[1]
N = XX.shape[2]

XX = XX.reshape(trials, sensors*N)

print "Features Normalization."
XX -= XX.mean(0)
XX = np.nan_to_num(XX / XX.std(0))

XX = XX.reshape(trials, sensors,N)

trials = range(1,XX.shape[0])
trials = range(1,10)
sensor = 1
#
faces = XX[yy==1,sensor,:]
nofaces = XX[yy==0,sensor,:]

for t in trials:
    data = XX[t,sensor,:]
    
    if yy[t]==1: 
        plt.plot(data, 'b')
    else:
        plt.plot(data,'r')
        
    plt.hold(True)

plt.show()

#
#ts = 1/sfreq
#ps = np.abs(np.fft.fft(data))**2
#
#freqs = np.fft.fftfreq(data.size, ts)
#idx = np.argsort(freqs)
#
#plot(freqs[idx], ps[idx])
