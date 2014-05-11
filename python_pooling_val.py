"""DecMeg2014 example code.

Simple prediction of the class labels of the test set by:
- pooling all the triaining trials of all subjects in one dataset.
- Extracting the MEG data in the first 500ms from when the
  stimulus starts.
- Using a linear classifier (logistic regression).
"""
from __future__ import division
import numpy as np
from sklearn.linear_model import LogisticRegression
from scipy.io import loadmat
from sklearn.decomposition import TruncatedSVD,PCA
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier

def create_features(XX, tmin, tmax, sfreq, tmin_original=-0.5):
    """Creation of the feature space:
    - restricting the time window of MEG data to [tmin, tmax]sec.
    - Concatenating the 306 timeseries of each trial in one long
      vector.
    - Normalizing each feature independently (z-scoring).
    """
    print "Applying the desired time window."
    beginning = np.round((tmin - tmin_original) * sfreq).astype(np.int)
    end = np.round((tmax - tmin_original) * sfreq).astype(np.int)
    XX = XX[:, :, beginning:end].copy()

    print "2D Reshaping: concatenating all 306 timeseries."
    XX = XX.reshape(XX.shape[0], XX.shape[1] * XX.shape[2])

    print "Features Normalization."
    XX -= XX.mean(0)
    XX = np.nan_to_num(XX / XX.std(0))

    return XX


if __name__ == '__main__':

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
    for subject in subjects_train:
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

        XX = create_features(XX, tmin, tmax, sfreq)

        X_train.append(XX)
        y_train.append(yy)

    X_train = np.vstack(X_train)
    y_train = np.concatenate(y_train)
    print "Trainset:", X_train.shape

    print
    print "Creating the validation set."
    subjects_val = range(9, 17)
    for subject in subjects_val:
        filename = 'data/train_subject%02d.mat' % subject
        print "Loading", filename
        data = loadmat(filename, squeeze_me=True)
        XX = data['X']
        yy = data['y']
        sfreq = data['sfreq']
        tmin_original = data['tmin']
        print "Dataset summary:"
        print "XX:", XX.shape
        print "sfreq:", sfreq

        XX = create_features(XX, tmin, tmax, sfreq)

        Xval.append(XX)
        yval.append(yy)

    Xval = np.vstack(Xval)
    yval = np.hstack(yval)
    print "Validation set:", Xval.shape
    
#    X_all = list(X_train)
#    X_all.append(Xval)
#    X_all = np.vstack(X_all)
#
#    print "PCA decomposition"
#    pca = PCA(n_components=300);#algorithm= 'randomized',n_iterations=10000,random_state = seed);
#    pca.fit(X_all)
#    X_train = pca.transform(X_train)
#    Xval = pca.transform(Xval)
    
    print
#    clf = SVC(C=1, kernel="linear",random_state=0) # Beware! You need 10Gb RAM to train LogisticRegression on all 16 subjects!
#   clf = GradientBoostingClassifier(loss='deviance',learning_rate=0.1, n_estimators=10, subsample=1.0, min_samples_split=2, min_samples_leaf=1, max_depth=3, init=None, random_state=None, max_features=None, verbose=0)
    clf = LogisticRegression(C=2,random_state=0)  
    
    print "Classifier:"
    print clf
    print "Training."
    clf.fit(X_train, y_train)
    print "Predicting."
    y_pred = clf.predict(Xval)
    
    acc = sum(y_pred==yval)/len(yval)
    print "Accuracy: %f" % acc

