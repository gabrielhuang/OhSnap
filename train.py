import segment
import transform_mfcc as transform
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import train_test_split, KFold
from sklearn.externals import joblib
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA


class TapRecognizer:
    '''Abstract class, transforms raw segmented data into key'''
    def transform(self, data):
        '''Return a Key or None if not recognized'''        
        pass
    
class ForestRecognizer(TapRecognizer):
    def set_forest(self, clf, labelInt2String):
        self.clf = clf
        self.labelInt2String = labelInt2String
    def transform(self, data):
        pred = self.clf.predict(data)[0]
        letter = self.labelInt2String[pred]
        return letter

class ForestPcaRecognizer(TapRecognizer):
    def set(self, clf, labelInt2String, pca):
        self.clf = clf
        self.pca = pca
        self.labelInt2String = labelInt2String
    def transform(self, data):
        data = self.pca.transform(data)
        pred = self.clf.predict(data)[0]
        letter = self.labelInt2String[pred]
        return letter
    def predict_proba(self, data):
        data = self.pca.transform(data)
        p = self.clf.predict_proba(data)
        return p
        
def files_to_data(files, file_labels, threshold):
    uniqueLabels = np.unique(file_labels)
    labelString2Int = {s:i for i,s in enumerate(uniqueLabels)}
    labels=[]
    data=[]
    for file,file_label in zip(files,file_labels):
        wav = segment.wav_to_np(file)[:,0]/32768. # left CH
        chunks = segment.chop_all(wav, threshold, afterlength=700, prelength=0)
        chunks = map(transform.sndFeature, chunks)
        data += list(chunks)
        labels += [labelString2Int[file_label],]*len(chunks)
        print 'Chunked {} examples for {}'.format(len(chunks),file)
    X = np.array(data)
    y = np.array(labels)
    return X,y,uniqueLabels

if __name__=='__main__':
    threshold = 0.3
    #actions = ['train']
    actions = ['prepare','train']
    # Load the data
    if 'prepare' in actions:
        X,y,uniqueLabels = files_to_data([
               'snaps/gab1.wav',
               'snaps/gab2.wav',
               'snaps/gab3.wav',
               'snaps/gab4.wav',
               'snaps/gab5.wav',
               'snaps/gab6.wav',
               'snaps/gab7.wav',
               'snaps/san1.wav',
               'snaps/san2.wav',
               'snaps/san3.wav',
               'snaps/san4.wav',
               'snaps/suc1.wav',
               'snaps/suc2.wav',
               'snaps/suc3.wav',
               'snaps/suc4.wav',
               'snaps/suc5.wav',
               'snaps/suc6.wav',
               'snaps/suc7.wav'],
               ['gab','gab','gab','gab','gab','gab','gab',
               'san','san','san','san',
               'suc','suc','suc','suc','suc','suc','suc'],
                threshold)
        uniqueLabels = ['gab','san','suc']
        plt.subplot(121)
        plt.imshow(X)
        joblib.dump((X,y), 'trainTest.bin')
    
    if 'train' in actions:
        n_folds = 5
        
        (X,y) = joblib.load('trainTest.bin')
        sum_accuracy,sum2_accuracy = 0.,0.
        for train_idx,test_idx in KFold(len(y), n_folds=n_folds, shuffle=True):
            pca = PCA()
            X_train,y_train = X[train_idx],y[train_idx]
            X_test,y_test = X[test_idx],y[test_idx]
            X_train = pca.fit_transform(X_train)
            X_test = pca.transform(X_test)
            clf = RandomForestClassifier(n_estimators=500)
            #clf = LinearSVC(C=0.05)
            clf = clf.fit(X_train, y_train)
            
            y_pred = clf.predict(X_test)    
            accuracy = np.mean(y_pred==y_test)
            sum_accuracy += accuracy
            sum2_accuracy += accuracy**2
        accuracy = sum_accuracy / float(n_folds)
        sum2_accuracy /= float(n_folds)
        std_dev = np.sqrt(sum2_accuracy-accuracy**2)
        cm = confusion_matrix(y_test, y_pred)
        cm2 = cm.astype(np.float)/cm.sum(axis=1).reshape((-1,1))
        ax = plt.subplot(122)    
        ax.matshow(cm2)
        ax.set_xticklabels([''] + uniqueLabels)
        ax.set_yticklabels([''] + uniqueLabels)
        print 'accuracy = {} +/- {}'.format(accuracy, std_dev)
        print 'confusion matrix = \n{}'.format(cm2)
        # create object
        tap_recog = ForestPcaRecognizer()
        tap_recog.set(clf, uniqueLabels, pca)
        tap_recog_file = 'forest_recog.bin'
        joblib.dump(tap_recog, tap_recog_file)
        print 'Saved model to {}'.format(tap_recog_file)