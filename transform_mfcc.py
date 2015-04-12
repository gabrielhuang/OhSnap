import numpy
from pylab import *
from scipy.io import wavfile
from features import mfcc, logfbank

sampFreq = 44100

def wavFeature(name, graph = False):
    """
    This function calls sndFeature with data from the given wav file.
    """
    t, snd = wavfile.read(name)
    return sndFeature(snd, graph = graph)

def sndFeature(snd, graph = False):
    #normalize rms here
    snd /= float(np.linalg.norm(snd))
    ft_mfcc = mfcc(snd, samplerate=sampFreq, nfilt=26, numcep=13)[0]
    ft_logf = logfbank(snd, sampFreq)[0]
    #print '{}\n*******\n{}'.format(ft_mfcc, ft_logf)
    #raw_input()
    ft = np.hstack((ft_mfcc, ft_logf))
    #print ft
    #raw_input()
    return ft