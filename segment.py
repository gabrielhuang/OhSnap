import pyaudio
import sys
import wave
import numpy as np
import scipy.ndimage
import scipy.signal
import matplotlib.pyplot as plt
from sklearn.externals import joblib

def decode(in_data, channels):
    """
    Convert a byte stream into a 2D numpy array with 
    shape (chunk_size, channels)
    Samples are interleaved, so for a stereo stream with left channel 
    of [L0, L1, L2, ...] and right channel of [R0, R1, R2, ...], the output 
    is ordered as [L0, R0, L1, R1, ...]
    """
    result = np.fromstring(in_data, dtype=np.int16)
    chunk_length = len(result) / channels
    assert chunk_length == int(chunk_length)
    result = np.reshape(result, (chunk_length, channels))
    return result


def encode(signal):
    """
    Convert a 2D numpy array into a byte stream for PyAudio
    Signal should be a numpy array with shape (chunk_size, channels)
    """
    interleaved = signal.flatten()

    # TODO: handle data type as parameter, convert between pyaudio/numpy types
    out_data = interleaved.astype(np.int16).tostring()
    return out_data 
	

def wav_to_np(filename, sample_width=np.int16, chunk_size=1024):
    wf = wave.open(filename, 'rb')
    print 'Open {}, BitsPerSample {}-bits, Channels {}, Rate {}'.format(
        filename, 8*wf.getsampwidth(), wf.getnchannels(), wf.getframerate())
    buf = wf.readframes(chunk_size)
    frames = []
    while buf:
        frames.append(buf)
        buf = wf.readframes(chunk_size)
    buf = b''.join(frames)
    data = np.frombuffer(buf, dtype=sample_width)
    data = data.reshape((-1, wf.getnchannels()))
    return data
    
def smooth(inp, sigma):
    return scipy.ndimage.filters.gaussian_filter(np.abs(inp), sigma=sigma)
    
def get_clicks(inp, threshold, min_samples, last_click=None):
    out = inp*0.
    for i,x in enumerate(inp):
        if x>threshold:
            if last_click is None or i-last_click>min_samples:
                last_click = i
                out[i] = 1
    return out, last_click
    
def chop(inp, clicks, afterlength=150, prelength=0):
    chunks = []
    for i,(x,clicked) in enumerate(zip(inp,clicks)):
        if clicked:
            overflow = i+afterlength-len(inp)
            underflow = i-prelength
            if underflow>=0:
                if overflow>0:
                    chunk = np.hstack((inp[underflow:], np.zeros(overflow)))
                else:
                    chunk = inp[underflow:i+afterlength]
            else:
                if overflow>0:
                    chunk = np.hstack((np.zeros(-underflow),inp[:], np.zeros(overflow)))
                else:
                    print [x.shape for x in (np.zeros(-underflow), inp[:i+afterlength])]                    
                    chunk = np.hstack((np.zeros(-underflow), inp[:i+afterlength]))
                    
            chunks.append(chunk)
    return np.array(chunks)
    
def chop_all(inp, threshold, click_inhibit=6666, afterlength=150, prelength=50):
    smoothed = smooth(inp, 4)
    clicks = get_clicks(smoothed, threshold, click_inhibit)[0]
    chunks = chop(inp, clicks, afterlength, prelength)
    return chunks
    
if __name__=='__main__':
    wav = wav_to_np('snaps/san4.wav')[:,0]/32768. # scale to [-1,+1]
    
    smoothed = smooth(wav, 1)
    threshold = 0.3
    clicks = get_clicks(smoothed, threshold, 6666)[0]
    num_clicks = sum(clicks)
    chunks = chop(wav, clicks, afterlength=300, prelength=0)
    
    #chunks = chop_all(wav, 0.02)
    print 'NumChunks: {}'.format(len(chunks))
    chunks=chunks
    for chunk in chunks:
        plt.plot(chunk)
    
    plt.figure('figure2')
    plt.plot(np.arange(len(smoothed)),smoothed,np.arange(len(smoothed)),np.ones(len(smoothed))*threshold)
    #plt.plot(smoothed)
    plt.plot(np.arange(len(smoothed)), clicks)