from Tkinter import *
import numpy as np

import pyaudio
import segment
from train import ForestPcaRecognizer
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import transform_mfcc as transform
from sklearn.externals import joblib
import sys

last_click = None
values = np.array([1,0.5,0.3])

tap_recog = joblib.load('forest_recog.bin')

p = pyaudio.PyAudio()
    
stream = p.open(format=p.get_format_from_width(2),
                channels=1,
                rate=44100,
                input=True)

master = Tk()
canvas = Canvas(master, width=1024, height=768)
canvas.pack()

CHUNK = 10000
base = 250
base_size = 64
shrinkage = 0.8
threshold = 0.4
power = 2.5
def loop():
    global last_click
    global values
    print 'loop'
    chunk = stream.read(CHUNK)
    chunk = segment.decode(chunk, 1)[:,0]/32678.
    smoothed = segment.smooth(chunk, 4)
    clicks, last_click = segment.get_clicks(smoothed, threshold, 10000, last_click=last_click)
    if last_click is not None:
        last_click -= CHUNK
    taps = segment.chop(chunk, clicks, afterlength=1300, prelength=50)
    if taps.shape[0]==0:
        print ' ... '
    for tap in taps:
        ft = transform.sndFeature(tap)            
        letter = tap_recog.transform(ft)
        proba = tap_recog.predict_proba(ft)[0]
        i = np.argmax(proba)
        dvalues = np.zeros(3)
        dvalues[i]=1
        values += power * dvalues
        print '{} --> {}'.format(proba, letter)
    values *= shrinkage
    
    canvas.delete("all")
    nvalues = (base_size*values).astype(np.int32)
    canvas.create_text(base*1, 500, text='Gab', font=("Arial",nvalues[0]), fill='blue')
    canvas.create_text(base*2, 200, text='San', font=("Arial",nvalues[1]), fill='green')
    canvas.create_text(base*3, 500, text='Suc', font=("Arial",nvalues[2]), fill='red')
    master.after(10, loop)  


fps = 30
master.after(1000//fps, loop)
mainloop()