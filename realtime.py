import pyaudio
import segment
import matplotlib.pyplot as plt
import transform_mfcc as transform
from sklearn.externals import joblib
import sys

tap_recog = joblib.load('forest_recog.bin')

p = pyaudio.PyAudio()
realtime = True
over=False

if realtime:
    stream = p.open(format=p.get_format_from_width(2),
                    channels=1,
                    rate=44100,
                    input=True)
else:
    chunk = segment.wav_to_np('snaps/gss.wav')[:,0]/32768.   
   # segment.play_wav(chunk)

frames = []
CHUNK = 10000
threshold = 0.4
text=[]
try: 
    last_click = None
    while True:
        print 'loop'
        if realtime:
            chunk = stream.read(CHUNK)
            chunk = segment.decode(chunk, 1)[:,0]/32678.
        elif over:
            break
        smoothed = segment.smooth(chunk, 4)
        clicks, last_click = segment.get_clicks(smoothed, threshold, 10000, last_click=last_click)
        if last_click is not None:
            last_click -= CHUNK
        taps = segment.chop(chunk, clicks, afterlength=1300, prelength=50)
        #print len(taps)
        if taps.shape[0]==0:
            print ' ... '
        for tap in taps:
            ft = transform.sndFeature(tap)            
            letter = tap_recog.transform(ft)
            proba = tap_recog.predict_proba(ft)
            text.append(letter)
            print '{} --> {}'.format(proba, letter)
        over = True
except KeyboardInterrupt:
    print "finished recording"
finally: 
    stream.stop_stream()
    stream.close()
    p.terminate()
    text_str = ' '.join(text)
    #print text_str