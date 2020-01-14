import numpy as np
import signal_ops
from guitar_dsp import Chord, SignalSink

fs = 48000
framesize = 512
fingering = [0, 2, 4, 1, 0, 0]
length = 10.
strumdelay = 0.01

eadd9 = Chord(fingering, framesize, fs, length, strumdelay)
sink = SignalSink()

while not eadd9.is_done():
	x = eadd9.process()
	sink.process(x)

s = sink.get_buffer()

filename = 'eadd9.wav'
signal_ops.write_to_disk(filename, fs, s)
signal_ops.plot(s)
