import numpy as np
import random


class Source (object):
	def __init__(self, frame_size, fs, length):
		self._frame_size = frame_size
		self._fs = fs
		self._length = length
		
	def reset(self):
		self._frame_count = 0
		
	def is_done(self):
		if self._frame_count * self._frame_size > self._length * self._fs:
			return True
		else:
			return False
		
	def process(self):
		self._frame_count += 1
		
class SignalSink():
	def __init__(self):
		self._buffer = np.zeros(0)
		
	def process(self, x):
		self._buffer = np.append(self._buffer, x)
		
	def get_buffer(self):
		return self._buffer
			
class Impulse (Source):
	def __init__(self, frame_size, fs, length):
		super().__init__(frame_size, fs, length)
		self.reset()
		
	def reset(self):
		super().reset()
		self._count = 0
		
	def process(self):
		super().process()
		y = np.zeros(self._frame_size)
		if self._count == 0:
			y[0] = 1.
		self._count += self._frame_size
		return y

class NoiseBurst (Source):
	def __init__(self, frame_size, fs, length, mean, std, burst_length, seed=None):
		super().__init__(frame_size, fs, length)
		self._mean = mean
		self._std = std
		self._burst_length = burst_length
		if seed is not None:
			random.seed(seed)
		self.reset()
		
	def reset(self):
		super().reset()
		self._count = 0
		
	def process(self):
		super().process()
		y = np.zeros(self._frame_size)
		index = 0
		while self._count < self._burst_length and index < self._frame_size:
			y[index] = random.gauss(mu=self._mean, sigma=self._std)
			self._count += 1
			index += 1
		return y

class Delay (object):
	def __init__(self, frame_size, delay):
		self._frame_size = frame_size
		self._delay = delay
		self.reset()
		
	def reset(self):
		self._buffer = np.zeros(self._delay)
		
	def process(self, x):
		self._buffer = np.append(self._buffer, x)
		y = self._buffer[0:self._frame_size]
		self._buffer = self._buffer[self._frame_size:]
		return y
		
class FeedbackDelayWithLpf(object):
	def __init__(self, frame_size, delay, alpha):
		self._frame_size = frame_size
		self._delay = delay
		self._alpha = alpha
		self.reset()
		
	def reset(self):
		self._state = 0.
		self._buffer = np.zeros(self._delay)
		
	def process(self, x):
		y = np.zeros(self._frame_size)
		for index in range(self._frame_size):
			y_lpf = (1.-self._alpha)*self._buffer[index] + self._alpha*self._state
			self._state = y_lpf
			y[index] = x[index] + y_lpf
			self._buffer = np.append(self._buffer, y[index])
		self._buffer = self._buffer[self._frame_size:]
		
		return y
		
class FirstOrderLowPassFilter (object):
	def __init__(self, frame_size, alpha):
		self._frame_size = frame_size
		self._alpha = alpha
		self.reset()
		
	def reset(self):
		self._state = 0.
		
	def process(self, x):
		y = np.zeros(self._frame_size)
		for index, sample in enumerate(x):
			y[index] = (1-self._alpha)*sample + self._alpha*self._state
			self._state = y[index]
		return y
		
class StringSynthesis (Source):
	def __init__(self, frame_size, fs, frequency, alpha=0.99, tau=0.05, std=1., length=1., strumdelay=0):
		super().__init__(frame_size, fs, length)
		
		period = int(self._fs/frequency)
		
		self._noise = NoiseBurst(
			frame_size=self._frame_size, 
			fs=self._fs,
			mean=0.,
			std=std,
			burst_length=int(tau*fs),
			length=length
			)
		self._delay = FeedbackDelayWithLpf(self._frame_size, delay=period, alpha=alpha)
		self._strumdelay = Delay(self._frame_size, delay=int(strumdelay*self._fs))		
		self.reset()

	def reset(self):
		super().reset()
		self._state = np.zeros(self._frame_size)
						
	def process(self):
		super().process()
		x = self._noise.process()
		y = self._delay.process(x)
		y_delay = self._strumdelay.process(y)
		return y_delay
		
class Chord (Source):
	def __init__(self, fingering, frame_size, fs, length, strumdelay=0.):
		super().__init__(frame_size, fs, length)
		self.strings = []
		self._strumdelay = strumdelay
		self.init_strings(fingering)
		self.reset()
		
	def reset(self):
		super().reset()
		for string in self.strings:
			string.reset()
			
	def init_strings(self, fingering):
		string_frequencies = [82., 110., 147., 196., 247., 330.]
		strumdelay = 0.
		for semitones, frequency in zip(fingering, string_frequencies):
			pitch = frequency * 2**(semitones/12)
			string = StringSynthesis(
				frame_size=self._frame_size, 
				fs=self._fs, 
				frequency=pitch, 
				alpha=0.4, 
				tau=0.01, 
				std=0.2*0.167,
				length=self._length,
				strumdelay=strumdelay
				)
			strumdelay = strumdelay + self._strumdelay
			self.strings.append(string)
					
	def process(self):
		super().process()
		y = np.zeros(self._frame_size)
		for string in self.strings:
			x = string.process()
			y += x
		return y
			
