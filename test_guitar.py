import unittest
import numpy as np
import matplotlib.pyplot as plt

from guitar_dsp import Impulse, Delay, NoiseBurst, FirstOrderLowPassFilter, StringSynthesis, SignalSink, FeedbackDelayWithLpf

class TestStringSynthesis (unittest.TestCase):
	
	def test_instantiation(self):
		string = StringSynthesis(frame_size=100, fs=48000, frequency=440., alpha=0.99, tau=0.05, std=1.)
		self.assertTrue(isinstance(string, StringSynthesis))
		
	def test_has_energy(self):
		string = StringSynthesis(frame_size=100, fs=48000, frequency=440., alpha=0.99, tau=0.05, std=1.)
		x = string.process()
		self.assertTrue(np.sum(x*x) > 0.)
		
	def test_is_done(self):
		string = StringSynthesis(frame_size=100, fs=48000, frequency=440., length=0.003)
		x = string.process()
		self.assertFalse(string.is_done())
		x = string.process()
		self.assertTrue(string.is_done())
		
class TestDsp (unittest.TestCase):
	
	def test_impulse(self):
		impulse = Impulse(frame_size=10, fs=48000, length=0.01)
		s = impulse.process()
		
		self.assertEqual(len(s), 10)
		self.assertAlmostEqual(s[0], 1., places=8)
		for sample in s[1:]:
			self.assertAlmostEqual(sample, 0., places=8)
			
	def test_delay(self):
		frame_size = 10
		delay = Delay(frame_size=frame_size, delay=5)
		impulse = Impulse(frame_size=frame_size, fs=48000, length=0.01)
		x = impulse.process()
		y = delay.process(x)
	
		self.assertAlmostEqual(y[5], 1., places=8)
		for sample in y[:5]:
			self.assertAlmostEqual(sample, 0., places=8)		
		for sample in y[6:]:
			self.assertAlmostEqual(sample, 0., places=8)
			
	def test_noise_burst(self):
		noise = NoiseBurst(
			frame_size=100, 
			fs=48000, 
			mean=0., 
			std=1., 
			burst_length=1000, 
			seed=42,
			length=0.1
			)
		x = np.ndarray(0)
		while not noise.is_done():
			x = np.append(x, noise.process())
			
		self.assertAlmostEqual(np.mean(x[:1000]), 0., places=1)		
		self.assertAlmostEqual(np.std(x[:1000]), 1., places=1)		
		for sample in x[1000:]:
			self.assertAlmostEqual(sample, 0., places=8)		
	
	def test_signal_sink(self):
		frame_size = 10
		impulse = Impulse(frame_size=frame_size, fs=48000, length=0.01)
		sink = SignalSink()
		
		while not impulse.is_done():
			x = impulse.process()
			sink.process(x)

		x = sink.get_buffer()
		
		self.assertAlmostEqual(x[0], 1., places=8)
		for sample in x[1:]:
				self.assertAlmostEqual(sample, 0., places=8)		
							
	def test_feedback_delay_periodicity(self):
		frame_size = 10
		impulse = Impulse(frame_size=frame_size, fs=48000, length=0.01)
		fb_delay = FeedbackDelayWithLpf(frame_size=frame_size, delay=4, alpha=0.)
		sink = SignalSink()
		
		while not impulse.is_done():
			x = impulse.process()
			y = fb_delay.process(x)
			sink.process(y)

		s = sink.get_buffer()
											
		for index, sample in enumerate(s):
			if index % 4 == 0:
				self.assertAlmostEqual(sample, 1., places=8)		
			else:
				self.assertAlmostEqual(sample, 0., places=8)		
				
	def test_feedback_delay_lpf(self):
		frame_size = 9
		impulse = Impulse(frame_size=frame_size, fs=48000, length=0.0001875)
		fb_delay = FeedbackDelayWithLpf(frame_size=frame_size, delay=3, alpha=0.01)
		sink = SignalSink()
		
		while not impulse.is_done():
			x = impulse.process()
			y = fb_delay.process(x)
			sink.process(y)

		s = sink.get_buffer()
		
		print(s)
		
		s_expected = [
			                      1.,
			                      0.,
			                      0.,
			            0.99*0.01**0,
			            0.99*0.01**1,
			            0.99*0.01**2,
			0.99*(0.99*0.01**0)+(0.99*0.01**2)*0.01,
			0.99*(0.99*0.01**1)+(0.99*(0.99*0.01**0)+(0.99*0.01**2)*0.01)*0.01,
			0.99*(0.99*0.01**2)+(0.99*(0.99*0.01**1)+(0.99*(0.99*0.01**0)+(0.99*0.01**2)*0.01)*0.01)*0.01,
			]
			
		for received, reference in zip(s, s_expected):
			self.assertAlmostEqual(received, reference, places=8)		
				
	def test_lpf(self):
		frame_size = 10000
		lpf = FirstOrderLowPassFilter(frame_size=frame_size, alpha=0.99)
		noise = NoiseBurst(
			frame_size=frame_size, 
			fs=48000, 
			mean=0.5, 
			std=1., 
			burst_length=frame_size, 
			seed=42,
			length=0.01
			)
		x = noise.process()
		y = lpf.process(x)
		self.assertAlmostEqual(np.mean(x), np.mean(y), places=2)		
		self.assertTrue(np.std(x) > np.std(y))
