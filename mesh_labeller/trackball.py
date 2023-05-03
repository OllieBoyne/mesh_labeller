import pyrender
from pyrender.trackball import Trackball as PyrenderTrackball
import numpy as np

class Trackball(PyrenderTrackball):
	def scroll(self, clicks):
		"""Zoom in to `target` based on clicks"""
		target = self._target
		ratio = 0.90

		mult = 1.0
		if clicks > 0:
			mult = ratio ** clicks
		elif clicks < 0:
			mult = (1.0 / ratio) ** abs(clicks)

		eye = self._n_pose[:3, 3].flatten()
		radius = np.linalg.norm(eye - target)
		scroll_direction = (target - eye) # direction to move in
		translation = (mult * radius - radius) * scroll_direction
		t_tf = np.eye(4)
		t_tf[:3, 3] = translation
		self._n_pose = t_tf.dot(self._n_pose)

		eye = self._pose[:3, 3].flatten()
		radius = np.linalg.norm(eye - target)
		scroll_direction = (target - eye)  # direction to move in
		translation = (mult * radius - radius) * scroll_direction
		t_tf = np.eye(4)
		t_tf[:3, 3] = translation
		self._pose = t_tf.dot(self._pose)