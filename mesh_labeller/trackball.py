import pyrender
from pyrender.trackball import Trackball as PyrenderTrackball
import numpy as np
from trimesh import transformations

def sign(x):
	return 1 if x >= 0 else -1

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

	def drag(self, point):
		"""Compute a mouse drag as a rotation about the target, about an axis perpendicular to both
		the depth direction and the direction of the drag."""

		point = np.array(point, dtype=np.float32)
		dx, dy = point - self._pdown

		target = self._target
		x_axis = self._pose[:3, 0].flatten()
		y_axis = self._pose[:3, 1].flatten()
		z_axis = self._pose[:3, 2].flatten()
		mindim = 0.3 * np.min(self._size)

		d = (dx ** 2 + dy ** 2) ** 0.5
		if d > 0:
			angle = d / mindim

			d_3d = dx * x_axis + dy * y_axis
			d_3d = d_3d / np.linalg.norm(d_3d)

			# Compute the axis of rotation
			axis = np.cross(d_3d, z_axis)

			rot_mat = transformations.rotation_matrix(angle, axis, target)
			self._n_pose = rot_mat.dot(self._pose)

	def pan(self, dx, dy):

		mindim = 0.3 * np.min(self._size)
		x_axis = self._pose[:3, 0].flatten()
		y_axis = self._pose[:3, 1].flatten()

		dx = -dx / (5.0 * mindim) * self._scale
		dy = -dy / (5.0 * mindim) * self._scale

		translation = dx * x_axis + dy * y_axis
		t_tf = np.eye(4)
		t_tf[:3, 3] = translation
		self._n_pose = t_tf.dot(self._n_pose)