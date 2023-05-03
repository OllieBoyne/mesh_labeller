import os

import pyrender
import numpy as np
import trimesh
from mesh_labeller.geometry import DrawableMesh, IcoSphere
from mesh_labeller.texture import Paint
from mesh_labeller.trackball import Trackball

from pyrender.constants import TextAlign
from tkinter import Tk, filedialog as filedialog

import pyglet
is_ctrl = lambda symbol: symbol == pyglet.window.key.LCTRL or symbol == pyglet.window.key.RCTRL
is_command = lambda symbol: symbol == pyglet.window.key.LCOMMAND or symbol == pyglet.window.key.RCOMMAND


class Viewer(pyrender.Viewer):
	scroll_mode = 'cursor'

	CIRCLE_RADIUS_DEFAULT = 0.01
	CIRCLE_RADIUS_STEP = 0.001  # change in radius per scroll tick

	CIRCLE_DEFAULT_ALPHA = 0.5
	CIRCLE_PDOWN_ALPHA = 0.1

	CAM_TRANSLATE_PAD = 0.25  # fraction of viewport size where the camera will pan if the mouse is in that region
	CAM_TRANSLATE_SPEED = 15.0  # Speed of sideways pan

	def __init__(self, cfg, viewport_size=500, *args, **kwargs):

		self._scene = scene = pyrender.Scene()

		camera = pyrender.camera.PerspectiveCamera(yfov=1.0, aspectRatio=1.0, znear=0.001)
		pose = np.eye(4)
		pose[2, 3] = 0.5
		scene.add(camera, pose=pose)

		# Load classes
		self.cfg = cfg
		self.labeller = Paint(cfg['classes'])

		self.mesh, self.mesh_node = None, None
		self.load_mesh()

		self.cursor_radius = self.CIRCLE_RADIUS_DEFAULT

		self.circle = IcoSphere(radius=self.cursor_radius)
		self.circle.set_alpha(self.CIRCLE_DEFAULT_ALPHA)
		self.circle_node = scene.add(self.circle.mesh)

		self.cam_node = scene.main_camera_node
		self.cam = self.cam_node.camera

		self.H, self.W = viewport_size


		super().__init__(scene, *args, **kwargs, run_in_thread=False,
						 use_raymond_lighting=True,
						 auto_start=False, viewport_size=viewport_size)

	def load_mesh(self, loc=None):

		if self.mesh_node is not None:
			self.scene.remove_node(self.mesh_node)

		self.obj_loc = loc
		if self.obj_loc is None:
			self.obj_loc = self.open_file(default_loc = self.cfg['default_mesh_loc'],
										  filetypes=[('Wavefront OBJ', '*.obj')], descr='Select mesh file')

		# load obj
		mesh = trimesh.load(self.obj_loc)
		mat = pyrender.material.MetallicRoughnessMaterial()
		mesh = DrawableMesh.from_trimesh(mesh, material=mat, base_colour= self.labeller[self.cfg['default_class']].rgb)
		self.mesh = mesh
		self.mesh_node = self.scene.add(mesh)

	def open_file(self, default_loc=os.getcwd(), filetypes=None, descr='Select file'):

		return filedialog.askopenfilename(
			initialdir=default_loc, title=descr,
			filetypes=filetypes
		)

	def _reset_view(self):
		"""Override with custom trackball"""
		scale = self.scene.scale
		if scale == 0.0:
			scale = 2.0
		centroid = self.scene.centroid

		self._trackball = Trackball(
			self._default_camera_pose, self.viewport_size, scale, centroid
		)

	def on_draw(self):
		if self.scroll_mode == 'pan':
			self.side_pan()

		self.update_captions()
		super().on_draw()

	def on_mouse_press(self, x, y, buttons, modifiers):

		if self.scroll_mode == 'cursor' and buttons == pyglet.window.mouse.LEFT:
			self.mesh.draw_from_sphere(self.circle_node.translation, self.cursor_radius,
									   status=self.labeller.rgb)
			self.circle.set_alpha(self.CIRCLE_PDOWN_ALPHA)
		else:
			super().on_mouse_press(x, y, buttons, modifiers)

	def on_mouse_release(self, x, y, button, modifiers):
		if button == pyglet.window.mouse.LEFT:
			self.circle.set_alpha(self.CIRCLE_DEFAULT_ALPHA)
		else:
			super().on_mouse_release(x, y, button, modifiers)

	def on_mouse_drag(self, x, y, dx, dy, buttons, modifiers):
		if self.scroll_mode == 'cursor' and buttons == pyglet.window.mouse.LEFT:
			self.mesh.draw_from_sphere(self.circle_node.translation, self.cursor_radius,
									   status=self.labeller.rgb)
			self.on_mouse_motion(x, y, dx, dy)
		else:
			super().on_mouse_drag(x, y, dx, dy, buttons, modifiers)

	def on_mouse_motion(self, x, y, dx, dy):
		"""Try to project a ray from camera to mouse on mesh.
		If collides, move cursor to collision point."""
		proj_3d = self.project_to_mesh(x, y)
		if proj_3d is not None:
			self._trackball._n_target = proj_3d
			self._trackball._target = proj_3d
			self.circle_node.translation = proj_3d

	def on_mouse_scroll(self, x, y, dx, dy):

		if self.scroll_mode == 'cursor':
			self.cursor_radius += dy * self.CIRCLE_RADIUS_STEP
			self.cursor_radius = np.clip(self.cursor_radius, 0.002, 0.1)
			s = self.cursor_radius / self.CIRCLE_RADIUS_DEFAULT
			self.circle_node.scale = (s, s, s)

		elif self.scroll_mode == 'camera':
			super().on_mouse_scroll(x, y, dx, dy)

	def on_key_press(self, symbol, modifiers):
		if is_ctrl(symbol) or is_command(symbol):
			self.scroll_mode = 'camera'

		if symbol == pyglet.window.key.P:
			self.scroll_mode = 'pan'

		if symbol == pyglet.window.key.S:
			# Save texture as 'label_tex.png' in same folder as obj
			loc = os.path.join(os.path.dirname(self.obj_loc), 'label_tex.png')
			self.mesh.texture.save(loc)
			print(f"Saved texture to {loc}")

		if symbol == pyglet.window.key.O:
			self.load_mesh()

		if symbol == pyglet.window.key.COMMA:
			self.labeller.cycle_down()

		if symbol == pyglet.window.key.PERIOD:
			self.labeller.cycle_up()

		if symbol == pyglet.window.key.Z:
			self.mesh.texture.undo() # undo last draw action

	def on_key_release(self, symbol, modifiers):
		self.scroll_mode = 'cursor' # revert back to cursor scroll mode

	@property
	def cam_pose(self):
		pose = self._trackball.pose.copy()
		R = pose[:3, :3]
		t = - R @ pose[:3, 3]
		out = np.eye(4)
		out[:3, :3] = R
		out[:3, 3] = t
		return out

	@property
	def cam_NDC_matrix(self):
		f = 1 / (2 * np.tan(self.cam.yfov / 2.0))
		K = np.eye(4)
		K[0, 0] = f
		K[1, 1] = f

		return K

	@property
	def camera_normal(self):
		"""Get normal vector of camera"""
		return self.get_ray(self.W//2, self.H//2)

	@property
	def NDC_transform(self):
		"""Get matrix from world point -> NDC = K[R|t]"""
		return self.cam_NDC_matrix @ self.cam_pose

	def get_ray(self, x, y):
		"""Get ray origin and direction for a given pixel coordinate"""

		R = self._trackball.pose[:3, :3]  # camera rotation

		ray_origin = self._trackball.pose[:3, 3]  # camera centre
		ndc = np.array([(x/self.W - 0.5) , (y/self.H - 0.5), -1, 0])  # normalized device coordinates

		ray_dir = (np.linalg.inv(self.cam_NDC_matrix) @ ndc)[:3]  # ray direction in world coords
		ray_dir = R @ ray_dir  # rotate to camera coords
		ray_dir /= np.linalg.norm(ray_dir)  # normalise

		return ray_origin, ray_dir

	def project_to_mesh(self, x, y):
		"""Find the intersection of the mouse-pointer ray with the mesh"""
		ray_origin, ray_dir = self.get_ray(x, y)

		intersect_loc = self.mesh.intersects(ray_origin, ray_dir)['locations']

		if intersect_loc.size == 0:
			return None

		else:
			return intersect_loc[0]

	def side_pan(self):
		"""If mouse is near edge of screen, pan camera in that direction"""

		x, y = self._mouse_x, self._mouse_y
		x_pan_val = min(x, (self.W - x)) / self.W
		y_pan_val = min(y, (self.H - y)) / self.H

		x_pan, y_pan = 0, 0
		if x_pan_val < self.CAM_TRANSLATE_PAD:
			x_pan = [1, -1][x>self.W/2] * x_pan_val * self.CAM_TRANSLATE_SPEED

		if y_pan_val < self.CAM_TRANSLATE_PAD:
			y_pan = [1, -1][y>self.H/2] * y_pan_val * self.CAM_TRANSLATE_SPEED

		self._trackball.pan(x_pan, y_pan)

	def update_captions(self):
		"""Set caption to show current scroll mode, as well as current class"""
		captions = []
		captions.append(dict(location=TextAlign.BOTTOM_LEFT, text=f"[{self.scroll_mode.title()}]",
							 font_name='OpenSans-Regular', font_pt=30, color=(0, 0, 0), scale=1.0))


		captions.append(dict(location=TextAlign.BOTTOM_RIGHT, text="[{L.ID}] {L.name}".format(L=self.labeller),
							 font_name='OpenSans-Regular', font_pt=30, color=self.labeller.rgb, scale=1.0))

		self.viewer_flags['caption'] = captions
