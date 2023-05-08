import os

import pyrender
import numpy as np
import trimesh
from mesh_labeller.geometry import DrawableMesh, IcoSphere
from mesh_labeller.texture import Paint
from mesh_labeller.trackball import Trackball

from pyrender.constants import TextAlign
from tkinter import Tk, filedialog as filedialog
import imageio

import pyglet
from pyglet import clock
is_ctrl = lambda symbol: symbol == pyglet.window.key.LCTRL or symbol == pyglet.window.key.RCTRL
is_command = lambda symbol: symbol == pyglet.window.key.LCOMMAND or symbol == pyglet.window.key.RCOMMAND

# Save texture as 'label_tex.png' in same folder as obj
def obj_loc_to_tex_loc(obj_loc):
	return os.path.join(os.path.dirname(obj_loc), 'label_tex.png')

class Viewer(pyrender.Viewer):
	mouse_mode = 'cursor'

	CIRCLE_RADIUS_DEFAULT = 0.01
	CIRCLE_RADIUS_STEP = 0.001  # change in radius per scroll tick

	CIRCLE_DEFAULT_ALPHA = 0.5
	CIRCLE_PDOWN_ALPHA = 0.1

	CAM_TRANSLATE_PAD = 0.25  # fraction of viewport size where the camera will pan if the mouse is in that region
	CAM_TRANSLATE_SPEED = 15.0  # Speed of sideways pan

	SCALE_OBJ = 0.3  # scale object so largest extent is this size

	def __init__(self, cfg, viewport_size=500, *args, **kwargs):

		# Set cursor to crosshair
		self._view_hwnd = None
		self.set_mouse_cursor(self.get_system_mouse_cursor(self.CURSOR_CROSSHAIR))

		self._scene = scene = pyrender.Scene()

		camera = pyrender.camera.PerspectiveCamera(yfov=1.0, aspectRatio=1.0, znear=0.001)
		pose = np.eye(4)
		pose[2, 3] = 0.5
		scene.add(camera, pose=pose)

		# Load classes
		self.cfg = cfg
		self.labeller = Paint(cfg['CLASSES'])
		self.labeller.cycle_up()

		self.mesh, self.mesh_node = None, None

		loc = None
		if cfg['DEFAULT_FILE'] is not None:
			loc = os.path.join(cfg['DEFAULT_MESH_LOC'], cfg['DEFAULT_FILE'])
			if not os.path.isfile(loc):
				print(f'Default file {loc} does not exist, defaulting to file dialog...')
				loc = None

		self.load_mesh(loc)

		self.cursor_radius = self.CIRCLE_RADIUS_DEFAULT

		self.circle = IcoSphere(radius=self.cursor_radius)
		self.circle.set_alpha(self.CIRCLE_DEFAULT_ALPHA)
		self.circle_node = scene.add(self.circle.mesh)

		self.cam_node = scene.main_camera_node
		self.cam = self.cam_node.camera

		self.H, self.W = viewport_size

		# schedule autosave
		self.autosave_interval = cfg['SETTINGS']['AUTOSAVE']
		if self.autosave_interval > 0:
			pyglet.clock.get_default().schedule_interval(self.save_texture, self.autosave_interval)

		super().__init__(scene, *args, **kwargs, run_in_thread=False,
						 use_raymond_lighting=True,
						 auto_start=False, viewport_size=viewport_size)

	def load_mesh(self, loc=None):

		if self.mesh_node is not None:
			self.scene.remove_node(self.mesh_node)

		self.obj_loc = loc
		if self.obj_loc is None:
			self.obj_loc = self.open_file(default_loc = self.cfg['DEFAULT_MESH_LOC'],
										  filetypes=[('Wavefront OBJ', '*.obj')], descr='Select mesh file')

		# try to load texture from same directory
		tex_loc = obj_loc_to_tex_loc(self.obj_loc)
		if not os.path.isfile(tex_loc):
			tex_loc = None

		mesh = trimesh.load(self.obj_loc)
		mesh = mesh.apply_scale(self.SCALE_OBJ / mesh.bounding_box.extents.max()) # resize to match bounds

		mat = pyrender.material.MetallicRoughnessMaterial()
		mesh = DrawableMesh.from_trimesh(mesh, material=mat, base_colour=self.labeller[self.cfg['DEFAULT_CLASS']].rgb,
										 tex_loc=tex_loc, dilation=self.cfg['SETTINGS']['DILATION'],
										 tex_size = self.cfg['SETTINGS']['TEXTURE_SIZE'])
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
		if self.mouse_mode == 'mouse-pan':
			self.side_pan()

		self.update_captions()
		super().on_draw()

	def on_mouse_press(self, x, y, buttons, modifiers):

		self._trackball.down(np.array([x, y]))
		self.viewer_flags['mouse_pressed'] = True 		# Stop animating while using the mouse

		# if middle mouse button, pan
		if buttons == pyglet.window.mouse.MIDDLE:
			self.mouse_mode = 'scroll-pan'

		# if right mouse button, camera rotate
		elif buttons == pyglet.window.mouse.RIGHT:
			self.mouse_mode = 'camera-rotate'

		elif self.mouse_mode == 'cursor' and buttons == pyglet.window.mouse.LEFT:
			self.mesh.draw_from_sphere(self.circle_node.translation, self.cursor_radius,
									   status=self.labeller.rgb)
			self.circle.set_alpha(self.CIRCLE_PDOWN_ALPHA)

	def on_mouse_release(self, x, y, button, modifiers):

		self.mouse_mode = 'cursor' # always set mouse mode back to cursor
		if button == pyglet.window.mouse.LEFT:
			self.circle.set_alpha(self.CIRCLE_DEFAULT_ALPHA)
		else:
			super().on_mouse_release(x, y, button, modifiers)

	def on_mouse_drag(self, x, y, dx, dy, buttons, modifiers):

		# if middle mouse button, pan
		if self.mouse_mode == 'scroll-pan':
			self._trackball.pan(dx, dy)

		# if middle mouse button, rotate
		if self.mouse_mode == 'camera-rotate':
			self._trackball.drag(np.array([x, y]))

		if self.mouse_mode == 'cursor' and buttons == pyglet.window.mouse.LEFT:
			self.mesh.draw_from_sphere(self.circle_node.translation, self.cursor_radius,
									   status=self.labeller.rgb)
			self.on_mouse_motion(x, y, dx, dy)

	def on_mouse_motion(self, x, y, dx, dy):
		"""Try to project a ray from camera to mouse on mesh.
		If collides, move cursor to collision point."""
		proj_3d = self.project_to_mesh(x, y)
		if proj_3d is not None:
			self._trackball._n_target = proj_3d
			self._trackball._target = proj_3d
			self.circle_node.translation = proj_3d

	def on_mouse_scroll(self, x, y, dx, dy):

		if self.mouse_mode == 'cursor':
			self.cursor_radius += dy * self.CIRCLE_RADIUS_STEP
			self.cursor_radius = np.clip(self.cursor_radius, 0.002, 0.1)
			s = self.cursor_radius / self.CIRCLE_RADIUS_DEFAULT
			self.circle_node.scale = (s, s, s)

		elif self.mouse_mode == 'camera':
			super().on_mouse_scroll(x, y, dx, dy)

	def on_key_press(self, symbol, modifiers):
		if is_ctrl(symbol) or is_command(symbol):
			self.mouse_mode = 'camera'

		elif symbol == pyglet.window.key.P:
			self.mouse_mode = 'mouse-pan'

		elif symbol == pyglet.window.key.C:
			self.reset_camera()

		elif symbol == pyglet.window.key.S:
			self.save_texture()

		elif symbol == pyglet.window.key.O:
			self.load_mesh()

		elif symbol == pyglet.window.key.COMMA:
			self.labeller.cycle_down()

		elif symbol == pyglet.window.key.PERIOD:
			self.labeller.cycle_up()

		elif symbol == pyglet.window.key.Z:
			self.mesh.texture.undo()  # undo last draw action

		# R starts recording frames
		elif symbol == pyglet.window.key.R:
			if self.viewer_flags['record']:
				self.save_gif()
				self.set_caption(self.viewer_flags['window_title'])
			else:
				self.set_caption(
					'{} (RECORDING)'.format(self.viewer_flags['window_title'])
				)
			self.viewer_flags['record'] = not self.viewer_flags['record']

		# if a number 0 - 9 pressed, cycle to that class
		elif symbol in range(48, 58):
			ID = symbol - 48
			self.labeller.set_class(ID)

	def on_key_release(self, symbol, modifiers):
		self.mouse_mode = 'cursor' # revert back to cursor scroll mode

	def on_resize(self, width, height):
		self.H = height
		self.W = width
		if self.cam is not None:
			self.scene.main_camera_node.camera.aspectRatio = (width/height)

		super().on_resize(width, height)

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
		ar = self.W / self.H
		K = np.eye(4)
		K[0, 0] = f / ar
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
		captions.append(dict(location=TextAlign.BOTTOM_LEFT, text=f"[{self.mouse_mode.title()}]",
							 font_name='OpenSans-Regular', font_pt=30, color=(0, 0, 0), scale=1.0))


		captions.append(dict(location=TextAlign.BOTTOM_RIGHT, text="[{L.ID}] {L.name}".format(L=self.labeller),
							 font_name='OpenSans-Regular', font_pt=30, color=self.labeller.rgb, scale=1.0))

		self.viewer_flags['caption'] = captions


	def save_gif(self, filename=None):
		if filename is None:
			filename = self._get_save_filename(['gif', 'all'])

		if not filename.endswith('.gif'):
			filename += '.gif'

		if filename is not None:
			self.viewer_flags['save_directory'] = os.path.dirname(filename)
			imageio.mimwrite(filename, self._saved_frames, extension='.gif',
							 duration=10)
		self._saved_frames = []

	def save_texture(self, dt=None):
		"""Save texture as 'label_tex.png' in same folder as obj"""
		loc = obj_loc_to_tex_loc(self.obj_loc)
		self.mesh.texture.save(loc)

		if dt is None:
			print(f"Saved texture to {loc}")

	def reset_camera(self):
		self._trackball._n_pose = self._default_camera_pose