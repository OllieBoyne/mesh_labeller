import pyrender
import numpy as np
import trimesh
from mesh_labeller.geometry import DrawableMesh, IcoSphere
from mesh_labeller.texture import Paint

import pyglet
is_ctrl = lambda symbol: pyglet.window.key.LCTRL or symbol == pyglet.window.key.RCTRL
is_command = lambda symbol: pyglet.window.key.LCOMMAND or symbol == pyglet.window.key.RCOMMAND


class Viewer(pyrender.Viewer):
	scroll_mode = 'cursor'
	CIRCLE_RADIUS_DEFAULT = 0.01
	CIRCLE_DEFAULT_ALPHA = 0.5
	CIRCLE_PDOWN_ALPHA = 0.1

	def __init__(self, cfg, viewport_size=500, *args, **kwargs):

		scene = pyrender.Scene()

		camera = pyrender.camera.PerspectiveCamera(yfov=1.0, aspectRatio=1.0, znear=0.001)
		pose = np.eye(4)
		pose[2, 3] = 0.5
		scene.add(camera, pose=pose)

		# Load classes
		self.paint = Paint(cfg['classes'])
		default_colour = self.paint[cfg['default_class']].rgb

		# load obj
		mesh = trimesh.load('test/mesh_w_uv.obj')
		mat = pyrender.material.MetallicRoughnessMaterial()
		mesh = DrawableMesh.from_trimesh(mesh, material=mat, base_colour=default_colour)
		self.mesh = mesh
		self.mesh_node = scene.add(mesh)

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


	def on_mouse_press(self, x, y, buttons, modifiers):

		if self.scroll_mode == 'cursor' and buttons == pyglet.window.mouse.LEFT:
			self.mesh.draw_from_sphere(self.circle_node.translation, self.cursor_radius, status=(255, 0, 0))
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
			self.mesh.draw_from_sphere(self.circle_node.translation, self.cursor_radius, status=(255, 0, 0))
			self.on_mouse_motion(x, y, dx, dy)
		else:
			super().on_mouse_drag(x, y, dx, dy, buttons, modifiers)

	def on_mouse_motion(self, x, y, dx, dy):
		proj_3d = self.project_to_mesh(x, y)
		if proj_3d is not None:
			self.circle_node.translation = proj_3d

	def on_mouse_scroll(self, x, y, dx, dy):

		if self.scroll_mode == 'cursor':
			self.cursor_radius += dy * 0.01
			self.cursor_radius = np.clip(self.cursor_radius, 0.002, 0.1)
			s = self.cursor_radius / self.CIRCLE_RADIUS_DEFAULT
			self.circle_node.scale = (s, s, s)

		elif self.scroll_mode == 'camera':
			super().on_mouse_scroll(x, y, dx, dy)

	def on_key_press(self, symbol, modifiers):
		if is_ctrl(symbol) or is_command(symbol):
			self.scroll_mode = 'camera'

		if symbol == pyglet.window.key.S:
			self.mesh.texture.save('tex.png')
			print("Saved texture to tex.png")

	def on_key_release(self, symbol, modifiers):
		if is_ctrl(symbol) or is_command(symbol):
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


