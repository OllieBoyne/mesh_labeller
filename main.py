from trimesh import viewer
from trimesh.ray.ray_pyembree import RayMeshIntersector
import trimesh
import pyrender
import numpy as np
import pyglet
from rasterize_tris import rasterize_tris
import cv2

viewport_size = (500, 500)

is_ctrl = lambda symbol: pyglet.window.key.LCTRL or symbol == pyglet.window.key.RCTRL
is_command = lambda symbol: pyglet.window.key.LCOMMAND or symbol == pyglet.window.key.RCOMMAND

class IcoSphere:
	def __init__(self, radius=0.05):
		self.trimesh = trimesh.creation.icosphere(radius=radius)
		self.trimesh.visual.vertex_colors = [0.0, 0.0, 1.0, 1.0]
		self.mesh = pyrender.Mesh.from_trimesh(self.trimesh)

	def set_alpha(self, alpha=1.):
		self.mesh.primitives[0].material.baseColorFactor[3] = alpha



class DrawableMesh(pyrender.Mesh):
	def __init__(self, primitives, is_visible=True, intersector: RayMeshIntersector=None):
		self.intersector = intersector
		super().__init__(primitives, is_visible=is_visible)

	def intersects(self, ray_origins, ray_directions):
		"""Given a set of rays, return the indices of the faces, and the locations that intersect with the rays.
		:param ray_origins: (N, 3) array of ray origins
		:param ray_directions: (N, 3) array of ray directions
		:return: dict of
			f_idxs: (N, ) array of face indices
			r_idxs: (N, ) array of ray indices
			locations: (N, 3) array of locations
		"""
		if ray_origins.ndim == 1: ray_origins = ray_origins[None, :]
		if ray_directions.ndim == 1: ray_directions = ray_directions[None, :]

		if self.intersector is None:
			raise ValueError("No intersector provided")

		index_tri, index_ray, locations = self.intersector.intersects_id(ray_origins, ray_directions,
																		 multiple_hits=False,
																		 return_locations=True)
		return {'f_idxs': index_tri, 'locations': locations, 'r_idxs': index_ray}

	def draw_triangles(self, triangles, colour=(1,0,0), fmt='UV'):
		"""Draw a series of triangles to the UV map.
		fmt='UV': triangles are in UV coordinates (0-1)
		fmt='pixel': triangles are in pixel coordinates (0-width, 0-height)"""

		source = self.primitives[0].material.baseColorTexture.source.copy()
		size, *_ = source.shape

		if fmt == 'UV':
			triangles = np.clip((triangles * size).astype(np.float32), 0, size-1)
			triangles[..., 1] = (size-1) - triangles[..., 1]  # +ve v -> -ve y

		# Rasterize triangles to mask
		# Seams in UV map will appear as gaps in the mask, so dilate the mask to fill them in
		mask = rasterize_tris(triangles, size, method='per-triangle', dilation=1)

		source[mask] = colour

		self.tex = pyrender.Texture(name='tex', width=size, height=size, source=source, source_channels='RGB')
		self.primitives[0].material = pyrender.material.MetallicRoughnessMaterial(baseColorTexture=self.tex)


	def draw(self, pts, colour=(1, 0, 0), fmt='UV'):
		"""For all points in (N x 2) pts, draw colour onto the UV map at those points.
		fmt='UV': pts are in UV coordinates (0-1)
		fmt='pixel': pts are in pixel coordinates (0-width, 0-height)"""

		source = self.primitives[0].material.baseColorTexture.source.copy()
		size, *_ = source.shape


		if fmt == 'UV':
			pts = np.clip((pts * size).astype(int), 0, size-1)
			pts[..., 1] = (size-1) - pts[..., 1]

		source[pts[..., 1], pts[..., 0]] = colour

		self.tex = pyrender.Texture(name='tex', width=size, height=size, source=source,
								 source_channels='RGB')
		self.primitives[0].material = pyrender.material.MetallicRoughnessMaterial(baseColorTexture=self.tex)


	def draw_from_sphere(self, sphere_centre, sphere_radius, status=(1, 0, 0)):
		"""Given a sphere centre and radius, collect all vertices within the sphere and
		colour them `status` in the UV texture"""

		# Gather all faces whose centres are within the sphere
		# project these to UV
		# rasterize

		faces = self.primitives[0].indices.astype(int)
		triangles_3d = self.primitives[0].positions[faces]
		face_centres = triangles_3d.mean(axis=1)
		dists = np.linalg.norm(face_centres - sphere_centre, axis=-1)
		faces_mask = dists <= sphere_radius

		# Project to UV map
		verts_2d = self.primitives[0].texcoord_0[faces[faces_mask]]

		self.draw_triangles(verts_2d, colour=status, fmt='UV')

	@staticmethod
	def from_trimesh(mesh, *args, **kwargs):
		m = pyrender.Mesh.from_trimesh(mesh, *args, **kwargs)
		intersector = RayMeshIntersector(mesh)
		return DrawableMesh(primitives=m.primitives, is_visible=m.is_visible, intersector=intersector)


class Viewer(pyrender.Viewer):
	scroll_mode = 'cursor'
	CIRCLE_RADIUS_DEFAULT = 0.05
	CIRCLE_DEFAULT_ALPHA = 0.5
	CIRCLE_PDOWN_ALPHA = 0.1

	def __init__(self, scene, *args, **kwargs):

		# load obj
		mesh = trimesh.load('test/mesh_w_uv.obj')

		# load texture
		size = 512
		self.tex = pyrender.Texture(name='tex', width=size, height=size, source=np.zeros((size, size, 3), dtype=np.uint8),
								 source_channels='RGB')
		mat = pyrender.material.MetallicRoughnessMaterial(baseColorTexture=self.tex)

		mesh = DrawableMesh.from_trimesh(mesh, material=mat)
		self.mesh = mesh
		self.mesh_node = scene.add(mesh)

		self.cursor_radius = self.CIRCLE_RADIUS_DEFAULT

		self.circle = IcoSphere(radius=self.cursor_radius)
		self.circle.set_alpha(self.CIRCLE_DEFAULT_ALPHA)
		self.circle_node = scene.add(self.circle.mesh)

		self.cam_node = scene.main_camera_node
		self.cam = self.cam_node.camera

		self.H, self.W = viewport_size

		super().__init__(scene, *args, **kwargs, auto_start=False, viewport_size=viewport_size)

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


scene = pyrender.Scene()

camera = pyrender.camera.PerspectiveCamera(yfov=1.0, aspectRatio=1.0, znear=0.001)
pose = np.eye(4)
pose[2, 3] = 0.5
scene.add(camera, pose=pose)

viewer = Viewer(scene, run_in_thread=False, use_raymond_lighting=True, show_world_axis=True)