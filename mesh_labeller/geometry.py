import pyrender
import trimesh
from trimesh.ray.ray_pyembree import RayMeshIntersector
import numpy as np

from rasterize_tris import rasterize_tris
from mesh_labeller.texture import Texture

class IcoSphere:
	def __init__(self, radius=0.05):
		self.trimesh = trimesh.creation.icosphere(radius=radius)
		self.trimesh.visual.vertex_colors = [0.0, 0.0, 1.0, 1.0]
		self.mesh = pyrender.Mesh.from_trimesh(self.trimesh)

	def set_alpha(self, alpha=1.):
		self.mesh.primitives[0].material.baseColorFactor[3] = alpha



class DrawableMesh(pyrender.Mesh):
	def __init__(self, primitives, is_visible=True, intersector: RayMeshIntersector=None,
				 base_colour=(0, 0, 0)):
		self.intersector = intersector

		super().__init__(primitives, is_visible=is_visible)

		self.texture = Texture(1024, 1024, base_colour=base_colour, target=self)
		self.draw_tex()

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

	def draw_tex(self):
		"""Draw texture on mesh"""
		self.texture.draw()

	def draw_triangles(self, triangles, colour=(1,0,0), fmt='UV'):
		"""Draw a series of triangles to the UV map.
		fmt='UV': triangles are in UV coordinates (0-1)
		fmt='pixel': triangles are in pixel coordinates (0-width, 0-height)"""

		size = self.texture.H

		if fmt == 'UV':
			triangles = np.clip((triangles * size).astype(np.float32), 0, size-1)
			triangles[..., 1] = (size-1) - triangles[..., 1]  # +ve v -> -ve y

		# Rasterize triangles to mask
		# Seams in UV map will appear as gaps in the mask, so dilate the mask to fill them in
		mask = rasterize_tris(triangles, size, method='per-triangle', dilation=1)
		self.texture.color_mask(mask, colour)


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
	def from_trimesh(mesh, *args, base_colour=(0, 0, 0), **kwargs):
		m = pyrender.Mesh.from_trimesh(mesh, *args, **kwargs)
		intersector = RayMeshIntersector(mesh)
		return DrawableMesh(primitives=m.primitives, is_visible=m.is_visible, intersector=intersector,
							base_colour=base_colour)
