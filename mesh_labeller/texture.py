import numpy as np
import cv2
from collections import namedtuple
import pyrender

Category = namedtuple('Category', 'ID name rgb')
class Paint:
	def __init__(self, cfg):
		"""cfg: dict of class: (r, g, b)"""
		self.cfg = cfg
		self.categories = []
		for n, (name, rgb) in enumerate(cfg.items()):
			self.categories.append(Category(ID=n, name=name, rgb=rgb))

		self.current_cat = 0

	def cycle_up(self):
		self.current_cat = (self.current_cat + 1) % len(self.categories)

	def cycle_down(self):
		self.current_cat = (self.current_cat - 1) % len(self.categories)

	def set_class(self, ID):
		self.current_cat = ID

	@property
	def rgb(self):
		return self.categories[self.current_cat].rgb

	@property
	def name(self):
		return self.categories[self.current_cat].name

	@property
	def ID(self):
		return self.categories[self.current_cat].ID

	def __getitem__(self, item) -> Category:
		if isinstance(item, int):
			return self.categories[item]

		if isinstance(item, str):
			for cat in self.categories:
				if cat.name == item:
					return cat

			raise IndexError(f"Category {item} not found.")



class Texture:
	def __init__(self, H, W, dtype=np.uint8, base_colour=(0, 0, 0),
				 UNDO_HISTORY=10, target=None, loc=None):
		self.H = H
		self.W = W
		self.dtype = dtype

		self.data = np.zeros((H, W, 3), dtype=dtype)
		self.data[:] = base_colour

		if loc is not None:
			self.data = cv2.cvtColor(cv2.imread(loc), cv2.COLOR_BGR2RGB).astype(dtype)

		self.target = target

		self.UNDO_HISTORY = UNDO_HISTORY
		self.undo_archive = [] # FIFO of previous states for undo
		self.update_archive()

	def draw(self):
		size = self.H
		tex = pyrender.Texture(name='tex', width=size, height=size, source=self.data, source_channels='RGB')
		self.target.primitives[0].material.baseColorTexture = tex

	def update_archive(self):
		self.undo_archive.insert(0, self.data.copy())
		if len(self.undo_archive) > self.UNDO_HISTORY:
			self.undo_archive.pop()

	def undo(self):
		if len(self.undo_archive) > 0:
			self.data = self.undo_archive.pop(0)
		self.draw()

	def color_mask(self, mask, color):
		self.update_archive()
		self.data[mask] = color
		self.draw()

	def save(self, path):
		cv2.imwrite(path, cv2.cvtColor(self.data, cv2.COLOR_RGB2BGR))