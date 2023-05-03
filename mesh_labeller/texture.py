import numpy as np
import cv2
from collections import namedtuple

Category = namedtuple('Category', 'ID name rgb')
class Paint:
	def __init__(self, cfg):
		"""cfg: dict of class: (r, g, b)"""
		self.cfg = cfg
		self.categories =  []
		for n, (name, rgb) in enumerate(cfg.items()):
			self.categories.append(Category(ID=n, name=name, rgb=rgb))

		self.current_cat = 0

	def cycle_up(self):
		self.current_cat = (self.current_cat + 1) % len(self.categories)

	def cycle_down(self):
		self.current_cat = (self.current_cat - 1) % len(self.categories)

	@property
	def rgb(self):
		return self.categories[self.current_cat].rgb

	@property
	def name(self):
		return self.categories[self.current_cat].name

	def __getitem__(self, item) -> Category:
		if isinstance(item, int):
			return self.categories[item]

		if isinstance(item, str):
			for cat in self.categories:
				if cat.name == item:
					return cat

			raise IndexError(f"Category {item} not found.")



class Texture:
	def __init__(self, H, W, dtype=np.uint8, base_colour=(0, 0, 0)):
		self.H = H
		self.W = W
		self.dtype = dtype
		self.data = np.zeros((H, W, 3), dtype=dtype)
		self.data[:] = base_colour

	def color_mask(self, mask, color):
		self.data[mask] = color

	def save(self, path):
		cv2.imwrite(path, cv2.cvtColor(self.data, cv2.COLOR_RGB2BGR))