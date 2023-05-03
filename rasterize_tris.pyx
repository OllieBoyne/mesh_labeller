#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
import cython
cimport numpy as np
import numpy as np

ctypedef np.float32_t DTYPE32_t
ctypedef np.uint8_t DTYPE8_t

cdef int ceil(float x):
	return int(x + 0.9999999999999999)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cdef int is_inside(DTYPE32_t[:, :] tri, int x, int y) nogil:
	"""Return True if point (x,y) is inside triangle defined by points a,b,c.
	Solution adapted from: https://stackoverflow.com/questions/2049582/how-to-determine-if-a-point-is-in-a-2d-triangle"""

	cdef float p0x = tri[0,0]
	cdef float p0y = tri[0,1]
	cdef float p1x = tri[1,0]
	cdef float p1y = tri[1,1]
	cdef float p2x = tri[2,0]
	cdef float p2y = tri[2,1]

	cdef float area = (-p1y * p2x + p0y * (-p1x + p2x) + p0x * (p1y - p2y) + p1x * p2y)
	cdef float s = (p0y * p2x - p0x * p2y + (p2y - p0y) * x + (p0x - p2x) * y) / area
	cdef float t = (p0x * p1y - p0y * p1x + (p0y - p1y) * x + (p1x - p0x) * y) / area

	return (0 < s) and (0 < t) and (s + t < 1)



# Rasterize triangles, given as a float array of N x 3 x 2 points
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cdef np.ndarray[np.uint8_t, ndim=2] _raster_tris(np.ndarray[np.float32_t, ndim=3] tris, int img_size=512,
												 method='per-triangle'):

	# Define output image as numpy array, dtype bool
	cdef np.ndarray[np.uint8_t, ndim=2] img = np.zeros((img_size, img_size), dtype=np.uint8)
	cdef DTYPE8_t[:, :] img_view = img

	cdef int NUM_TRIS = tris.shape[0]
	cdef int i, j
	cdef float min_x, max_x, min_y, max_y


	if method == 'per-pixel':

		# First find bounding box of triangles
		min_x = np.clip(np.min(tris[...,0]), 0, img_size-1)
		max_x = np.clip(np.max(tris[...,0]), 0, img_size-1)
		min_y = np.clip(np.min(tris[...,1]), 0, img_size-1)
		max_y = np.clip(np.max(tris[...,1]), 0, img_size-1)

		# Iterate through all pixels in bounding box
		for i in range(int(min_x), int(max_x)):
			for j in range(int(min_y), int(max_y)):

				for t in range(NUM_TRIS):
					if is_inside(tris[t], i, j) > 0:
						img_view[j,i] = 255
						break

	elif method == 'per-triangle':

		min_xs = np.clip(np.min(tris[..., 0], axis=1), 0, img_size - 1)
		max_xs = np.clip(np.max(tris[..., 0], axis=1), 0, img_size - 1)
		min_ys = np.clip(np.min(tris[..., 1], axis=1), 0, img_size - 1)
		max_ys = np.clip(np.max(tris[..., 1], axis=1), 0, img_size - 1)

		for t in range(NUM_TRIS):
			# Iterate through all pixels in triangle
			for i in range(int(min_xs[t]), ceil(max_xs[t])):
				for j in range(int(min_ys[t]), ceil(max_ys[t])):
					if (img_view[j, i] == 0) and is_inside(tris[t], i, j) > 0:
						img_view[j,i] = 255

	return img

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cdef np.ndarray[np.uint8_t, ndim=2] shrink_image(np.ndarray[np.uint8_t, ndim=2] img, int factor, str method='max'):
	"""Shrink an image by a target factor, using max or mean pooling.
	Not currently used."""
	cdef int img_size = img.shape[0]
	cdef int new_size = img_size // factor

	cdef np.ndarray[np.uint8_t, ndim=2] img_small = np.zeros((new_size, new_size), dtype=np.uint8)

	cdef DTYPE8_t[:, :] img_view = img
	cdef DTYPE8_t[:, :] img_view_small = img_small

	cdef int i, j, k, l
	cdef int max_val
	if method == 'max':
		for i in range(new_size):
			for j in range(new_size):
				max_val = 0
				for k in range(factor):
					for l in range(factor):
						max_val = max(max_val, img_view[i*factor + k, j*factor + l])

				img_view_small[i,j] = max_val

	elif method == 'mean':
		for i in range(new_size):
			for j in range(new_size):
				img_view_small[i,j] = np.mean(img_view[i*factor:(i+1)*factor, j*factor:(j+1)*factor])

	return img_small


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cdef np.ndarray[np.uint8_t, ndim=2] dilate(np.ndarray[np.uint8_t, ndim=2] orig_img, int dilation=1):

	cdef int img_size = orig_img.shape[0]
	cdef np.ndarray[np.uint8_t, ndim=2] img_dilated = orig_img.copy()

	cdef DTYPE8_t[:, :] orig_img_view = orig_img
	cdef DTYPE8_t[:, :] dil_img_view = img_dilated

	cdef int i, j, xmin, xmax, ymin, ymax

	for i in range(img_size):
		for j in range(img_size):
			if orig_img_view[j, i] > 0:
				xmin = max(0, i - dilation)
				xmax = min(img_size, i + dilation + 1)
				ymin = max(0, j - dilation)
				ymax = min(img_size, j + dilation + 1)

				dil_img_view[ymin:ymax, xmin:xmax] = 255

	return img_dilated


cpdef rasterize_tris(tris, img_size=512, method='per-triangle', dilation=0):
	"""Rasterize triangles, given as a float array of N x 3 points
	
	for dilation > 0, run a dilation filter on the output image, with kernel size 1+2*dilation
	(eg dilation = 1 gives a 3x3 kernel)
	"""

	res = _raster_tris(tris, img_size=img_size, method=method) > 0

	if dilation > 0:
		res = dilate(res, dilation=dilation)

	return res
