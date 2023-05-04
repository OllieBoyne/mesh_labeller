from mesh_labeller.viewer import Viewer
import yaml

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--cfg_file', type=str, default='cfgs/default.yaml')

viewport_size = (500, 500)

if __name__ == '__main__':

	args = parser.parse_args()

	with open(args.cfg_file) as infile:
		cfg = yaml.load(infile, Loader=yaml.FullLoader)

	viewer = Viewer(cfg=cfg,
					viewport_size=viewport_size,
					window_title='Mesh Labeller')