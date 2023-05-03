from mesh_labeller.viewer import Viewer
import yaml

viewport_size = (500, 500)

if __name__ == '__main__':

	with open('cfg.yaml') as infile:
		cfg = yaml.load(infile, Loader=yaml.FullLoader)

	viewer = Viewer(cfg=cfg,
					viewport_size=viewport_size,
					show_world_axis=True,
					window_title='Mesh Labeller')