## MeshLabeller

A tool for labelling 3D meshes. You define custom classes, load an OBJ file, and 'paint' each class onto the mesh.
The result is an output PNG texture which can be used as a label map for the mesh.


<p float="left">
<img src="/docs/recording.gif" width="45%"/> <img src="/docs/render.gif" width="45%"/>
<p>

## Usage

```python main.py```

This loads the settings defined in `cfgs/default.yaml` - edit this or create your own with custom classes.

## Controls

####Mouse

 Movement     | Response 
 ---          | --- 
 Left click   | Paint at target 
 Middle click | Pan camera about target 
 Right click  | Rotate camera about target 
 Scroll | Change target size OR (if CTRL pressed) Zoom in/out to target 


####Keyboard

Key | Action
--- | ---
`0-9` | Select class
`>` | Next class
`<` | Previous class
`CTRL`/`⌘` | While held, Scroll to zoom
`S` | Save label map as `.png`
`O` | Open new `.obj` file
`Z` | Undo last paint
`P` | Use mouse pan mode
`R` | Start recording / Save recording as `.gif`


## Contribution

## Installation

```python setup.py install```

Tested on Windows 10, MacOS 13.1 [^1].

[^1]: See [`docs/mac_install.md`](docs/mac_install.md) for details of installation on Apple Silicon.
