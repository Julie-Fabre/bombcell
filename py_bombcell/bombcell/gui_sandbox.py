import numpy as np
import ctypes
import datoviz as dvz
from datoviz import (
    S_,  # Python string to ctypes char*
    V_, # booleans
    vec2,
    vec3,
    vec4,
)


# -------------------------------------------------------------------------------------------------
# 1. Creating the app > batch > scene > figure > panels
# -------------------------------------------------------------------------------------------------

# Boilerplate.
app = dvz.app(0)
batch = dvz.app_batch(app)
scene = dvz.scene(batch)

# Create a figure.
# NOTE: to use a GUI, use the flag dvz.CANVAS_FLAGS_IMGUI. Use 0 instead if there is no GUI.
W = 800
H = 800
figure = dvz.figure(scene, W, H, dvz.CANVAS_FLAGS_IMGUI)

# Create panels
# panel = dvz.panel_default(figure)
n_rows = 2
n_cols = 2
w = W * 1.0 / n_cols
h = H * 1.0 / n_rows
p00 = dvz.panel(figure, 0 * w, 0 * h, w, h)
p01 = dvz.panel(figure, 1 * w, 0 * h, w, h)
p10 = dvz.panel(figure, 0 * w, 1 * h, w, h)
p11 = dvz.panel(figure, 1 * w, 1 * h, w, h)
panels = [p00, p01, p10, p11]

# -------------------------------------------------------------------------------------------------
# 2. Create visuals in batch
# -------------------------------------------------------------------------------------------------

# Cube colors.
colors = np.array([
    [255, 0, 0, 255],
    [0, 255, 0, 255],
    [0, 0, 255, 255],
    [255, 255, 0, 255],
    [255, 0, 255, 255],
    [0, 255, 255, 255],
], dtype=np.uint8)
shape = dvz.shape_cube(colors)

cubes = []
for i in range(len(panels) - 1):
    # Create mesh visuals directly instantiated with the shape data.
    cube = dvz.mesh_shape(batch, shape, dvz.MESH_FLAGS_LIGHTING)
    dvz.mesh_light_pos(cube, vec3(-1, +1, +10)) # light source position
    dvz.mesh_light_params(cube, vec4(.5, .5, .5, 16)) # light parameters
    cubes.append(cube)


# Markers
markers = dvz.marker(batch, 0)
n = 1_000
dvz.marker_alloc(markers, n) # memory allocation.

# Marker positions.
pos = np.random.normal(size=(n, 3), scale=.25).astype(np.float32)
dvz.marker_position(markers, 0, n, pos, 0)

# Marker colors.
color = np.random.uniform(size=(n, 4), low=50, high=240).astype(np.uint8)
color[:, 3] = 240
dvz.marker_color(markers, 0, n, color, 0)

# Marker sizes.
size = np.random.uniform(size=(n,), low=1, high=20).astype(np.float32)
dvz.marker_size(markers, 0, n, size, 0)

# Marker parameters.
dvz.marker_aspect(markers, dvz.MARKER_ASPECT_OUTLINE)
dvz.marker_shape(markers, dvz.MARKER_SHAPE_DISC)


visuals = cubes + [markers]

# -------------------------------------------------------------------------------------------------
# 3. Add visuals to panels
# -------------------------------------------------------------------------------------------------

for i, (panel, visual) in enumerate(zip(panels, visuals)):

    # 3D Arcball interactivity - mutually exclusive with panzoom.
    if i < 3:
        dvz.arcball_initial(dvz.panel_arcball(panel), vec3(+0.6, -1.2, +3.0))

    # 2D Panzoom interactivity - mutually exclusive with arcball.
    else:
        dvz.panel_panzoom(panel)

    # Add the visual to the panels.
    dvz.panel_visual(panel, visual, 0)

    # update panels with arcballs and visuals
    dvz.panel_update(panel)


# -------------------------------------------------------------------------------------------------
# 4. Defining the GUI associated with the full figure (not specific panels)
# -------------------------------------------------------------------------------------------------

# There are four steps to add a GUI
# i.    Initialize the figure with the flag `dvz.CANVAS_FLAGS_IMGUI`` (above)
# ii.   Define a global-scoped object representing the variable to be updated by the GUI.
# iii.  Define the GUI callback.
# iv.   Call `dvz.app_gui(...)`

# A wrapped boolean value with initial value False.
checked = V_(True, ctypes.c_bool)

# GUI callback function. Called at every frame.
@dvz.gui
def gui_callback_check(app, fid, ev):
    global visuals
    # Set the size of the next GUI dialog.
    dvz.gui_corner(0, vec2(0, 0))
    dvz.gui_size(vec2(190, 60))

    # Start a GUI dialog with a dialog title.
    dvz.gui_begin(S_("Check GUI"), 0)

    # Add a checkbox
    with checked:  # Wrap the boolean value.
        # Return True if the checkbox's state has changed.
        if dvz.gui_checkbox(S_("Show visual in panel 00"), checked.P_):
            #                                              ^^^^^^^^^^ pass a C pointer to our wrapped bool
            is_checked = checked.value  # Python variable with the checkbox's state

            # Show/hide the visual.
            dvz.visual_show(visuals[0], is_checked)

            # Update the figure after its composition has changed.
            dvz.figure_update(figure)

    # End the GUI dialog.
    dvz.gui_end()

@dvz.gui
def gui_callback_resize(app, fid, ev):
    global shape
    global visuals
    # Set the size of the next GUI dialog.
    dvz.gui_corner(0, vec2(190, 0))
    dvz.gui_size(vec2(180, 100))

    # Start a GUI dialog with a dialog title.
    dvz.gui_begin(S_("Resize GUI"), 0)

    # Add two buttons. The functions return whether the button was pressed.
    incr = dvz.gui_button(S_("Increase"), 150, 30)
    decr = dvz.gui_button(S_("Decrease"), 150, 30)

    # Scaling factor.
    scale = 1.0
    if incr:
        scale = 1.1
    elif decr:
        scale = 0.9

    if incr or decr:
        # Start recording shape transforms for all vertices in the shape (first=0, count=0=all).
        dvz.shape_begin(shape, 0, 0)

        # Scaling transform.
        dvz.shape_scale(shape, vec3(scale, scale, scale))

        # Stop recording the shape transforms.
        dvz.shape_end(shape)

        # Update the mesh visual data with the new shape's data.
        for i, visual in enumerate(visuals):
            if i < 3:
                dvz.mesh_reshape(visual, shape)

    # End the GUI dialog.
    dvz.gui_end()

# Associate GUIs with the figure.
dvz.app_gui(app, dvz.figure_id(figure), gui_callback_check, None)
dvz.app_gui(app, dvz.figure_id(figure), gui_callback_resize, None)


# -------------------------------------------------------------------------------------------------
# 5. Run and cleanup
# -------------------------------------------------------------------------------------------------

# Run the application.
dvz.scene_run(scene, app, 0)

# Cleanup.
dvz.shape_destroy(shape)
dvz.scene_destroy(scene)
dvz.app_destroy(app)