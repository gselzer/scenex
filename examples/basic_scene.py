import cmap
import numpy as np

import scenex as snx

try:
    from scenex.imgui import add_imgui_controls
except ImportError:
    print("imgui not available, skipping imgui controls")
    add_imgui_controls = None  # type: ignore[assignment]

view = snx.View(
    blending="default",
    scene=snx.Scene(
        children=[
            snx.Image(
                name="foo",
                data=np.random.randint(0, 255, (200, 200)).astype(np.uint8),
                cmap=cmap.Colormap("viridis"),
                # transform=snx.Transform().translated((0, -100, 0)),
                transform=snx.Transform()
                .translated((0, -100, 0))
                .rotated(45, (0, 1, 0))
                .rotated(45, (1, 0, 0)),
                # transform=snx.Transform().translated((0, -50, 0)).scaled((0.5, 1, 1)),
                clims=(0, 255),
                opacity=0.7,
            ),
            snx.Image(
                name="bar",
                data=np.random.randint(0, 255, (200, 200)).astype(np.uint8),
                cmap=cmap.Colormap("inferno"),
                clims=(0, 255),
                opacity=0.7,
            ),
            # snx.Points(
            #     coords=np.random.randint(0, 200, (100, 2)).astype(np.uint8),
            #     size=5,
            #     face_color=cmap.Color("coral"),
            #     edge_color=cmap.Color("purple"),
            #     transform=snx.Transform().translated((0, -50)),
            # ),
        ]
    ),
)

# example of adding an object to a scene
X, Y = np.meshgrid(np.linspace(-10, 10, 100), np.linspace(-10, 10, 100))
sine_img = (np.sin(X) * np.cos(Y)).astype(np.float32)
# image = snx.Image(name="sine image", data=sine_img, clims=(-1, 1))
# view.scene.add_child(image)

# both are optional, just for example
snx.use("pygfx")
# snx.use("vispy")

snx.show(view)
# if add_imgui_controls is not None:
#     add_imgui_controls(view)
snx.run()
