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
                name="bar",
                data=np.random.randint(0, 255, (200, 200)).astype(np.uint8),
                cmap=cmap.Colormap("inferno"),
                clims=(0, 255),
                opacity=0.7,
                interactive=True,
            ),
        ],
        interactive=True,
    ),
)

view.camera.interactive = True

# both are optional, just for example
snx.use("pygfx")
# snx.use("vispy")

snx.show(view)
snx.run()
