import cmap
import numpy as np

import scenex as snx
from scenex.events.events import Event, MouseEvent

try:
    from scenex.imgui import add_imgui_controls
except ImportError:
    print("imgui not available, skipping imgui controls")
    add_imgui_controls = None  # type: ignore[assignment]

img = snx.Image(
    data=np.random.randint(0, 255, (200, 200)).astype(np.uint8),
    cmap=cmap.Colormap("viridis"),
    transform=snx.Transform().scaled((1.3, 0.5)).translated((-40, 20)),
    clims=(0, 255),
    opacity=0.7,
)

view1 = snx.View(
    blending="default",
    scene=snx.Scene(
        children=[
            img,
            snx.Points(
                coords=np.random.randint(0, 200, (100, 2)).astype(np.uint8),
                size=5,
                face_color=cmap.Color("coral"),
                edge_color=cmap.Color("purple"),
                transform=snx.Transform().translated((0, -50)),
            ),
        ]
    ),
)
view2 = snx.View(
    blending="default",
    scene=snx.Scene(
        children=[
            snx.Points(
                coords=np.random.randint(0, 200, (100, 2)).astype(np.uint8),
                size=5,
                face_color=cmap.Color("coral"),
                edge_color=cmap.Color("purple"),
                transform=snx.Transform().translated((0, -50)),
            ),
        ]
    ),
)
canvas = snx.Canvas()
canvas.views.extend([view1, view2])


def img_filter(event: Event) -> bool:
    """Example filter function for the image."""
    if isinstance(event, MouseEvent) and event.type == "move":
        print(f"Mouse moved over image at {event.pos}")
        return True
    return False


img.filter = img_filter

# example of adding an object to a scene

# both are optional, just for example
# snx.use("pygfx")
snx.use("vispy")

snx.show(canvas)
# if add_imgui_controls is not None:
#  add_imgui_controls(view)
snx.run()
