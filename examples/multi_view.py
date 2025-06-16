import random
from typing import cast

import cmap
import numpy as np

import scenex as snx
from scenex.events.events import Event, MouseEvent
from scenex.model import Node

img = snx.Image(
    name="image",
    data=np.random.randint(0, 255, (200, 200)).astype(np.uint8),
    cmap=cmap.Colormap("viridis"),
    transform=snx.Transform().rotated(90).translated((0, -50, -50)),
    clims=(0, 255),
    opacity=0.7,
)

view1 = snx.View(
    blending="default",
    scene=snx.Scene(
        children=[
            img,
            snx.Points(
                name="points1",
                coords=np.random.randint(0, 200, (100, 2)).astype(np.uint8),
                size=5,
                face_color=cmap.Color("coral"),
                edge_color=cmap.Color("purple"),
                transform=snx.Transform().translated((0, -50, -50)),
            ),
        ]
    ),
)
view2 = snx.View(
    blending="default",
    scene=snx.Scene(
        children=[
            snx.Points(
                name="points2",
                coords=np.random.randint(0, 200, (100, 2)).astype(np.uint8),
                size=5,
                face_color=cmap.Color("red"),
                edge_color=cmap.Color("blue"),
                transform=snx.Transform().translated((0, -50)),
            ),
        ]
    ),
)
canvas = snx.Canvas()
canvas.views.extend([view1])
# canvas.views.extend([view1, view2])


def scene_filter(event: Event, n: Node) -> bool:
    """Example filter function for the image."""
    if isinstance(event, MouseEvent) and event.type == "move":
        print(f"Scene saw Mouse moved over {n.name} at {event.pos}")
        # return isinstance(n, snx.Image)
    return False


# FIXME: It'd be cool to know this was an image
def img_filter(event: Event, n: Node) -> bool:
    """Example filter function for the image."""
    if isinstance(event, MouseEvent) and event.type == "move":
        print(f"Image saw Mouse moved over {n.name} at {event.pos}")
        return True
    elif isinstance(event, MouseEvent) and event.type == "press":
        img = cast("snx.Image", n)
        img.cmap = cmap.Colormap(
            random.choice(
                [
                    "viridis",
                    "plasma",
                    "inferno",
                    "magma",
                    "cividis",
                    "turbo",
                    "coolwarm",
                ]
            )
        )

    return True


# img.set_event_filter(img_filter)
# view1.scene.set_event_filter(scene_filter)

# example of adding an object to a scene

# both are optional, just for example
snx.use("pygfx")
# snx.use("vispy")

snx.show(canvas)
snx.run()
