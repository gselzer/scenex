from __future__ import annotations

from typing import TYPE_CHECKING

import glfw

from scenex.events._auto import App, EventFilter
from scenex.events.events import MouseButton, MouseEvent

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import Any

    from scenex.events import Event

BUTTONMAP = {
    glfw.MOUSE_BUTTON_LEFT: MouseButton.LEFT,
    glfw.MOUSE_BUTTON_RIGHT: MouseButton.RIGHT,
    glfw.MOUSE_BUTTON_MIDDLE: MouseButton.MIDDLE,
}


class GlfwEventFilter(EventFilter):
    def __init__(self, canvas: Any, filter_func: Callable[[Event], bool]) -> None:
        self._canvas = canvas
        self._filter_func = filter_func
        self._active_buttons: set[MouseButton] = set()
        self._window_id = self._guess_id(canvas)
        # TODO: Maybe save the old callbacks?
        glfw.set_cursor_pos_callback(self._window_id, self._cursor_pos_callback)
        glfw.set_mouse_button_callback(self._window_id, self._mouse_button_callback)

    def _guess_id(self, canvas: Any) -> Any:
        # vispy
        if window := getattr(canvas, "_id", None):
            return window
        # rendercanvas
        if window := getattr(canvas, "_window", None):
            return window

    def uninstall(self) -> None:
        raise NotImplementedError(
            "Uninstalling GLFW event filters is not yet supported."
        )

    def _cursor_pos_callback(self, window: Any, xpos: float, ypos: float) -> None:
        """Handle cursor position events."""
        self._filter_func(MouseEvent(type="move", pos=(xpos, ypos), buttons=set()))

    def _mouse_button_callback(
        self, window: Any, button: int, action: int, mods: int
    ) -> None:
        pos = glfw.get_cursor_pos(window)
        # Mouse click event
        if button in BUTTONMAP:
            if action == glfw.PRESS:
                self._active_buttons.add(BUTTONMAP[button])
                self._filter_func(
                    MouseEvent(type="press", pos=pos, buttons=self._active_buttons)
                )
            elif action == glfw.RELEASE:
                self._active_buttons.remove(BUTTONMAP[button])
                self._filter_func(
                    MouseEvent(type="release", pos=pos, buttons=self._active_buttons)
                )


class GlfwAppWrap(App):
    """Provider for GLFW."""

    def create_app(self) -> Any:
        glfw.init()
        # Nothing really to return here...
        return None

    def install_event_filter(
        self, canvas: Any, filter_func: Callable[[Event], bool]
    ) -> EventFilter:
        return GlfwEventFilter(canvas, filter_func)

    def show(self, canvas: Any, visible: bool) -> None:
        if visible:
            glfw.show_window(canvas._id)
        else:
            glfw.hide_window(canvas._id)
