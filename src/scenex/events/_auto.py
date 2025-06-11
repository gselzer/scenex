from __future__ import annotations

from typing import TYPE_CHECKING

from qtpy.QtCore import QEvent, QObject, Qt
from qtpy.QtGui import QMouseEvent
from qtpy.QtWidgets import QWidget

from scenex.events.events import MouseButton, MouseEvent

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import Any

    from scenex.events import Event


class EventFilter:
    def uninstall(self) -> None:
        """Uninstall the event filter."""
        raise NotImplementedError("This method should be implemented by subclasses.")

    pass


class QtEventFilter(QObject, EventFilter):
    def __init__(self, canvas: QWidget, filter_func: Callable[[Event], bool]) -> None:
        super(QObject, self).__init__()
        self._canvas = canvas
        self._filter_func = filter_func
        self._active_buttons: set[MouseButton] = set()

    def eventFilter(self, a0: QObject | None = None, a1: QEvent | None = None) -> bool:
        if isinstance(a0, QWidget) and isinstance(a1, QEvent):
            if evt := self._convert_event(a1):
                return self._filter_func(evt)
        return False

    def uninstall(self) -> None:
        self._canvas.removeEventFilter(self)

    def mouse_btn(self, btn: Any) -> MouseButton:
        if btn == Qt.MouseButton.LeftButton:
            return MouseButton.LEFT
        if btn == Qt.MouseButton.RightButton:
            return MouseButton.RIGHT
        if btn == Qt.MouseButton.NoButton:
            return MouseButton.NONE

        raise Exception(f"Qt mouse button {btn} is unknown")

    def _convert_event(self, qevent: QEvent) -> Event | None:
        """Convert a QEvent to a SceneX Event."""
        if isinstance(qevent, QMouseEvent):
            pos = qevent.pos()
            etype = qevent.type()
            btn = self.mouse_btn(qevent.button())
            if etype == QEvent.Type.MouseMove:
                return MouseEvent(
                    type="move", pos=(pos.x(), pos.y()), buttons=self._active_buttons
                )
            elif etype == QEvent.Type.MouseButtonDblClick:
                return MouseEvent(
                    type="double_click", pos=(pos.x(), pos.y()), buttons={btn}
                )
            elif etype == QEvent.Type.MouseButtonPress:
                self._active_buttons.add(btn)
                return MouseEvent(
                    type="press", pos=(pos.x(), pos.y()), buttons=self._active_buttons
                )
            elif etype == QEvent.Type.MouseButtonRelease:
                self._active_buttons.remove(btn)
                return MouseEvent(
                    type="release", pos=(pos.x(), pos.y()), buttons=self._active_buttons
                )
        return None


def install_event_filter(
    canvas: Any, filter_func: Callable[[Event], bool]
) -> EventFilter:
    if isinstance(canvas, QWidget):
        f = QtEventFilter(canvas, filter_func)
        canvas.installEventFilter(f)
        return f

    raise RuntimeError("Canvas is not a QWidget. Cannot install event filter.")
