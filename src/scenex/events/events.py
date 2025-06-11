from __future__ import annotations

from dataclasses import dataclass
from enum import IntFlag, auto


@dataclass
class Event:
    """A general interaction event."""

    # TODO: Enum?
    type: str


class MouseButton(IntFlag):
    """A general mouse interaction event."""

    LEFT = auto()
    MIDDLE = auto()
    RIGHT = auto()
    NONE = auto()


@dataclass
class MouseEvent(Event):
    """A general mouse interaction event."""

    type: str
    pos: tuple[float, float]
    # TODO: Enum?
    buttons: set[MouseButton]

    @property
    def x(self) -> float:
        """The x-coordinate of the mouse event."""
        return self.pos[0]

    @property
    def y(self) -> float:
        """The y-coordinate of the mouse event."""
        return self.pos[1]
