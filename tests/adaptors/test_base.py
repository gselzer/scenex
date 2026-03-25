import logging
from typing import TYPE_CHECKING

from psygnal import EmissionInfo, PathStep, Signal

from scenex.adaptors._base import Adaptor

if TYPE_CHECKING:
    import pytest


class _MinimalAdaptor(Adaptor):
    def __init__(self, _obj: object) -> None:
        pass


class _TestSignals:
    """Minimal signal holder for constructing real EmissionInfo objects in tests."""

    foo = Signal(str)
    nonexistent_field = Signal(int)


_sigs = _TestSignals()


def test_handle_event_nested_field_no_error(
    caplog: "pytest.LogCaptureFixture",
) -> None:
    """handle_event must not warn for missing methods corresponding to child fields."""
    adaptor = _MinimalAdaptor(None)

    # The pattern we want to avoid here corresponds to an evented model parent with an
    # evented model field "child" with a field "foo". When "parent" has an adaptor but
    # "child" does not, we want to allow "parent"'s adaptor to be able to handle changes
    # to "child.foo" without logging errors about missing "_snx_set_child_foo" method
    # on the parent.
    info = EmissionInfo(
        signal=_sigs.foo,
        args=("bar",),
        path=(PathStep(attr="child"), PathStep(attr="foo")),  # len > 1 → nested
    )

    with caplog.at_level(logging.ERROR, logger="scenex.adaptors"):
        adaptor.handle_event(info)

    assert len(caplog.records) == 0


def test_handle_event_direct_field_missing_setter_logs_error(
    caplog: "pytest.LogCaptureFixture",
) -> None:
    """handle_event must log an error when a direct model field has no setter."""
    adaptor = _MinimalAdaptor(None)

    info = EmissionInfo(
        signal=_sigs.nonexistent_field,
        args=(42,),
        path=(PathStep(attr="nonexistent_field"),),  # len == 1 → direct field
    )

    with caplog.at_level(logging.ERROR, logger="scenex.adaptors"):
        adaptor.handle_event(info)

    assert len(caplog.records) == 1
