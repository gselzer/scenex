from typing import Any

from scenex.adaptors._registry import AdaptorRegistry


class Qt3DAdaptorRegistry(AdaptorRegistry):
    def get_adaptor_class(self, obj: Any) -> type:
        from scenex.adaptors import _qt3d

        obj_type_name = obj.__class__.__name__
        return getattr(_qt3d, f"{obj_type_name}")  # type: ignore


adaptors = Qt3DAdaptorRegistry()
get_adaptor = adaptors.get_adaptor
