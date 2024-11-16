
from collections.abc import Callable
import importlib


def load_class_from_module(class_name: str, from_module: str) -> Callable:
    """ Load a pre-implemented class from a module. """
    module = importlib.import_module(from_module)
    try:
        return getattr(module, class_name)
    except AttributeError:
        try:
            return getattr(module, class_name.capitalize())
        except AttributeError:
            raise AttributeError(
                f'Class {class_name} not found in module {from_module}!')
