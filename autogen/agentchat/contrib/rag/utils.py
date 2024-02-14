import sys
import logging
import importlib

logger = logging.getLogger("autogen.agentchat.contrib.rag")
logger.setLevel(logging.INFO)

lazy_imported = {}


def lazy_import(module_name: str, attr_name: str = None):
    """lazy import module and attribute.
    Args:
        module_name: The name of the module to import.
        attr_name: The name of the attribute to import.
    Returns:
        The imported module or attribute.

    ```python
    from autogen.agentchat.contrib.rag.utils import lazy_import
    os = lazy_import("os")
    p = lazy_import("os", "path")
    print(os)
    print(p)
    print(os.path is p)  # True
    ```
    """
    if module_name not in lazy_imported:
        try:
            lazy_imported[module_name] = importlib.import_module(module_name)
        except ImportError:
            logger.error(f"Failed to import {module_name}.")
            return None
    if attr_name:
        attr = getattr(lazy_imported[module_name], attr_name, None)
        if attr is None:
            logger.error(f"Failed to import {attr_name} from {module_name}")
            return None
        else:
            return attr
    else:
        return lazy_imported[module_name]
