from typing import TYPE_CHECKING, Any

from langchain._api import create_importer

if TYPE_CHECKING:
    from langchain_community.llms import KlusterAI
    from langchain_community.chat_models import ChatKlusterAI

# Create a way to dynamically look up deprecated imports.
# Used to consolidate logic for raising deprecation warnings and
# handling optional imports.
DEPRECATED_LOOKUP = {
    "KlusterAI": "langchain_community.llms",
    "ChatKlusterAI": "langchain_community.chat_models",
}

_import_attribute = create_importer(__package__, deprecated_lookups=DEPRECATED_LOOKUP)


def __getattr__(name: str) -> Any:
    """Look up attributes dynamically."""
    return _import_attribute(name)


__all__ = [
    "KlusterAI",
    "ChatKlusterAI",
]
