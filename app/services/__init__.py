from importlib import import_module

# Lazily expose heavy submodules so that a plain
#   import app.services
# or
#   from app.services import call_tracker
# does NOT pull in provider-dependent code (anthropic, openai) eagerly.
#
# unittest.mock.patch("app.services.fact_lock_service.*") works because
# patch calls _dot_lookup → getattr(app.services, "fact_lock_service"),
# which triggers __getattr__ below, which imports the module on first access
# and caches it in globals() for subsequent lookups.

__all__ = ["fact_lock_service"]


def __getattr__(name: str):
    if name == "fact_lock_service":
        module = import_module("app.services.fact_lock_service")
        globals()[name] = module   # cache so __getattr__ is not called again
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
