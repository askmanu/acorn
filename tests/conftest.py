"""Shared test configuration and fixtures."""


def pytest_collection_modifyitems(session, config, items):
    """Set a default model on Module subclasses that don't have one.

    This avoids having to set model = "test-model" on every test Module subclass.
    """
    from acorn.module import Module
    Module._original_model = Module.model
    if not Module.model:
        Module.model = "test-model"


def pytest_unconfigure(config):
    """Restore original Module model default."""
    from acorn.module import Module
    if hasattr(Module, '_original_model'):
        Module.model = Module._original_model
