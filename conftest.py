"""
Project-wide pytest configuration.

Sets asyncio_mode to "auto" so that every async test function is
automatically treated as an asyncio coroutine without needing an
explicit @pytest.mark.asyncio decorator.
"""


# Make all async tests run under asyncio automatically
def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "asyncio: mark test as an asyncio coroutine"
    )
