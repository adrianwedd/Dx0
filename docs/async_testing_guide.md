# Async Testing Guide

This guide covers the asynchronous testing configuration and best practices for the Dx0 project.

## Overview

The Dx0 project uses async/await patterns extensively, particularly for:
- WebSocket connections in the UI
- HTTP clients for API testing
- Database operations
- LLM client interactions

## Configuration

### pytest-asyncio Setup

The project is configured to use `pytest-asyncio` for async test execution. The configuration is defined in `pyproject.toml`:

```toml
[tool.pytest.ini_options]
asyncio_mode = "auto"
markers = [
    "asyncio: mark test as async",
]
testpaths = ["tests"]
python_files = "test_*.py"
python_classes = "Test*"
python_functions = "test_*"
addopts = [
    "-v",
    "--tb=short",
    "--strict-markers",
]
```

### Dependencies

Async testing requires the following packages (included in `requirements-dev.txt`):
- `pytest-asyncio==0.23.6` - Async test support
- `httpx==0.27.2` - Async HTTP client for API testing
- `httpx_ws==0.7.2` - WebSocket support for httpx

## Writing Async Tests

### Basic Async Test

```python
import pytest

@pytest.mark.asyncio
async def test_async_function():
    result = await some_async_function()
    assert result == expected_value
```

### WebSocket Testing Pattern

For WebSocket tests (see `tests/test_ui.py`):

```python
@pytest.mark.asyncio
async def test_websocket_chat():
    config = uvicorn.Config(app, host="127.0.0.1", port=8000, log_level="error")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    while not server.started:
        await asyncio.sleep(0.01)

    async with httpx.AsyncClient(base_url="http://127.0.0.1:8000") as client:
        # Login first
        res = await client.post("/api/v1/login", json={"username": "physician", "password": "secret"})
        token = res.json()["access_token"]
        
        # Test WebSocket connection
        async with aconnect_ws(f"ws://127.0.0.1:8000/api/v1/ws?token={token}", client) as ws:
            await ws.send_json({"action": "question", "content": "test"})
            data = await ws.receive_json()
            assert "reply" in data

    server.should_exit = True
    thread.join()
```

### HTTP Client Testing

```python
@pytest.mark.asyncio
async def test_api_endpoint():
    async with httpx.AsyncClient(base_url="http://testserver") as client:
        response = await client.get("/api/v1/endpoint")
        assert response.status_code == 200
```

## Session Backend Testing

The project uses a session backend system for managing user sessions, failed logins, and message rate limiting. For testing, helper methods are available:

```python
@pytest.mark.asyncio
async def test_with_clean_session():
    # Clear session data before test
    await ui_app.SESSION_BACKEND.clear_all_session_data()
    
    # Your test code here
    
    # Session data is automatically cleaned between tests
```

### Available Testing Methods

- `clear_failed_logins()` - Clear failed login attempts
- `clear_all_session_data()` - Clear all session data including message history

## Common Patterns

### Monkeypatching in Async Tests

```python
@pytest.mark.asyncio
async def test_with_monkeypatch(monkeypatch):
    monkeypatch.setattr(module, "SETTING", new_value)
    await ui_app.SESSION_BACKEND.clear_all_session_data()
    
    # Test with modified settings
```

### Test Isolation

Each async test should be independent. Use these patterns:

1. **Clean session data**: Use session backend clearing methods
2. **Separate ports**: Use different ports for concurrent server tests
3. **Timeout handling**: Always set timeouts for long-running operations

### Server Lifecycle Management

```python
@pytest.mark.asyncio
async def test_server_lifecycle():
    # Start server
    config = uvicorn.Config(app, host="127.0.0.1", port=8001, log_level="error")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    
    # Wait for server to start
    while not server.started:
        await asyncio.sleep(0.01)
    
    try:
        # Your test logic here
        pass
    finally:
        # Always clean up
        server.should_exit = True
        thread.join()
```

## Running Async Tests

### Run All Async Tests
```bash
python -m pytest tests/test_ui.py -v
python -m pytest tests/test_llm_engine_async.py -v
python -m pytest tests/test_orchestrator.py -v
python -m pytest tests/test_evaluation.py -v
```

### Run Specific Async Test
```bash
python -m pytest tests/test_ui.py::test_websocket_chat -v
```

### Debug Async Tests
```bash
python -m pytest tests/test_ui.py::test_websocket_chat -v -s --tb=long
```

## Troubleshooting

### Common Issues

1. **"async def functions are not natively supported"**
   - Ensure `pytest-asyncio` is installed and configured
   - Check that `@pytest.mark.asyncio` decorator is present

2. **Tests hanging or timing out**
   - Add proper cleanup in finally blocks
   - Use `asyncio.wait_for()` for operations with timeouts
   - Ensure servers are properly shut down

3. **Port conflicts**
   - Use different ports for concurrent tests
   - Consider using pytest-xdist for parallel execution

4. **Session data conflicts**
   - Use session backend clearing methods
   - Ensure proper test isolation

### Configuration Verification

To verify async configuration is working:

```bash
python -m pytest --version  # Should show pytest-asyncio plugin
python -m pytest tests/test_ui.py::test_websocket_chat -v  # Should pass
```

## Best Practices

1. **Always use unique ports** for server tests to avoid conflicts
2. **Clean up resources** in finally blocks or using context managers  
3. **Use session backend helpers** for test isolation
4. **Set reasonable timeouts** for async operations
5. **Test both success and error cases** for async operations
6. **Mock external dependencies** to avoid flaky tests
7. **Use descriptive test names** that indicate async behavior

## Examples

See the following files for comprehensive async testing examples:
- `tests/test_ui.py` - WebSocket and HTTP testing
- `tests/test_llm_engine_async.py` - Concurrent async operations
- `tests/test_orchestrator.py` - Async orchestration patterns
- `tests/test_evaluation.py` - Batch async evaluation