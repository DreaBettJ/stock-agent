"""Tests for retry module."""

from __future__ import annotations

import asyncio

import pytest

from mini_agent.retry import RetryConfig, RetryExhaustedError, async_retry


class TestRetryConfig:
    """Test RetryConfig class."""

    def test_default_values(self):
        config = RetryConfig()
        assert config.enabled is True
        assert config.max_retries == 3
        assert config.initial_delay == 1.0
        assert config.max_delay == 60.0
        assert config.exponential_base == 2.0
        assert config.retryable_exceptions == (Exception,)

    def test_custom_values(self):
        config = RetryConfig(
            enabled=False,
            max_retries=5,
            initial_delay=0.5,
            max_delay=30.0,
            exponential_base=1.5,
            retryable_exceptions=(ValueError, TypeError),
        )
        assert config.enabled is False
        assert config.max_retries == 5
        assert config.initial_delay == 0.5
        assert config.max_delay == 30.0
        assert config.exponential_base == 1.5
        assert config.retryable_exceptions == (ValueError, TypeError)

    def test_calculate_delay(self):
        config = RetryConfig(initial_delay=1.0, exponential_base=2.0, max_delay=100.0)
        
        # Exponential backoff: initial * base^attempt
        assert config.calculate_delay(0) == 1.0
        assert config.calculate_delay(1) == 2.0
        assert config.calculate_delay(2) == 4.0
        assert config.calculate_delay(3) == 8.0
        assert config.calculate_delay(4) == 16.0
        assert config.calculate_delay(5) == 32.0
        assert config.calculate_delay(6) == 64.0
        assert config.calculate_delay(7) == 100.0  # capped

    def test_calculate_delay_custom_base(self):
        config = RetryConfig(initial_delay=1.0, exponential_base=3.0, max_delay=100.0)
        
        assert config.calculate_delay(0) == 1.0
        assert config.calculate_delay(1) == 3.0
        assert config.calculate_delay(2) == 9.0
        assert config.calculate_delay(3) == 27.0
        assert config.calculate_delay(4) == 81.0
        assert config.calculate_delay(5) == 100.0  # capped


class TestRetryExhaustedError:
    """Test RetryExhaustedError class."""

    def test_error_properties(self):
        original_error = ValueError("Original error")
        error = RetryExhaustedError(original_error, 3)
        
        assert error.last_exception is original_error
        assert error.attempts == 3
        assert "Retry failed after 3 attempts" in str(error)
        assert "Original error" in str(error)


class TestAsyncRetry:
    """Test async_retry decorator."""

    @pytest.mark.asyncio
    async def test_successful_call(self):
        """Test successful function call without retry."""
        call_count = 0
        
        @async_retry(RetryConfig(max_retries=3))
        async def successful_func():
            nonlocal call_count
            call_count += 1
            return "success"
        
        result = await successful_func()
        assert result == "success"
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_retry_on_failure(self):
        """Test retry on failure."""
        call_count = 0
        
        @async_retry(RetryConfig(max_retries=3, initial_delay=0.01))
        async def failing_func():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError("Temporary failure")
            return "success"
        
        result = await failing_func()
        assert result == "success"
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_exhausted_retries(self):
        """Test exhausting all retries."""
        call_count = 0
        
        @async_retry(RetryConfig(max_retries=3, initial_delay=0.01))
        async def always_failing_func():
            nonlocal call_count
            call_count += 1
            raise ValueError("Permanent failure")
        
        with pytest.raises(RetryExhaustedError) as exc_info:
            await always_failing_func()
        
        assert exc_info.value.attempts == 4  # initial + 3 retries
        assert call_count == 4

    @pytest.mark.asyncio
    async def test_specific_exception_type(self):
        """Test retry only on specific exception types."""
        call_count = 0
        
        @async_retry(RetryConfig(max_retries=3, initial_delay=0.01, retryable_exceptions=(ValueError,)))
        async def selective_func():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError("Retryable")
            return "success"
        
        result = await selective_func()
        assert result == "success"

    @pytest.mark.asyncio
    async def test_non_retryable_exception(self):
        """Test non-retryable exceptions are not retried."""
        call_count = 0
        
        @async_retry(RetryConfig(max_retries=3, retryable_exceptions=(ValueError,)))
        async def non_retryable_func():
            nonlocal call_count
            call_count += 1
            raise TypeError("Not retryable")
        
        with pytest.raises(TypeError):
            await non_retryable_func()
        
        assert call_count == 1  # No retries

    @pytest.mark.asyncio
    async def test_on_retry_callback(self):
        """Test on_retry callback is called."""
        call_count = 0
        retry_attempts = []
        
        def on_retry(exc, attempt):
            nonlocal call_count
            call_count += 1
            retry_attempts.append(attempt)
        
        @async_retry(RetryConfig(max_retries=3, initial_delay=0.01), on_retry=on_retry)
        async def callback_func():
            if call_count < 3:
                raise ValueError("Retry")
            return "success"
        
        await callback_func()
        
        assert call_count == 3
        assert retry_attempts == [1, 2, 3]

    @pytest.mark.asyncio
    async def test_default_config(self):
        """Test using default config."""
        call_count = 0
        
        @async_retry()  # Uses default config
        async def default_config_func():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise ValueError("Retry")
            return "success"
        
        result = await default_config_func()
        assert result == "success"
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_sync_function_error(self):
        """Test that sync functions are handled correctly."""
        # This should work but decorated function becomes async
        call_count = 0
        
        @async_retry(RetryConfig(max_retries=1, initial_delay=0.01))
        async def sync_like_func():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise ValueError("Retry")
            return "success"
        
        result = await sync_like_func()
        assert result == "success"

    @pytest.mark.asyncio
    async def test_zero_max_retries(self):
        """Test with zero max_retries (no retry)."""
        call_count = 0
        
        @async_retry(RetryConfig(max_retries=0, initial_delay=0.01))
        async def no_retry_func():
            nonlocal call_count
            call_count += 1
            raise ValueError("Fail")
        
        with pytest.raises(RetryExhaustedError):
            await no_retry_func()
        
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_arguments_passed(self):
        """Test that arguments are passed correctly."""
        @async_retry(RetryConfig(max_retries=1, initial_delay=0.01))
        async def func_with_args(a, b, c=10):
            return f"{a}-{b}-{c}"
        
        result = await func_with_args("x", "y", c=20)
        assert result == "x-y-20"

    @pytest.mark.asyncio
    async def test_preserves_function_name(self):
        """Test that decorated function preserves its name."""
        @async_retry(RetryConfig(max_retries=1))
        async def my_function():
            return "success"
        
        assert my_function.__name__ == "my_function"
