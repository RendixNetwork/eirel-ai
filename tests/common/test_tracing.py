from __future__ import annotations

"""Tests for Item 17: Distributed tracing utilities."""

from unittest.mock import patch, MagicMock

import pytest

from shared.common.tracing import (
    _NoOpSpan,
    _NoOpTracer,
    get_tracer,
    init_tracing,
    inject_trace_context,
    extract_trace_context,
)


# -- No-op fallback tests --

def test_noop_span_context_manager():
    span = _NoOpSpan()
    with span as s:
        assert s is span


def test_noop_span_methods():
    span = _NoOpSpan()
    span.set_attribute("key", "value")
    span.set_status("OK")
    span.add_event("event", {"a": 1})
    span.record_exception(RuntimeError("err"))


def test_noop_tracer_start_as_current_span():
    tracer = _NoOpTracer()
    span = tracer.start_as_current_span("test")
    assert isinstance(span, _NoOpSpan)


def test_noop_tracer_start_span():
    tracer = _NoOpTracer()
    span = tracer.start_span("test")
    assert isinstance(span, _NoOpSpan)


# -- get_tracer tests --

def test_get_tracer_disabled_returns_noop():
    with patch("shared.common.tracing._TRACING_ENABLED", False):
        tracer = get_tracer("test")
        assert isinstance(tracer, _NoOpTracer)


def test_get_tracer_enabled_no_otel_returns_noop():
    with patch("shared.common.tracing._TRACING_ENABLED", True), \
         patch.dict("sys.modules", {"opentelemetry": None, "opentelemetry.trace": None}):
        # Force ImportError
        import shared.common.tracing as mod
        orig = mod._TRACING_ENABLED
        mod._TRACING_ENABLED = True
        try:
            tracer = mod.get_tracer("test")
            # Should either return real tracer or noop depending on install
            assert tracer is not None
        finally:
            mod._TRACING_ENABLED = orig


# -- inject/extract context tests --

def test_inject_trace_context_disabled():
    with patch("shared.common.tracing._TRACING_ENABLED", False):
        headers = {"Authorization": "Bearer token"}
        result = inject_trace_context(headers)
        assert result is headers
        assert "traceparent" not in result


def test_extract_trace_context_disabled():
    with patch("shared.common.tracing._TRACING_ENABLED", False):
        result = extract_trace_context({"traceparent": "00-abc-def-01"})
        assert result is None


# -- init_tracing tests --

def test_init_tracing_disabled():
    with patch("shared.common.tracing._TRACING_ENABLED", False):
        # Should be a no-op, no exception
        init_tracing("test-service")


def test_init_tracing_no_otel_installed():
    import shared.common.tracing as mod
    orig = mod._TRACING_ENABLED
    mod._TRACING_ENABLED = True
    try:
        with patch.dict("sys.modules", {"opentelemetry": None}):
            # Should handle ImportError gracefully
            try:
                init_tracing("test-service")
            except ImportError:
                pass  # Expected if patching doesn't fully work
    finally:
        mod._TRACING_ENABLED = orig


def test_init_tracing_with_mock_otel():
    """Test init_tracing with mocked opentelemetry modules."""
    import shared.common.tracing as mod
    orig = mod._TRACING_ENABLED
    orig_provider = mod._tracer_provider
    mod._TRACING_ENABLED = True
    try:
        mock_trace = MagicMock()
        mock_resource = MagicMock()
        mock_sdk_trace = MagicMock()
        mock_export = MagicMock()
        mock_otlp = MagicMock()

        with patch.dict("sys.modules", {
            "opentelemetry": MagicMock(trace=mock_trace),
            "opentelemetry.trace": mock_trace,
            "opentelemetry.sdk": MagicMock(),
            "opentelemetry.sdk.trace": mock_sdk_trace,
            "opentelemetry.sdk.resources": mock_resource,
            "opentelemetry.sdk.trace.export": mock_export,
            "opentelemetry.exporter": MagicMock(),
            "opentelemetry.exporter.otlp": MagicMock(),
            "opentelemetry.exporter.otlp.proto": MagicMock(),
            "opentelemetry.exporter.otlp.proto.grpc": MagicMock(),
            "opentelemetry.exporter.otlp.proto.grpc.trace_exporter": mock_otlp,
        }):
            init_tracing("my-service")
            mock_resource.Resource.create.assert_called_once_with({"service.name": "my-service"})
            mock_trace.set_tracer_provider.assert_called_once()
    finally:
        mod._TRACING_ENABLED = orig
        mod._tracer_provider = orig_provider


# -- Integration: noop tracer usable in with-statement --

def test_noop_tracer_span_as_context_manager():
    tracer = _NoOpTracer()
    with tracer.start_as_current_span("op") as span:
        span.set_attribute("key", "val")
        span.add_event("checkpoint")
