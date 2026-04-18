from __future__ import annotations

"""OpenTelemetry tracing utilities for EIREL services.

Usage::

    from shared.common.tracing import init_tracing, get_tracer

    init_tracing("dag-executor")
    tracer = get_tracer(__name__)
    with tracer.start_as_current_span("execute_node") as span:
        span.set_attribute("node_id", node.node_id)
"""

import logging
import os
from typing import Any

_logger = logging.getLogger(__name__)

_TRACING_ENABLED = bool(os.getenv("EIREL_TRACING_ENABLED", ""))
_OTLP_ENDPOINT = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4317")

_tracer_provider: Any = None


class _NoOpSpan:
    """No-op span for when tracing is disabled."""

    def __enter__(self) -> _NoOpSpan:
        return self

    def __exit__(self, *args: Any) -> None:
        pass

    def set_attribute(self, key: str, value: Any) -> None:
        pass

    def set_status(self, status: Any) -> None:
        pass

    def add_event(self, name: str, attributes: dict[str, Any] | None = None) -> None:
        pass

    def record_exception(self, exception: BaseException) -> None:
        pass


class _NoOpTracer:
    """No-op tracer for when OTel is not installed or disabled."""

    def start_as_current_span(self, name: str, **kwargs: Any) -> _NoOpSpan:
        return _NoOpSpan()

    def start_span(self, name: str, **kwargs: Any) -> _NoOpSpan:
        return _NoOpSpan()


def init_tracing(service_name: str) -> None:
    """Initialize OpenTelemetry tracing for a service.

    No-op if EIREL_TRACING_ENABLED is not set or opentelemetry is not installed.
    """
    global _tracer_provider

    if not _TRACING_ENABLED:
        _logger.debug("Tracing disabled (EIREL_TRACING_ENABLED not set)")
        return

    try:
        from opentelemetry import trace
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace.export import BatchSpanProcessor

        resource = Resource.create({"service.name": service_name})
        provider = TracerProvider(resource=resource)

        try:
            from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
            exporter = OTLPSpanExporter(endpoint=_OTLP_ENDPOINT)
            provider.add_span_processor(BatchSpanProcessor(exporter))
        except ImportError:
            _logger.warning("OTLP gRPC exporter not available, using console exporter")
            from opentelemetry.sdk.trace.export import ConsoleSpanExporter
            provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))

        trace.set_tracer_provider(provider)
        _tracer_provider = provider
        _logger.info("Tracing initialized for service=%s endpoint=%s", service_name, _OTLP_ENDPOINT)
    except ImportError:
        _logger.debug("opentelemetry not installed, tracing disabled")


def get_tracer(name: str) -> Any:
    """Get a tracer instance.  Returns a no-op tracer if OTel is unavailable."""
    if not _TRACING_ENABLED:
        return _NoOpTracer()
    try:
        from opentelemetry import trace
        return trace.get_tracer(name)
    except ImportError:
        return _NoOpTracer()


def inject_trace_context(headers: dict[str, str]) -> dict[str, str]:
    """Inject the current trace context into HTTP headers."""
    if not _TRACING_ENABLED:
        return headers
    try:
        from opentelemetry.propagate import inject
        inject(headers)
    except ImportError:
        pass
    return headers


def extract_trace_context(headers: dict[str, str]) -> Any:
    """Extract trace context from incoming HTTP headers."""
    if not _TRACING_ENABLED:
        return None
    try:
        from opentelemetry.propagate import extract
        return extract(headers)
    except ImportError:
        return None
