"""Optional OpenTelemetry tracing.

Disabled by default. Enable with OTEL_ENABLED=true and point OTEL_ENDPOINT
at an OTLP receiver. Requires the ``otel`` extra:
pip install -e ".[otel]"
"""

from app.core.config import Settings
from app.core.logging import get_logger

logger = get_logger(__name__)


def setup_tracing(settings: Settings, app) -> None:
    """Instrument the FastAPI app if tracing is enabled and available."""

    if not settings.otel_enabled:
        return
    try:
        from opentelemetry import trace
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
    except ImportError:
        logger.warning(
            "OTEL_ENABLED is true but OpenTelemetry packages are missing. "
            'Install them with: pip install -e ".[otel]"'
        )
        return

    resource = Resource.create(
        attributes={"service.name": settings.app_name, "deployment.environment": settings.app_env}
    )
    provider = TracerProvider(resource=resource)
    if settings.otel_endpoint:
        provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=settings.otel_endpoint)))
    trace.set_tracer_provider(provider)
    FastAPIInstrumentor.instrument_app(app, tracer_provider=provider)
    logger.info(
        "OpenTelemetry tracing enabled",
        extra={"endpoint": settings.otel_endpoint or "none"},
    )
