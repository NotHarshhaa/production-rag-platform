"""FastAPI application factory."""

import time
import uuid
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from threading import Lock

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app import __version__
from app.api.dependencies import build_container
from app.api.routes import documents, health, query
from app.core.config import Settings, get_settings
from app.core.logging import configure_logging, get_logger, request_id_var
from app.observability.metrics import (
    IN_PROGRESS,
    REQUEST_LATENCY,
    REQUESTS_TOTAL,
    metrics_router,
)
from app.observability.tracing import setup_tracing

logger = get_logger(__name__)


class RateLimiter:
    """Fixed-window per-client-IP limiter, applied when a limit is configured."""

    def __init__(self, limit_per_minute: int) -> None:
        self.limit = limit_per_minute
        self._hits: dict[str, deque] = defaultdict(lambda: deque(maxlen=limit_per_minute))
        self._lock = Lock()

    def allow(self, client_ip: str) -> bool:
        if self.limit <= 0:
            return True
        window_start = time.time() - 60.0
        with self._lock:
            hits = self._hits[client_ip]
            while hits and hits[0] < window_start:
                hits.popleft()
            if len(hits) >= self.limit:
                return False
            hits.append(time.time())
            return True


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    configure_logging(settings.log_level, settings.log_json)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        container = build_container(settings)
        app.state.container = container
        logger.info(
            "RAG platform started",
            extra={
                "environment": settings.app_env,
                "vector_store": settings.vector_store,
                "embedding_provider": settings.embedding_provider,
                "llm_provider": settings.llm_provider,
            },
        )
        yield
        container.shutdown()
        logger.info("RAG platform stopped")

    app = FastAPI(
        title=settings.app_name,
        version=__version__,
        description=(
            "Production-grade Retrieval-Augmented Generation: document ingestion, "
            "hybrid retrieval, grounded answers with citations, evaluation, and metrics."
        ),
        lifespan=lifespan,
    )

    # --- Middleware -------------------------------------------------------
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origin_list,
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    limiter = RateLimiter(settings.rate_limit_per_minute)

    @app.middleware("http")
    async def request_context(request: Request, call_next):
        request_id = uuid.uuid4().hex[:12]
        token = request_id_var.set(request_id)
        client_ip = request.client.host if request.client else "unknown"
        try:
            if not limiter.allow(client_ip):
                return JSONResponse(
                    status_code=429,
                    content={"detail": "Rate limit exceeded. Try again later."},
                    headers={"X-Request-ID": request_id, "Retry-After": "60"},
                )
            response = await call_next(request)
            response.headers["X-Request-ID"] = request_id
            return response
        finally:
            request_id_var.reset(token)

    @app.middleware("http")
    async def metrics_middleware(request: Request, call_next):
        if not settings.metrics_enabled or request.url.path == "/metrics":
            return await call_next(request)
        method = request.method
        started = time.perf_counter()
        IN_PROGRESS.inc()
        try:
            response = await call_next(request)
            status_code = str(response.status_code)
            return response
        except Exception:
            status_code = "500"
            raise
        finally:
            IN_PROGRESS.dec()
            route = request.scope.get("route")
            route_template = route.path if route is not None else request.url.path
            REQUESTS_TOTAL.labels(method=method, route=route_template, status=status_code).inc()
            REQUEST_LATENCY.labels(method=method, route=route_template).observe(
                time.perf_counter() - started
            )

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception):
        logger.exception(
            "Unhandled error",
            extra={"path": request.url.path, "error_type": type(exc).__name__},
        )
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error"},
        )

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        return JSONResponse(
            status_code=422,
            content={"detail": exc.errors()},
        )

    # --- Routes -----------------------------------------------------------
    app.include_router(health.router)
    app.include_router(documents.router)
    app.include_router(query.router)
    if settings.metrics_enabled:
        app.include_router(metrics_router)

    setup_tracing(settings, app)
    return app


app = create_app()
