"""FastAPI application factory."""
from __future__ import annotations

import time
import uuid
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.config import Settings, get_settings
from app.dependencies import ServiceContainer
from app.errors import ApiError, error_body
from app.routes import router
from src.ingestion.base import DocumentLoadError
from src.llm.base import LLMError
from src.utils.logging import configure_logging, get_logger
from src.vectorstore.faiss_store import VectorStoreError

logger = get_logger("intellirag.api")


def create_app(settings: Optional[Settings] = None, container: Optional[ServiceContainer] = None) -> FastAPI:
    """Build the FastAPI app. Pass ``container`` to inject fakes (tests) or custom services."""
    settings = settings or get_settings()
    container = container or ServiceContainer(settings)
    configure_logging(settings.log_level, settings.logs_dir)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        settings.ensure_directories()
        container.metrics.set_model_versions(container.model_versions())
        logger.info("startup", extra={"versions": container.model_versions()})
        yield
        logger.info("shutdown")

    app = FastAPI(
        title=f"{settings.app_name} - Neural Document Intelligence & RAG",
        version=settings.app_version,
        description="Document ingestion, semantic retrieval and grounded question answering.",
        lifespan=lifespan,
    )
    app.state.container = container

    @app.middleware("http")
    async def observe(request: Request, call_next):
        request_id = request.headers.get("x-request-id") or uuid.uuid4().hex
        request.state.request_id = request_id
        started = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception as exc:  # unhandled -> structured 500
            logger.exception("unhandled_error", extra={"request_id": request_id, "path": request.url.path})
            container.metrics.record_error("unhandled_exception", f"{exc.__class__.__name__}: {exc}", request.url.path)
            response = JSONResponse(
                status_code=500,
                content=error_body("internal_error", "An unexpected error occurred.", request_id=request_id),
            )
        latency_ms = (time.perf_counter() - started) * 1000
        container.metrics.record_request(request.url.path, request.method, response.status_code, latency_ms)
        container.events.log(
            "request", request_id=request_id, method=request.method, path=request.url.path,
            status=response.status_code, latency_ms=round(latency_ms, 2),
        )
        response.headers["X-Request-ID"] = request_id
        return response

    def _respond(request: Request, status: int, code: str, message: str, details=None) -> JSONResponse:
        container.metrics.record_error(code, message, request.url.path)
        body = error_body(code, message, details, getattr(request.state, "request_id", None))
        return JSONResponse(status_code=status, content=jsonable_encoder(body))

    @app.exception_handler(ApiError)
    async def _api_error(request: Request, exc: ApiError):
        return _respond(request, exc.status_code, exc.code, exc.message, exc.details)

    @app.exception_handler(DocumentLoadError)
    async def _doc_error(request: Request, exc: DocumentLoadError):
        return _respond(request, 422, exc.code, exc.reason, {"path": exc.path})

    @app.exception_handler(LLMError)
    async def _llm_error(request: Request, exc: LLMError):
        return _respond(request, 502, "llm_error", str(exc))

    @app.exception_handler(VectorStoreError)
    async def _store_error(request: Request, exc: VectorStoreError):
        return _respond(request, 409, "vector_store_error", str(exc))

    @app.exception_handler(RequestValidationError)
    async def _validation_error(request: Request, exc: RequestValidationError):
        return _respond(request, 422, "validation_error", "Request validation failed.", exc.errors())

    @app.exception_handler(StarletteHTTPException)
    async def _http_error(request: Request, exc: StarletteHTTPException):
        return _respond(request, exc.status_code, "http_error", str(exc.detail))

    app.include_router(router)
    return app


app = create_app()
