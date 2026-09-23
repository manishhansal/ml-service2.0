"""
Explainability endpoint for ml-service2.0.

Endpoint:
    POST /v2/explain/{model_name}

Returns SHAP-based (or rule-based for HEURISTIC) feature attributions for a
given prediction.  The endpoint never returns HTTP 500 due to a SHAP failure;
instead it returns an ``ExplainResponse`` with all contributions set to 0.0
(Req 11.6).

Authentication:
    Requires ``X-API-KEY`` header (enforced by ``api_key_middleware`` in
    ``src/main.py``).
"""
from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from src.explainability.explainer import ModelExplainer
from src.logging_config import get_logger
from src.schemas.base import PredictionProvenance
from src.schemas.predictions import ExplainRequest, ExplainResponse

logger = get_logger(__name__)

router = APIRouter(tags=["explainability"])

# Module-level singleton — shared across all requests.
# Models must be registered via explainer.register_model() after model artifacts
# are loaded (typically in the FastAPI lifespan startup handler).
explainer = ModelExplainer()


@router.post(
    "/explain/{model_name}",
    response_model=ExplainResponse,
    summary="Get SHAP feature attributions for a prediction",
    description=(
        "Returns top-10 SHAP feature contributions for the specified model "
        "and prediction.  For HEURISTIC predictions rule attribution is used "
        "instead of SHAP.  Never returns HTTP 500 — SHAP failures yield "
        "contributions=[] with base_value=historical_mean."
    ),
)
async def explain_prediction(
    model_name: str,
    request_body: ExplainRequest,
    request: Request,
) -> ExplainResponse:
    """POST /v2/explain/{model_name}

    Args:
        model_name:    Model identifier (path parameter).  Must match a
                       registered model name
                       (``regime`` | ``ranker`` | ``strategy`` | ``risk`` |
                       ``rl_agent``).
        request_body:  :class:`~src.schemas.predictions.ExplainRequest` with
                       ``features`` dict and ``prediction`` string.
        request:       FastAPI ``Request`` — used to access ``app.state``.

    Returns:
        :class:`~src.schemas.predictions.ExplainResponse` with SHAP or
        rule-based contributions.
    """
    # Prefer provenance carried by the request; default to TRAINED_MODEL so
    # that SHAP is attempted when no explicit provenance is provided.
    # ExplainRequest does not carry a provenance field in the current schema,
    # so we always attempt SHAP and let the explainer degrade gracefully.
    logger.info(
        "explain_request_received",
        model_name=model_name,
        n_features=len(request_body.features),
        prediction=request_body.prediction,
    )

    response = explainer.explain(
        model_name=model_name,
        features=request_body.features,
        prediction=request_body.prediction,
        top_k=10,
        provenance=PredictionProvenance.TRAINED_MODEL,
    )

    logger.info(
        "explain_response_generated",
        model_name=model_name,
        n_contributions=len(response.contributions),
        top_drivers=response.top_drivers[:3],
    )
    return response


@router.post(
    "/explain/{model_name}/heuristic",
    response_model=ExplainResponse,
    summary="Get rule-based attributions for a HEURISTIC prediction",
    description=(
        "Generates magnitude-based rule attributions when the model is "
        "operating in heuristic fallback mode (no trained artifact loaded)."
    ),
)
async def explain_heuristic_prediction(
    model_name: str,
    request_body: ExplainRequest,
    request: Request,
) -> ExplainResponse:
    """POST /v2/explain/{model_name}/heuristic

    Forces HEURISTIC provenance, bypassing SHAP entirely and returning
    magnitude-ranked rule attributions.
    """
    logger.info(
        "explain_heuristic_request_received",
        model_name=model_name,
        n_features=len(request_body.features),
        prediction=request_body.prediction,
    )

    response = explainer.explain(
        model_name=model_name,
        features=request_body.features,
        prediction=request_body.prediction,
        top_k=10,
        provenance=PredictionProvenance.HEURISTIC,
    )
    return response
