"""Convention-based owner evaluation bundle loader.

The production task source is Cloudflare R2. The bucket layout is fixed by
convention:

    s3://${EIREL_EVAL_POOL_BUCKET}/${family_id}/pool-run-${run_id}.json

The publish side (eirel-eval-pool repo CI) writes to that key with an
operator-only WRITE token. Owner-api reads it with a separate READ-only
token (``EIREL_R2_*``). The convention removes the need for a registration
step in a DB table — there is no ``OwnerDatasetBinding`` indirection.

A small disk cache keyed by ETag (returned by the R2 HEAD response) keeps
repeated scrapes within a single run from re-fetching the bundle.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path

from shared.common.object_store import ObjectStore
from shared.core.evaluation_models import FamilyEvaluationBundle
from eirel.groups import ensure_family_id

from control_plane.owner_api._helpers import _enforce_strict_analyst_contract


_logger = logging.getLogger(__name__)


_DEFAULT_KEY_TEMPLATE = "{family_id}/pool-run-{run_id}.json"


class EvalPoolConfigError(RuntimeError):
    """Required ``EIREL_EVAL_POOL_BUCKET`` env var is unset."""


class EvalPoolFetchError(RuntimeError):
    """Bundle fetch from R2 failed (network, 404, parse, contract)."""


@dataclass(slots=True)
class LoaderResult:
    bundle: FamilyEvaluationBundle
    bundle_uri: str
    cache_hit: bool
    bytes_fetched: int


# -- public API ----------------------------------------------------------


def resolve_pool_uri(family_id: str, run_id: str) -> str:
    """Derive the R2 URI for ``(family_id, run_id)`` from the convention.

    Reads ``EIREL_EVAL_POOL_BUCKET`` (required, no default) and the optional
    ``EIREL_EVAL_POOL_KEY_TEMPLATE`` override (default
    ``{family_id}/pool-run-{run_id}.json``).
    """
    family_id = ensure_family_id(family_id)
    bucket = os.environ.get("EIREL_EVAL_POOL_BUCKET", "").strip()
    if not bucket:
        raise EvalPoolConfigError(
            "EIREL_EVAL_POOL_BUCKET must be set on owner-api to resolve eval bundles"
        )
    template = os.environ.get(
        "EIREL_EVAL_POOL_KEY_TEMPLATE", _DEFAULT_KEY_TEMPLATE
    )
    key = template.format(family_id=family_id, run_id=run_id)
    return f"s3://{bucket}/{key}"


def load_owner_evaluation_bundle(
    *,
    family_id: str,
    run_id: str,
    object_store: ObjectStore,
    cache_dir: str | None = None,
) -> LoaderResult:
    """Fetch + parse the bundle for ``(family_id, run_id)`` from R2.

    Raises ``EvalPoolConfigError`` if the bucket env var is unset.
    Raises ``EvalPoolFetchError`` for network / parse / contract errors.
    """
    family_id = ensure_family_id(family_id)
    bundle_uri = resolve_pool_uri(family_id, run_id)
    payload, cache_hit = _fetch_bundle_bytes(
        object_store=object_store,
        bundle_uri=bundle_uri,
        cache_dir=cache_dir,
    )
    try:
        bundle = FamilyEvaluationBundle.model_validate_json(payload)
    except Exception as exc:
        raise EvalPoolFetchError(
            f"failed to parse bundle from {bundle_uri}: {exc}"
        ) from exc
    if bundle.family_id != family_id:
        raise EvalPoolFetchError(
            f"bundle family_id mismatch at {bundle_uri}: "
            f"bundle declares {bundle.family_id!r}, expected {family_id!r}"
        )
    _enforce_strict_analyst_contract(bundle)
    return LoaderResult(
        bundle=bundle,
        bundle_uri=bundle_uri,
        cache_hit=cache_hit,
        bytes_fetched=len(payload),
    )


# -- internals -----------------------------------------------------------


def _fetch_bundle_bytes(
    *,
    object_store: ObjectStore,
    bundle_uri: str,
    cache_dir: str | None,
) -> tuple[bytes, bool]:
    cache_path = _cache_path(cache_dir, bundle_uri)
    if cache_path is not None and cache_path.exists():
        try:
            cached = cache_path.read_bytes()
        except OSError as exc:
            _logger.warning("failed to read bundle cache %s: %s", cache_path, exc)
        else:
            return cached, True
    try:
        payload = object_store.fetch_sync(bundle_uri)
    except Exception as exc:
        raise EvalPoolFetchError(
            f"failed to fetch bundle from {bundle_uri}: {exc}"
        ) from exc
    if cache_path is not None:
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_bytes(payload)
        except OSError as exc:
            _logger.warning("failed to write bundle cache %s: %s", cache_path, exc)
    return payload, False


def _cache_path(cache_dir: str | None, bundle_uri: str) -> Path | None:
    if not cache_dir:
        return None
    import hashlib
    digest = hashlib.sha256(bundle_uri.encode("utf-8")).hexdigest()
    return Path(cache_dir) / f"{digest}.json"
