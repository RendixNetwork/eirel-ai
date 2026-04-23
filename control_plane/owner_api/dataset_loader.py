"""Binding-aware owner evaluation bundle loader.

This module is the production path for fetching analyst evaluation bundles.
Instead of reading a flat JSON file from disk, it consults the
``OwnerDatasetBinding`` table for the active binding tied to ``(family_id,
run_id)``, fetches the bundle bytes from the recorded ``bundle_uri`` via
``ObjectStore``, verifies the SHA256 (and optionally the signature), parses
into a ``FamilyEvaluationBundle``, and re-enforces the analyst contract.

A small disk cache keyed by ``bundle_sha256`` keeps repeated scrapes within a
single run from re-fetching the bundle from S3.

This loader is fully synchronous because owner-api's ``RunManager`` holds a
sync SQLAlchemy session, and the underlying object_store IO (boto3 + file
read) is synchronous anyway. ``ObjectStore.fetch_sync`` exists specifically
to avoid the ``asyncio.run() inside running loop`` trap.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from sqlalchemy.orm import Session

from shared.common.models import OwnerDatasetBinding
from shared.common.object_store import ObjectStore
from shared.core.evaluation_models import FamilyEvaluationBundle
from eirel.groups import ensure_family_id

from control_plane.owner_api._helpers import _enforce_strict_analyst_contract


_logger = logging.getLogger(__name__)


class BindingNotFoundError(LookupError):
    """No active or pending OwnerDatasetBinding exists for the requested run."""


class BundleIntegrityError(ValueError):
    """The fetched bundle bytes did not match the binding's SHA256 / signature."""


@dataclass(slots=True)
class LoaderResult:
    bundle: FamilyEvaluationBundle
    binding: OwnerDatasetBinding
    cache_hit: bool
    bytes_fetched: int


class SignatureVerifier(Protocol):
    """Optional protocol for verifying bundle signatures.

    Production wires a real ``bittensor.Keypair`` here. Pass ``None`` to
    skip signature checks (sha256 integrity is always enforced).
    """

    def verify_bundle(
        self,
        canonical_bytes: bytes,
        *,
        signer_ss58: str,
        signature_hex: str,
    ) -> bool:
        ...


class KeypairSignatureVerifier:
    """SignatureVerifier backed by a :class:`Keypair`-shaped object.

    Accepts anything with ``ss58_address`` + ``verify(data, signature)``,
    so the production bittensor hotkey (or a compatible test double) can
    plug in directly.
    """

    __slots__ = ("_keypair",)

    def __init__(self, keypair: object) -> None:
        if not hasattr(keypair, "ss58_address") or not hasattr(keypair, "verify"):
            raise TypeError(
                "KeypairSignatureVerifier requires a keypair with ss58_address and verify()"
            )
        self._keypair = keypair

    @property
    def ss58_address(self) -> str:
        return str(self._keypair.ss58_address)  # type: ignore[attr-defined]

    def verify_bundle(
        self,
        canonical_bytes: bytes,
        *,
        signer_ss58: str,
        signature_hex: str,
    ) -> bool:
        if signer_ss58 != self.ss58_address:
            raise BundleIntegrityError(
                f"binding signer_ss58={signer_ss58!r} does not match owner "
                f"hotkey {self.ss58_address!r}"
            )
        try:
            signature = bytes.fromhex(signature_hex)
        except ValueError as exc:
            raise BundleIntegrityError(f"invalid hex signature: {exc}") from exc
        return bool(self._keypair.verify(canonical_bytes, signature))  # type: ignore[attr-defined]


# -- public API ----------------------------------------------------------


def load_owner_evaluation_bundle_via_binding(
    *,
    family_id: str,
    run_id: str,
    session: Session,
    object_store: ObjectStore,
    cache_dir: str | None = None,
    signature_verifier: SignatureVerifier | None = None,
) -> LoaderResult:
    """Look up the active binding and fetch + verify the bundle bytes.

    Raises ``BindingNotFoundError`` if no active or pending binding exists for
    ``(family_id, run_id)``. Raises ``BundleIntegrityError`` if the fetched
    bytes do not match the binding's recorded SHA256 (or signature, when a
    verifier is supplied). Raises ``ValueError`` from
    ``_enforce_strict_analyst_contract`` if the parsed bundle violates the
    analyst contract.
    """
    family_id = ensure_family_id(family_id)
    binding = _resolve_active_binding(session, family_id=family_id, run_id=run_id)
    if binding is None:
        raise BindingNotFoundError(
            f"no active OwnerDatasetBinding for family={family_id!r} run_id={run_id!r}"
        )

    payload, cache_hit = _fetch_bundle_bytes(
        object_store=object_store,
        bundle_uri=binding.bundle_uri,
        expected_sha256=binding.bundle_sha256,
        cache_dir=cache_dir,
    )

    actual_sha = hashlib.sha256(payload).hexdigest()
    if actual_sha != binding.bundle_sha256:
        raise BundleIntegrityError(
            f"bundle sha256 mismatch for binding={binding.id!r}: "
            f"expected={binding.bundle_sha256!r} actual={actual_sha!r}"
        )

    if signature_verifier is not None:
        try:
            verified = signature_verifier.verify_bundle(
                payload,
                signer_ss58=binding.generated_by,
                signature_hex=binding.signature_hex,
            )
        except Exception as exc:
            raise BundleIntegrityError(
                f"signature verification raised for binding={binding.id!r}: {exc}"
            ) from exc
        if not verified:
            raise BundleIntegrityError(
                f"signature verification failed for binding={binding.id!r}"
            )

    bundle = FamilyEvaluationBundle.model_validate_json(payload)
    if bundle.family_id != family_id:
        raise BundleIntegrityError(
            f"bundle family_id mismatch: bundle declares {bundle.family_id!r}, "
            f"binding claims {family_id!r}"
        )
    _enforce_strict_analyst_contract(bundle)

    return LoaderResult(
        bundle=bundle,
        binding=binding,
        cache_hit=cache_hit,
        bytes_fetched=len(payload),
    )


# -- internals -----------------------------------------------------------


def _resolve_active_binding(
    session: Session,
    *,
    family_id: str,
    run_id: str,
) -> OwnerDatasetBinding | None:
    """Look up the binding for ``(family_id, run_id)``.

    Prefers an ``active`` binding; falls back to a ``pending`` one so the
    register → activate → load flow works even if the operator forgets to
    flip the status. ``superseded`` and ``failed`` are never returned.
    """
    rows = (
        session.query(OwnerDatasetBinding)
        .filter_by(family_id=family_id, run_id=run_id)
        .all()
    )
    if not rows:
        return None
    by_status = {row.status: row for row in rows}
    return by_status.get("active") or by_status.get("pending")


def _fetch_bundle_bytes(
    *,
    object_store: ObjectStore,
    bundle_uri: str,
    expected_sha256: str,
    cache_dir: str | None,
) -> tuple[bytes, bool]:
    cache_path = _cache_path(cache_dir, expected_sha256)
    if cache_path is not None and cache_path.exists():
        try:
            cached = cache_path.read_bytes()
        except OSError as exc:
            _logger.warning("failed to read bundle cache %s: %s", cache_path, exc)
        else:
            return cached, True
    payload = object_store.fetch_sync(bundle_uri)
    if cache_path is not None:
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_bytes(payload)
        except OSError as exc:
            _logger.warning("failed to write bundle cache %s: %s", cache_path, exc)
    return payload, False


def _cache_path(cache_dir: str | None, sha256_hex: str) -> Path | None:
    if not cache_dir:
        return None
    return Path(cache_dir) / f"{sha256_hex}.json"
