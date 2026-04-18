from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, field_validator

from eirel.groups import ensure_family_id


def _default_corpus_root() -> Path:
    env = os.getenv("EIREL_WORKFLOW_CORPUS_ROOT_PATH")
    if env:
        return Path(env).expanduser().resolve()
    return (Path(__file__).resolve().parents[2] / "data" / "workflow_corpus").resolve()


class WorkflowCorpusSlice(BaseModel):
    slice_id: str
    task_prompt: str
    initial_context: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


class WorkflowCorpusFile(BaseModel):
    workflow_spec_id: str
    workflow_version: str
    visibility: str
    slices: list[WorkflowCorpusSlice] = Field(default_factory=list)


class WorkflowReplayBaseline(BaseModel):
    node_id: str
    family_id: str
    role_id: str
    baseline_id: str | None = None
    selector: dict[str, Any] = Field(default_factory=dict)
    output: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("family_id", mode="before")
    @classmethod
    def normalize_family_id(cls, value: Any) -> str:
        return ensure_family_id(str(value))


class WorkflowReplayBaselineFile(BaseModel):
    workflow_spec_id: str
    workflow_version: str
    baselines: list[WorkflowReplayBaseline] = Field(default_factory=list)


class WorkflowCorpusArtifactRef(BaseModel):
    path: str
    sha256: str


class WorkflowCorpusManifestEntry(BaseModel):
    workflow_spec_id: str
    workflow_version: str
    required_workflow_version: str
    public_file: WorkflowCorpusArtifactRef
    hidden_file: WorkflowCorpusArtifactRef
    baseline_file: WorkflowCorpusArtifactRef | None = None
    replay_enabled: bool = True


class WorkflowCorpusManifest(BaseModel):
    corpus_version: str
    credit_assignment_version: str
    workflows: list[WorkflowCorpusManifestEntry] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class WorkflowCorpusWorkflowSummary(BaseModel):
    workflow_spec_id: str
    workflow_version: str
    required_workflow_version: str
    public_slice_count: int = 0
    hidden_slice_count: int = 0
    baseline_count: int = 0
    replay_enabled: bool = True
    supported_modalities: list[str] = Field(default_factory=list)


class WorkflowCorpusReport(BaseModel):
    valid: bool
    corpus_root: str
    manifest_path: str
    manifest_digest: str | None = None
    corpus_version: str | None = None
    credit_assignment_version: str | None = None
    workflow_spec_count: int = 0
    public_slice_count: int = 0
    hidden_slice_count: int = 0
    baseline_count: int = 0
    public_workflow_specs: list[str] = Field(default_factory=list)
    workflow_summaries: list[WorkflowCorpusWorkflowSummary] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    manifest: WorkflowCorpusManifest | None = Field(default=None, exclude=True)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _manifest_path(root: Path) -> Path:
    return root / "manifest.json"


def _resolve_relative_path(*, root: Path, relative_path: str) -> Path:
    path = (root / relative_path).resolve()
    if root not in path.parents and path != root:
        raise ValueError(f"corpus artifact path escapes corpus root: {relative_path}")
    return path


def _load_manifest(*, root: Path) -> tuple[WorkflowCorpusManifest | None, str | None, list[str]]:
    errors: list[str] = []
    manifest_path = _manifest_path(root)
    if not manifest_path.is_file():
        return None, None, [f"workflow corpus manifest not found: {manifest_path}"]
    try:
        raw_bytes = manifest_path.read_bytes()
        payload = json.loads(raw_bytes.decode("utf-8"))
        manifest = WorkflowCorpusManifest.model_validate(payload)
    except Exception as exc:
        return None, None, [f"invalid workflow corpus manifest schema: {exc}"]
    return manifest, _sha256_bytes(raw_bytes), errors


def _load_validated_json_file(
    *,
    root: Path,
    artifact: WorkflowCorpusArtifactRef,
    label: str,
) -> tuple[dict[str, Any] | None, list[str]]:
    errors: list[str] = []
    try:
        path = _resolve_relative_path(root=root, relative_path=artifact.path)
    except ValueError as exc:
        return None, [str(exc)]
    if not path.is_file():
        return None, [f"{label} file not found: {path}"]
    raw_bytes = path.read_bytes()
    digest = _sha256_bytes(raw_bytes)
    if digest != artifact.sha256:
        return None, [
            f"{label} digest mismatch for {artifact.path}: expected {artifact.sha256}, got {digest}"
        ]
    try:
        return json.loads(raw_bytes.decode("utf-8")), errors
    except Exception as exc:
        return None, [f"{label} is not valid JSON ({artifact.path}): {exc}"]


def workflow_corpus_report(
    *,
    corpus_root: str | Path | None = None,
) -> WorkflowCorpusReport:
    from shared.workflow_specs.catalog import get_workflow_spec

    root = Path(corpus_root).resolve() if corpus_root is not None else _default_corpus_root()
    manifest, manifest_digest, errors = _load_manifest(root=root)
    report = WorkflowCorpusReport(
        valid=False,
        corpus_root=str(root),
        manifest_path=str(_manifest_path(root)),
        manifest_digest=manifest_digest,
        manifest=manifest,
    )
    if manifest is None:
        report.errors = errors
        return report

    report.corpus_version = manifest.corpus_version
    report.credit_assignment_version = manifest.credit_assignment_version
    seen_workflow_ids: set[str] = set()

    for entry in manifest.workflows:
        if entry.workflow_spec_id in seen_workflow_ids:
            errors.append(f"duplicate workflow spec in manifest: {entry.workflow_spec_id}")
            continue
        seen_workflow_ids.add(entry.workflow_spec_id)

        try:
            workflow_spec = get_workflow_spec(entry.workflow_spec_id)
        except KeyError:
            errors.append(f"manifest references unknown workflow spec: {entry.workflow_spec_id}")
            continue

        if entry.workflow_version != workflow_spec.workflow_version:
            errors.append(
                f"manifest workflow version mismatch for {entry.workflow_spec_id}: "
                f"expected {workflow_spec.workflow_version}, got {entry.workflow_version}"
            )
        if entry.required_workflow_version != workflow_spec.workflow_version:
            errors.append(
                f"manifest required workflow version mismatch for {entry.workflow_spec_id}: "
                f"expected {workflow_spec.workflow_version}, got {entry.required_workflow_version}"
            )

        summary = WorkflowCorpusWorkflowSummary(
            workflow_spec_id=entry.workflow_spec_id,
            workflow_version=entry.workflow_version,
            required_workflow_version=entry.required_workflow_version,
            replay_enabled=entry.replay_enabled,
        )
        slice_modalities: set[str] = set()

        for visibility, artifact in (("public", entry.public_file), ("hidden", entry.hidden_file)):
            payload, artifact_errors = _load_validated_json_file(
                root=root,
                artifact=artifact,
                label=f"{entry.workflow_spec_id} {visibility} corpus",
            )
            errors.extend(artifact_errors)
            if payload is None:
                continue
            try:
                corpus_file = WorkflowCorpusFile.model_validate(payload)
            except Exception as exc:
                errors.append(
                    f"{entry.workflow_spec_id} {visibility} corpus schema invalid: {exc}"
                )
                continue
            if corpus_file.workflow_spec_id != entry.workflow_spec_id:
                errors.append(
                    f"{entry.workflow_spec_id} {visibility} corpus workflow spec mismatch: "
                    f"got {corpus_file.workflow_spec_id}"
                )
            if corpus_file.workflow_version != entry.workflow_version:
                errors.append(
                    f"{entry.workflow_spec_id} {visibility} corpus workflow version mismatch: "
                    f"got {corpus_file.workflow_version}"
                )
            if corpus_file.visibility != visibility:
                errors.append(
                    f"{entry.workflow_spec_id} {visibility} corpus visibility mismatch: "
                    f"got {corpus_file.visibility}"
                )
            slice_ids = [item.slice_id for item in corpus_file.slices]
            if len(slice_ids) != len(set(slice_ids)):
                errors.append(
                    f"{entry.workflow_spec_id} {visibility} corpus has duplicate slice_id values"
                )
            for item in corpus_file.slices:
                modality = str(item.metadata.get("output_modality") or "").strip()
                if modality:
                    slice_modalities.add(modality)
            if visibility == "public":
                summary.public_slice_count = len(corpus_file.slices)
                report.public_slice_count += len(corpus_file.slices)
            else:
                summary.hidden_slice_count = len(corpus_file.slices)
                report.hidden_slice_count += len(corpus_file.slices)

        if entry.baseline_file is None:
            if entry.replay_enabled:
                errors.append(f"{entry.workflow_spec_id} replay is enabled but baseline_file is missing")
        else:
            payload, artifact_errors = _load_validated_json_file(
                root=root,
                artifact=entry.baseline_file,
                label=f"{entry.workflow_spec_id} baseline corpus",
            )
            errors.extend(artifact_errors)
            if payload is not None:
                try:
                    baseline_file = WorkflowReplayBaselineFile.model_validate(payload)
                except Exception as exc:
                    errors.append(
                        f"{entry.workflow_spec_id} baseline corpus schema invalid: {exc}"
                    )
                else:
                    if baseline_file.workflow_spec_id != entry.workflow_spec_id:
                        errors.append(
                            f"{entry.workflow_spec_id} baseline workflow spec mismatch: "
                            f"got {baseline_file.workflow_spec_id}"
                        )
                    if baseline_file.workflow_version != entry.workflow_version:
                        errors.append(
                            f"{entry.workflow_spec_id} baseline workflow version mismatch: "
                            f"got {baseline_file.workflow_version}"
                        )
                    summary.baseline_count = len(baseline_file.baselines)
                    report.baseline_count += len(baseline_file.baselines)
                    baseline_ids = [
                        item.baseline_id or f"{entry.workflow_spec_id}:{item.node_id}"
                        for item in baseline_file.baselines
                    ]
                    if len(baseline_ids) != len(set(baseline_ids)):
                        errors.append(
                            f"{entry.workflow_spec_id} baseline corpus has duplicate baseline identifiers"
                        )
                    if entry.replay_enabled:
                        expected_node_ids = {node.node_id for node in workflow_spec.nodes}
                        for node_id in sorted(expected_node_ids):
                            matching = [item for item in baseline_file.baselines if item.node_id == node_id]
                            if not matching:
                                errors.append(
                                    f"{entry.workflow_spec_id} baseline corpus missing replay coverage for node: {node_id}"
                                )
                                continue
                            if slice_modalities:
                                for modality in sorted(slice_modalities):
                                    if not any(_selector_matches(item.selector, {"output_modality": modality}) for item in matching):
                                        errors.append(
                                            f"{entry.workflow_spec_id} baseline corpus missing replay coverage for node {node_id} modality {modality}"
                                        )
        summary.supported_modalities = sorted(slice_modalities)

        report.workflow_summaries.append(summary)

    report.workflow_spec_count = len(report.workflow_summaries)
    report.public_workflow_specs = [
        item.workflow_spec_id for item in report.workflow_summaries if item.public_slice_count > 0
    ]
    report.errors = errors
    report.valid = not errors
    return report


def validate_workflow_corpus(
    *,
    corpus_root: str | Path | None = None,
) -> WorkflowCorpusReport:
    report = workflow_corpus_report(corpus_root=corpus_root)
    if not report.valid:
        raise ValueError("invalid workflow corpus: " + "; ".join(report.errors))
    return report


def load_workflow_corpus_manifest(
    *,
    corpus_root: str | Path | None = None,
) -> WorkflowCorpusManifest:
    return validate_workflow_corpus(corpus_root=corpus_root).manifest or WorkflowCorpusManifest(
        corpus_version="",
        credit_assignment_version="",
    )


def _manifest_entry_for_workflow(
    manifest: WorkflowCorpusManifest,
    *,
    workflow_spec_id: str,
) -> WorkflowCorpusManifestEntry:
    for entry in manifest.workflows:
        if entry.workflow_spec_id == workflow_spec_id:
            return entry
    raise FileNotFoundError(f"workflow corpus manifest does not declare workflow spec: {workflow_spec_id}")


def load_workflow_corpus(
    workflow_spec_id: str,
    *,
    visibility: str,
    corpus_root: str | Path | None = None,
) -> WorkflowCorpusFile:
    report = validate_workflow_corpus(corpus_root=corpus_root)
    manifest = report.manifest
    if manifest is None:
        raise ValueError("workflow corpus manifest could not be loaded")
    root = Path(report.corpus_root)
    entry = _manifest_entry_for_workflow(manifest, workflow_spec_id=workflow_spec_id)
    artifact = entry.hidden_file if visibility == "hidden" else entry.public_file
    payload, errors = _load_validated_json_file(
        root=root,
        artifact=artifact,
        label=f"{workflow_spec_id} {visibility} corpus",
    )
    if errors:
        raise ValueError("invalid workflow corpus: " + "; ".join(errors))
    if payload is None:
        raise FileNotFoundError(f"workflow corpus file not found for {workflow_spec_id} {visibility}")
    corpus_file = WorkflowCorpusFile.model_validate(payload)
    if corpus_file.visibility != visibility:
        raise ValueError(
            f"invalid workflow corpus: {workflow_spec_id} expected visibility {visibility}, got {corpus_file.visibility}"
        )
    return corpus_file


def load_workflow_baselines(
    workflow_spec_id: str,
    *,
    selectors: dict[str, Any] | None = None,
    corpus_root: str | Path | None = None,
) -> dict[str, WorkflowReplayBaseline]:
    report = validate_workflow_corpus(corpus_root=corpus_root)
    manifest = report.manifest
    if manifest is None:
        raise ValueError("workflow corpus manifest could not be loaded")
    root = Path(report.corpus_root)
    entry = _manifest_entry_for_workflow(manifest, workflow_spec_id=workflow_spec_id)
    if entry.baseline_file is None:
        return {}
    payload, errors = _load_validated_json_file(
        root=root,
        artifact=entry.baseline_file,
        label=f"{workflow_spec_id} baseline corpus",
    )
    if errors:
        raise ValueError("invalid workflow corpus: " + "; ".join(errors))
    if payload is None:
        raise FileNotFoundError(f"workflow baseline file not found for {workflow_spec_id}")
    baseline_file = WorkflowReplayBaselineFile.model_validate(payload)
    baselines: dict[str, WorkflowReplayBaseline] = {}
    requested_selector = dict(selectors or {})
    baselines_by_node: dict[str, list[WorkflowReplayBaseline]] = {}
    for item in baseline_file.baselines:
        if item.baseline_id is None:
            suffix = item.node_id
            if item.selector:
                selector_suffix = "-".join(
                    f"{key}={item.selector[key]}"
                    for key in sorted(item.selector)
                    if str(item.selector[key]).strip()
                )
                if selector_suffix:
                    suffix = f"{suffix}:{selector_suffix}"
            item.baseline_id = f"{workflow_spec_id}:{suffix}"
        baselines_by_node.setdefault(item.node_id, []).append(item)
    for node_id, candidates in baselines_by_node.items():
        baselines[node_id] = _resolve_baseline_for_selector(
            workflow_spec_id=workflow_spec_id,
            node_id=node_id,
            selectors=requested_selector,
            candidates=candidates,
        )
    return baselines


def workflow_corpus_public_metadata(
    *,
    corpus_root: str | Path | None = None,
) -> dict[str, Any]:
    report = workflow_corpus_report(corpus_root=corpus_root)
    return {
        "valid": report.valid,
        "corpus_version": report.corpus_version,
        "credit_assignment_version": report.credit_assignment_version,
        "manifest_digest": report.manifest_digest,
        "workflow_spec_count": report.workflow_spec_count,
        "public_slice_count": report.public_slice_count,
        "errors": list(report.errors),
        "workflow_specs": [
            {
                "workflow_spec_id": item.workflow_spec_id,
                "workflow_version": item.workflow_version,
                "required_workflow_version": item.required_workflow_version,
                "public_slice_count": item.public_slice_count,
                "replay_enabled": item.replay_enabled,
                "supported_modalities": list(item.supported_modalities or []),
            }
            for item in report.workflow_summaries
        ],
    }


def _selector_matches(selector: dict[str, Any], requested: dict[str, Any]) -> bool:
    selector = dict(selector or {})
    requested = dict(requested or {})
    for key, value in selector.items():
        if requested.get(key) != value:
            return False
    return True


def _selector_specificity(selector: dict[str, Any]) -> int:
    return len([key for key, value in dict(selector or {}).items() if value is not None and str(value) != ""])


def _resolve_baseline_for_selector(
    *,
    workflow_spec_id: str,
    node_id: str,
    selectors: dict[str, Any],
    candidates: list[WorkflowReplayBaseline],
) -> WorkflowReplayBaseline:
    matches = [item for item in candidates if _selector_matches(item.selector, selectors)]
    if not matches:
        raise ValueError(
            f"invalid workflow corpus: no replay baseline matches selectors for {workflow_spec_id} node {node_id}"
        )
    matches.sort(
        key=lambda item: (
            _selector_specificity(item.selector),
            str(item.baseline_id or ""),
        ),
        reverse=True,
    )
    selected = matches[0]
    top_specificity = _selector_specificity(selected.selector)
    tied = [
        item
        for item in matches
        if _selector_specificity(item.selector) == top_specificity
    ]
    if len(tied) > 1:
        tied_signatures = {
            json.dumps(dict(item.selector or {}), sort_keys=True, default=str) for item in tied
        }
        if len(tied_signatures) > 1 or len(tied) > 1:
            raise ValueError(
                f"invalid workflow corpus: ambiguous replay baseline selectors for {workflow_spec_id} node {node_id}"
            )
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description="Workflow corpus tooling.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate_parser = subparsers.add_parser("validate")
    validate_parser.add_argument("--corpus-root")

    args = parser.parse_args()
    if args.command == "validate":
        report = workflow_corpus_report(corpus_root=args.corpus_root)
        print(json.dumps(report.model_dump(mode="json"), indent=2))
        raise SystemExit(0 if report.valid else 1)


if __name__ == "__main__":
    main()
