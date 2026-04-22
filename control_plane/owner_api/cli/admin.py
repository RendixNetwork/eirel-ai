"""Subnet operator CLI for the Eirel owner API.

Signs requests with the owner hotkey (Bittensor wallet) and calls the
/v1/admin/* endpoints. Run from the operator's host machine where the
wallet files live.

Usage:
  eirel-ai admin whoami
  eirel-ai admin validators add <hotkey> [--uid N] [--stake N]
  eirel-ai admin validators remove <hotkey>
  eirel-ai admin validators list
  eirel-ai admin validators enable <hotkey>
  eirel-ai admin validators disable <hotkey>
  eirel-ai admin runs list
  eirel-ai admin runs current
  eirel-ai admin neurons list
  eirel-ai admin submissions list [--limit N]
  eirel-ai admin deployments list [--limit N]

Configuration (env vars):
  OWNER_API_PUBLIC_URL      Owner API base URL (default: http://127.0.0.1:18020)
                            Fallback: EIREL_ADMIN_API_URL (legacy)
  EIREL_OWNER_WALLET_NAME   Bittensor wallet name (default: owner)
                            Fallback: EIREL_ADMIN_WALLET_NAME (legacy)
  EIREL_OWNER_HOTKEY_NAME   Bittensor hotkey name (default: op1)
                            Fallback: EIREL_ADMIN_HOTKEY_NAME (legacy)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from typing import Any

import httpx

from shared.common.bittensor_signing import load_signer, LoadedSigner


def _default_base_url() -> str:
    # Canonical is OWNER_API_PUBLIC_URL (same var every other piece uses).
    # EIREL_ADMIN_API_URL kept as legacy fallback.
    return (
        os.getenv("OWNER_API_PUBLIC_URL")
        or os.getenv("EIREL_ADMIN_API_URL")
        or "http://127.0.0.1:18020"
    ).rstrip("/")


def _load_owner_signer() -> LoadedSigner:
    # Aligns with EIREL_OWNER_WALLET_NAME / EIREL_OWNER_HOTKEY_NAME used
    # everywhere else (.env.compose.example, owner-api ConfigMap, operator
    # stack). Legacy EIREL_ADMIN_* names still accepted for backward compat.
    wallet_name = (
        os.getenv("EIREL_OWNER_WALLET_NAME")
        or os.getenv("EIREL_ADMIN_WALLET_NAME")
        or "owner"
    )
    hotkey_name = (
        os.getenv("EIREL_OWNER_HOTKEY_NAME")
        or os.getenv("EIREL_ADMIN_HOTKEY_NAME")
        or "op1"
    )
    try:
        return load_signer(wallet_name=wallet_name, hotkey_name=hotkey_name)
    except Exception as exc:
        sys.stderr.write(
            f"Failed to load wallet {wallet_name}/{hotkey_name}: {exc}\n"
            f"Set EIREL_OWNER_WALLET_NAME and EIREL_OWNER_HOTKEY_NAME if different.\n"
        )
        sys.exit(2)


def _request(
    *,
    method: str,
    path: str,
    body: dict[str, Any] | None = None,
) -> Any:
    signer = _load_owner_signer()
    base_url = _default_base_url()
    body_bytes = json.dumps(body).encode() if body is not None else b""
    body_hash = hashlib.sha256(body_bytes).hexdigest()
    headers = signer.signed_headers(method=method, path=path, body_hash=body_hash)
    if body is not None:
        headers["Content-Type"] = "application/json"

    url = f"{base_url}{path}"
    try:
        resp = httpx.request(
            method=method,
            url=url,
            headers=headers,
            content=body_bytes if body is not None else None,
            timeout=30.0,
        )
    except httpx.HTTPError as exc:
        sys.stderr.write(f"HTTP request failed: {exc}\n")
        sys.exit(1)

    if resp.status_code >= 400:
        sys.stderr.write(f"{method} {path} → {resp.status_code}\n")
        sys.stderr.write(resp.text + "\n")
        sys.exit(1)
    try:
        return resp.json()
    except json.JSONDecodeError:
        return resp.text


def _print(payload: Any) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True))


# ------------------------------------------------------------------
# Subcommand handlers
# ------------------------------------------------------------------


def cmd_whoami(_args) -> None:
    _print(_request(method="GET", path="/v1/admin/whoami"))


def cmd_validator_add(args) -> None:
    _print(_request(
        method="POST",
        path="/v1/admin/validators",
        body={"hotkey": args.hotkey, "uid": args.uid, "stake": args.stake},
    ))


def cmd_validator_remove(args) -> None:
    _print(_request(method="DELETE", path=f"/v1/admin/validators/{args.hotkey}"))


def cmd_validator_list(_args) -> None:
    _print(_request(method="GET", path="/v1/admin/validators"))


def cmd_validator_enable(args) -> None:
    _print(_request(
        method="PATCH",
        path=f"/v1/admin/validators/{args.hotkey}",
        body={"is_active": True},
    ))


def cmd_validator_disable(args) -> None:
    _print(_request(
        method="PATCH",
        path=f"/v1/admin/validators/{args.hotkey}",
        body={"is_active": False},
    ))


def cmd_runs_list(_args) -> None:
    _print(_request(method="GET", path="/v1/admin/runs"))


def cmd_runs_current(_args) -> None:
    _print(_request(method="GET", path="/v1/admin/runs/current"))


def cmd_neurons_list(_args) -> None:
    _print(_request(method="GET", path="/v1/admin/neurons"))


def cmd_submissions_list(args) -> None:
    _print(_request(method="GET", path=f"/v1/admin/submissions?limit={args.limit}"))


def cmd_deployments_list(args) -> None:
    _print(_request(method="GET", path=f"/v1/admin/deployments?limit={args.limit}"))


# ------------------------------------------------------------------
# Parser
# ------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="eirel-ai admin",
        description="Eirel subnet operator CLI (owner-signed).",
    )
    sub = parser.add_subparsers(dest="group", required=True)

    # whoami
    sub.add_parser("whoami", help="Verify this wallet is recognised as the owner")

    # validators
    v = sub.add_parser("validators", help="Manage the validator whitelist")
    v_sub = v.add_subparsers(dest="cmd", required=True)

    v_add = v_sub.add_parser("add", help="Add (or re-enable) a validator hotkey")
    v_add.add_argument("hotkey")
    v_add.add_argument("--uid", type=int, default=0)
    v_add.add_argument("--stake", type=int, default=0)

    v_rm = v_sub.add_parser("remove", help="Remove a validator hotkey")
    v_rm.add_argument("hotkey")

    v_sub.add_parser("list", help="List all whitelisted validators")

    v_en = v_sub.add_parser("enable", help="Enable a temporarily-disabled validator")
    v_en.add_argument("hotkey")
    v_dis = v_sub.add_parser("disable", help="Temporarily disable a validator")
    v_dis.add_argument("hotkey")

    # runs
    r = sub.add_parser("runs", help="View evaluation runs")
    r_sub = r.add_subparsers(dest="cmd", required=True)
    r_sub.add_parser("list", help="List the last 20 runs")
    r_sub.add_parser("current", help="Show the currently-open run")

    # neurons
    n = sub.add_parser("neurons", help="View metagraph-registered hotkeys")
    n_sub = n.add_subparsers(dest="cmd", required=True)
    n_sub.add_parser("list", help="List all registered hotkeys")

    # submissions
    s = sub.add_parser("submissions", help="View miner submissions")
    s_sub = s.add_subparsers(dest="cmd", required=True)
    s_list = s_sub.add_parser("list", help="List recent submissions")
    s_list.add_argument("--limit", type=int, default=50)

    # deployments
    d = sub.add_parser("deployments", help="View miner deployments")
    d_sub = d.add_subparsers(dest="cmd", required=True)
    d_list = d_sub.add_parser("list", help="List recent deployments")
    d_list.add_argument("--limit", type=int, default=50)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    dispatch = {
        ("whoami", None): cmd_whoami,
        ("validators", "add"): cmd_validator_add,
        ("validators", "remove"): cmd_validator_remove,
        ("validators", "list"): cmd_validator_list,
        ("validators", "enable"): cmd_validator_enable,
        ("validators", "disable"): cmd_validator_disable,
        ("runs", "list"): cmd_runs_list,
        ("runs", "current"): cmd_runs_current,
        ("neurons", "list"): cmd_neurons_list,
        ("submissions", "list"): cmd_submissions_list,
        ("deployments", "list"): cmd_deployments_list,
    }
    key = (args.group, getattr(args, "cmd", None))
    handler = dispatch.get(key)
    if handler is None:
        parser.print_help()
        sys.exit(2)
    handler(args)


if __name__ == "__main__":
    main()
