from __future__ import annotations

"""Unified operator CLI.

Dispatches to subcommands that implement operator-side administration:

    eirel-ai migrate           Run pending DB migrations.
    eirel-ai admin <sub> ...   Signed admin endpoint calls (validators,
                               runs, neurons, submissions, deployments).
"""

import sys

_USAGE = """\
usage: eirel-ai <command> [args]

commands:
  migrate          Run pending database migrations.
  admin <sub> ...  Owner admin tools (validators, runs, neurons,
                   submissions, deployments).

Run `eirel-ai admin` for admin subcommands.
"""


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help", "help"):
        sys.stdout.write(_USAGE)
        sys.exit(0 if len(sys.argv) >= 2 else 2)

    cmd = sys.argv[1]
    # Hand the downstream main() a sys.argv whose [0] is the full
    # command path so its own argparse help prints the right prog.
    sys.argv = [f"eirel-ai {cmd}", *sys.argv[2:]]

    if cmd == "migrate":
        from shared.common.db_migrate import main as _migrate_main

        _migrate_main()
        return

    if cmd == "admin":
        from control_plane.owner_api.cli.admin import main as _admin_main

        _admin_main()
        return

    sys.stderr.write(f"eirel-ai: unknown command '{cmd}'\n\n")
    sys.stderr.write(_USAGE)
    sys.exit(2)


if __name__ == "__main__":
    main()
