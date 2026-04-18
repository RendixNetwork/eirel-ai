from __future__ import annotations

"""Parse Ansible inventory files to discover baremetal miner nodes.

The owner API reads the Ansible inventory on each ``list_runtime_nodes()``
call to discover available servers.  The inventory format follows standard
Ansible YAML inventory with a ``miner_nodes`` host group.
"""

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(slots=True)
class BaremetalNode:
    """Connection details for one baremetal server."""

    node_name: str
    ssh_host: str
    ssh_user: str
    ssh_key_path: str
    ssh_port: int


def parse_ansible_inventory(inventory_path: str) -> list[BaremetalNode]:
    """Parse an Ansible YAML inventory and return nodes from the ``miner_nodes`` group.

    Supports variable inheritance from ``all.vars`` to individual hosts.
    """
    path = Path(inventory_path)
    if not path.exists():
        return []
    data = yaml.safe_load(path.read_text())
    if not isinstance(data, dict):
        return []

    global_vars = data.get("all", {}).get("vars", {}) or {}
    default_user = str(global_vars.get("ansible_user", "eirel"))
    default_key = str(global_vars.get("ansible_ssh_private_key_file", "~/.ssh/eirel_deploy"))
    default_port = int(global_vars.get("ansible_port", global_vars.get("ansible_ssh_port", 22)))

    children = data.get("all", {}).get("children", {}) or {}
    miner_group = children.get("miner_nodes", {}) or {}
    hosts = miner_group.get("hosts", {}) or {}
    if not isinstance(hosts, dict):
        return []

    nodes: list[BaremetalNode] = []
    for host_alias, host_vars in hosts.items():
        if host_vars is None:
            host_vars = {}
        ssh_host = str(host_vars.get("ansible_host", host_alias))
        ssh_user = str(host_vars.get("ansible_user", default_user))
        ssh_key = str(host_vars.get("ansible_ssh_private_key_file", default_key))
        ssh_port = int(host_vars.get("ansible_port", host_vars.get("ansible_ssh_port", default_port)))
        nodes.append(
            BaremetalNode(
                node_name=str(host_alias),
                ssh_host=ssh_host,
                ssh_user=ssh_user,
                ssh_key_path=ssh_key,
                ssh_port=ssh_port,
            )
        )
    return nodes
