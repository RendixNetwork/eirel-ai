"""Owner-only dataset forge (general_chat edition).

The forge produces per-run ``GeneralChatBundle`` payloads of multi-turn
conversations. See ``shared/dataset_forge/general_chat_forge.py`` for the
generator and ``shared/dataset_forge/api.py`` for the public entry point.
"""

from __future__ import annotations

from shared.dataset_forge.api import (
    FORGE_GENERATOR_VERSION,
    ForgeError,
    GeneralChatBundle,
    ScriptedConversationFixture,
    SimulatedUserFixture,
    forge_general_chat_bundle,
)


FORGE_VERSION = FORGE_GENERATOR_VERSION

__all__ = [
    "FORGE_VERSION",
    "FORGE_GENERATOR_VERSION",
    "ForgeError",
    "GeneralChatBundle",
    "ScriptedConversationFixture",
    "SimulatedUserFixture",
    "forge_general_chat_bundle",
]
