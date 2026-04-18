from __future__ import annotations

"""High-level dataset forge API.

The forge now produces a single bundle type: multi-turn ``general_chat``
conversations. The legacy analyst / 5-domain harvest pipeline has been
retired — see ``shared/dataset_forge/general_chat_forge.py`` for the
new generator.

Public entry point:

    forge_general_chat_bundle(size=200, rng_seed=42) -> GeneralChatBundle
"""

from shared.dataset_forge.general_chat_forge import (
    CATEGORY_PERCENTAGES,
    GeneralChatBundle,
    ScriptedConversationFixture,
    SimulatedUserFixture,
    forge_general_chat_bundle as _forge_general_chat_bundle,
    forge_scripted_conversation,
    forge_simulated_user_conversation,
)


FORGE_GENERATOR_VERSION = "owner_dataset_forge_v2"


class ForgeError(RuntimeError):
    """Raised by any forge entry point for logical errors."""


def forge_general_chat_bundle(
    size: int = 200,
    rng_seed: int = 42,
) -> GeneralChatBundle:
    """Generate a deterministic ``general_chat`` bundle."""
    try:
        return _forge_general_chat_bundle(size=size, rng_seed=rng_seed)
    except ValueError as exc:
        raise ForgeError(str(exc)) from exc


__all__ = [
    "CATEGORY_PERCENTAGES",
    "FORGE_GENERATOR_VERSION",
    "ForgeError",
    "GeneralChatBundle",
    "ScriptedConversationFixture",
    "SimulatedUserFixture",
    "forge_general_chat_bundle",
    "forge_scripted_conversation",
    "forge_simulated_user_conversation",
]
