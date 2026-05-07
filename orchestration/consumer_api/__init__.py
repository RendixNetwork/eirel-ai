"""Consumer-facing chat API for the subnet's product runtime.

Single product-mode entry shape: ``GraphChatRequest`` against
``/v1/graph/chat`` and ``/v1/graph/chat/stream`` (in ``main.py``).
The orchestrator backing this is the ``ProductOrchestrator`` which
loads user state from the product DB and routes to a
``ServingDeployment``.
"""
