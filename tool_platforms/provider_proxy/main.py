from __future__ import annotations

import logging

import uvicorn


def main() -> None:
    # Propagate INFO from our own modules to stdout — without this,
    # ``logger.info`` calls in app/redis_store/chutes_pricing are
    # silenced because the root logger defaults to WARNING when nothing
    # configures it.  uvicorn sets up its own access logger only.
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    uvicorn.run(
        "tool_platforms.provider_proxy.app:app",
        host="0.0.0.0",
        port=8092,
    )


if __name__ == "__main__":
    main()
