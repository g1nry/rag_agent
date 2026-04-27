import argparse

import uvicorn

from rag_agent.core.config import get_settings


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the RAG agent backend.")
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload for local development.",
    )
    args = parser.parse_args()

    settings = get_settings()
    uvicorn.run(
        "rag_agent.main:app",
        host=settings.app_host,
        port=settings.app_port,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
