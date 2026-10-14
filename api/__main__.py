"""启动接口服务：python -m api [--host 127.0.0.1] [--port 8300]

环境变量（见 api/settings.py）：KB_AGENT_GENERATE / KB_AGENT_INTERRUPT /
KB_AGENT_TOKEN_STREAM / KB_AGENT_MAX_CONCURRENCY 等。
"""
from __future__ import annotations

import argparse

import uvicorn

from api.settings import ApiSettings


def main() -> None:
    settings = ApiSettings.from_env()
    parser = argparse.ArgumentParser(description="Run kb-agent API")
    parser.add_argument("--host", default=settings.host)
    parser.add_argument("--port", type=int, default=settings.port)
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args()
    uvicorn.run("api.app:app", host=args.host, port=args.port,
                reload=args.reload, log_level="info")


if __name__ == "__main__":
    main()
