#!/usr/bin/env python3
"""Health check script for the medical AI load testing service."""

import sys
import asyncio
import aiohttp
from typing import Dict, Any


async def check_service_health() -> Dict[str, Any]:
    """Check if the service is healthy."""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get('http://localhost:8000/health', timeout=5) as response:
                if response.status == 200:
                    data = await response.json()
                    return {"status": "healthy", "details": data}
                else:
                    return {"status": "unhealthy", "error": f"HTTP {response.status}"}
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}


def main():
    """Main health check function."""
    try:
        result = asyncio.run(check_service_health())
        if result["status"] == "healthy":
            print("Service is healthy")
            sys.exit(0)
        else:
            print(f"Service is unhealthy: {result.get('error', 'Unknown error')}")
            sys.exit(1)
    except Exception as e:
        print(f"Health check failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()