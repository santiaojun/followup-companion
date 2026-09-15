"""
Quick smoke-test: verify CalleClient reads CALLE_API_KEY from .env correctly.
No network calls are made.
"""
import os, sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from providers.calle_client import CalleClient

client = CalleClient()

key = client.api_key
print(f"Key loaded : {'YES' if key else 'NO'}")
print(f"Length     : {len(key)} chars")
print(f"Prefix     : {key[:12]}...")   # show just enough to confirm it's the right key
print(f"Base URL   : {client.base_url}")
print(f"Timeout    : {client.timeout_s}s")
print("OK – CalleClient initialised without errors.")
