"""
Shared MongoDB client singletons for the Supply Chain Reorder Agent.

A single MongoClient (sync) and AsyncIOMotorClient (async) are created once
at import time and reused across all in-process agent modules — tools, graph
nodes, and background workers that run in the same Python process.

Per the MongoDB connection skill: "Create client once only and reuse across
the application."  Two clients (old pattern) doubled the monitoring connection
overhead against Atlas:

    Old: 2 MongoClient × (100 pool + 2 monitoring) × 3 replica members = 612 max
    New: 1 MongoClient × (100 pool + 2 monitoring) × 3 replica members = 306 max

Exports
-------
sync_client  : PyMongo MongoClient   — used by @tool functions (run in threads)
               and by MongoDBSaver (LangGraph checkpointer).
async_client : Motor AsyncIOMotorClient — used by async graph node coroutines.
db_sync      : sync_client[DB_NAME]
db_async     : async_client[DB_NAME]
DB_NAME      : str — database name ("supply_chain_demo")
MONGO_KWARGS : dict — connection parameters shared by both clients.

Standalone workers (memory_retry_worker.py) that need different timeouts
create their own short-lived client — that is intentional and correct.
"""

import os

from dotenv import load_dotenv
from motor.motor_asyncio import AsyncIOMotorClient
from pymongo import MongoClient

load_dotenv()

DB_NAME = "supply_chain_demo"

# ---------------------------------------------------------------------------
# Connection parameters
# ---------------------------------------------------------------------------
# serverSelectionTimeoutMS — how long to wait for a reachable primary before
#     raising ServerSelectionTimeoutError.  30 s is generous; adjust down for
#     latency-sensitive paths.
# connectTimeoutMS — TCP + TLS handshake budget per socket.
# socketTimeoutMS  — max time to block waiting for a response on an open socket.
MONGO_KWARGS: dict = dict(
    serverSelectionTimeoutMS=30_000,
    connectTimeoutMS=10_000,
    socketTimeoutMS=30_000,
)

# ---------------------------------------------------------------------------
# Singletons — created once, reused everywhere in this process
# ---------------------------------------------------------------------------
sync_client: MongoClient = MongoClient(os.environ["MONGODB_URI"], **MONGO_KWARGS)
db_sync = sync_client[DB_NAME]

async_client: AsyncIOMotorClient = AsyncIOMotorClient(
    os.environ["MONGODB_URI"], **MONGO_KWARGS
)
db_async = async_client[DB_NAME]
