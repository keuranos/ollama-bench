#!/usr/bin/env python3
"""Shared FastAPI application instance.

All modules that define routes import `app` from here.
The actual lifespan and startup happen in app.py.
"""
from fastapi import FastAPI
from contextlib import asynccontextmanager

# Placeholder lifespan — app.py will set the real one before route registration
@asynccontextmanager
async def _default_lifespan(app):
    yield

app = FastAPI(lifespan=_default_lifespan)
