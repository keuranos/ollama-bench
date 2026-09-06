#!/usr/bin/env python3
"""Shared FastAPI app singleton.

Route-bearing modules (app.py, compare.py, runner.py) import `app` from here.
app.py creates the actual FastAPI instance with lifespan and assigns it.
Other modules register routes on the app before uvicorn starts.
"""

# Will be set by app.py during initialization
app = None
