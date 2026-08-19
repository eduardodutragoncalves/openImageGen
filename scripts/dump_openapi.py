#!/usr/bin/env python
"""Print the OpenAPI schema without starting a server.

ADR-004: the TypeScript client is generated from this, so the two sides of the
repo cannot drift. Importing the app is enough — no GPU, no port, no weights.
"""

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("OIG_DRY_RUN", "true")

from app.main import app  # noqa: E402

json.dump(app.openapi(), sys.stdout, indent=2)
