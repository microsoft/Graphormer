# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from pathlib import Path
import importlib

# automatically import any Python files in the criterions/ directory
for file in sorted(Path(__file__).parent.glob("*.py")):
    if not file.name.startswith("_"):
        importlib.import_module("graphormer.criterions." + file.name[:-3])
