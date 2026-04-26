# HDF5Container

## Overview

HDF5Container (`hdf5_container`) is a lightweight Python package for hierarchical HDF5 read/write operations.
It provides a simple interface for storing values into nested groups, retrieving datasets, and handling periodic flushes.

For module-level details, see [src/hdf5_container/README.md](src/hdf5_container/README.md).

## Installation

From the package root (the directory containing `pyproject.toml`):

```bash
pip install .
```

For development, install in editable mode:

```bash
pip install -e .
```

Dependencies are installed automatically.
To install dependencies only:

```bash
pip install -r requirements.txt
```

## Example

```python
from hdf5_container import HDF5Container

with HDF5Container.from_path("sample.h5") as container:
    container.store(keys=["users", "alice"], name="age", data=30)
    container.store(keys=["users", "alice"], name="city", data="Tokyo")
    age = container.access_value(keys=["users", "alice"], name="age")
    city = container.access_value(keys=["users", "alice"], name="city")
    print(age, city)  # 30 Tokyo
```
