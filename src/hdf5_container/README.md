# hdf5_container

## Overview

Small HDF5 utility package built on top of `h5py`.
It provides a container class for hierarchical group access, dataset read/write, and periodic flush handling.

## Components

| Component | Description |
|-----------|-------------|
| [`container.py`](./container.py) | Core `HDF5Container` implementation for store/access operations. |
| [`utils.py`](./utils.py) | Utility helpers such as `reset_hdf5()` for recreating container files. |
| [`mixin/`](./mixin/README.md) | Mixin layer for I/O methods, special methods, and typing protocols. |
