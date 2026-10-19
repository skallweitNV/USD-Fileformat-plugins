# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""FBX regressions; run with usdFbx installed on PXR_PLUGINPATH_NAME.

Requires pytest, numpy and USD Python bindings. No renderer or Falcor dependency.
"""

from pathlib import Path
import shutil
import struct
import zlib

import numpy as np
import pytest
from pxr import Ar, Sdf, Usd, UsdGeom, UsdShade

DATA = Path(__file__).parent / "data"


def open_fbx(path: Path) -> Usd.Stage:
    assert Sdf.FileFormat.FindByExtension(
        "fbx"
    ), "Register the usdFbx plugin before running tests"
    layer = Sdf.Layer.FindOrOpen(
        str(path),
        {
            "writeUsdPreviewSurface": "true",
            "writeOpenPBR": "false",
            "writeASM": "false",
        },
    )
    assert layer
    stage = Usd.Stage.Open(layer)
    assert stage
    return stage


def test_empty_texture_slot() -> None:
    stage = open_fbx(DATA / "empty_texture.fbx")
    material = next(
        UsdShade.Material(p) for p in stage.Traverse() if p.IsA(UsdShade.Material)
    )
    normal = material.ComputeSurfaceSource()[0].GetInput("normal")
    assert not normal or not normal.HasConnectedSource()
