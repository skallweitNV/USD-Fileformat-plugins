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


def write_png(path: Path, color: tuple[int, int, int]) -> None:
    # A single RGB pixel; generate fixtures without an imaging dependency.
    def chunk(kind: bytes, payload: bytes) -> bytes:
        return (
            struct.pack(">I", len(payload))
            + kind
            + payload
            + struct.pack(">I", zlib.crc32(kind + payload))
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(bytes((0, *color))))
        + chunk(b"IEND", b"")
    )


def test_distinct_images_with_same_basename(tmp_path: Path) -> None:
    source = tmp_path / "duplicates.fbx"
    shutil.copyfile(DATA / "duplicate_texture_names.fbx", source)
    for directory, color in zip(("a", "b"), ((255, 0, 0), (0, 255, 0))):
        write_png(tmp_path / directory / "shared.png", color)
    stage = open_fbx(source)
    images = set()
    for prim in stage.Traverse():
        for attribute in prim.GetAttributes():
            value = attribute.Get()
            if isinstance(value, Sdf.AssetPath) and value.path:
                resolved = value.resolvedPath or str(
                    Ar.GetResolver().Resolve(value.path)
                )
                asset = Ar.GetResolver().OpenAsset(Ar.ResolvedPath(resolved))
                assert asset, value.path
                images.add(bytes(asset.GetBuffer()))
    assert (
        len(images) == 2
    ), "Distinct source images must not merge in the material image cache"


def test_large_nonuniform_scale() -> None:
    stage = open_fbx(DATA / "empty_texture.fbx")
    mesh = next(UsdGeom.Mesh(p) for p in stage.Traverse() if p.IsA(UsdGeom.Mesh))
    matrix = np.array(mesh.ComputeLocalToWorldTransform(Usd.TimeCode.Default()))
    np.testing.assert_allclose(
        matrix, np.diag([100000, 3.5355439387, 4, 1]), rtol=1e-6, atol=1e-7
    )


def test_normalized_double_orientations() -> None:
    stage = open_fbx(DATA / "cubic_camera_pivot.fbx")
    orientations = [
        a
        for p in stage.Traverse()
        for a in p.GetAttributes()
        if a.GetName() == "xformOp:orient"
    ]
    assert orientations
    for attribute in orientations:
        assert attribute.GetTypeName() == Sdf.ValueTypeNames.Quatd
        for time in [Usd.TimeCode.Default(), *attribute.GetTimeSamples()]:
            value = attribute.Get(time)
            if value is not None:
                assert abs(value.GetLength() - 1) < 1e-14


def test_source_frame_timing() -> None:
    stage = open_fbx(DATA / "cubic_camera_pivot.fbx")
    assert stage.GetTimeCodesPerSecond() == 24
    assert stage.GetFramesPerSecond() == 24
    assert stage.GetStartTimeCode() == 0
    assert stage.GetEndTimeCode() == 48
