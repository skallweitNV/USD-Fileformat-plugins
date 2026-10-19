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
from pxr import Ar, Sdf, Tf, Usd, UsdGeom, UsdShade

DATA = Path(__file__).parent / "data"


def open_fbx(path: Path, animation_mode: str | None = None) -> Usd.Stage:
    assert Sdf.FileFormat.FindByExtension(
        "fbx"
    ), "Register the usdFbx plugin before running tests"
    args = {
        "writeUsdPreviewSurface": "true",
        "writeOpenPBR": "false",
        "writeASM": "false",
    }
    if animation_mode is not None:
        args["animationMode"] = animation_mode
    layer = Sdf.Layer.FindOrOpen(str(path), args)
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


@pytest.mark.parametrize("animation_mode", [None, "adaptive"])
@pytest.mark.parametrize("kind", ["camera", "pivot", "held", "held_next"])
def test_animation_between_source_keys(kind: str, animation_mode: str | None) -> None:
    stage = open_fbx(DATA / "cubic_camera_pivot.fbx", animation_mode)
    names = {
        "camera": "CubicCamera",
        "pivot": "OrbitingTriangle",
        "held": "Held",
        "held_next": "HeldNext",
    }
    prim = next(p for p in stage.Traverse() if p.GetName() == names[kind])
    transform = UsdGeom.Xformable(prim)
    for seconds in (0, 0.125, 0.2375, 0.5, 0.999, 1, 1.001, 1.375, 1.75, 2):
        if kind == "camera":
            t = seconds / 2
            expected = (10 * (3 * t * t - 2 * t * t * t), 0, 0)
        elif kind == "pivot":
            angle = 2 * np.pi * seconds
            expected = (4 + 2 * np.cos(angle), 2 * np.sin(angle), 0)
        elif kind == "held":
            expected = (10 * int(seconds), 0, 0)
        else:
            expected = (10 * int(np.ceil(seconds)), 0, 0)
        matrix = transform.ComputeLocalToWorldTransform(
            seconds * stage.GetTimeCodesPerSecond()
        )
        np.testing.assert_allclose(
            matrix.ExtractTranslation(),
            expected,
            atol=5e-4,
            rtol=0,
            err_msg=f"{kind} at {seconds}s",
        )
        if kind == "pivot":
            basis = (
                (np.cos(angle), np.sin(angle), 0),
                (-np.sin(angle), np.cos(angle), 0),
                (0, 0, 1),
            )
            np.testing.assert_allclose(
                np.asarray(matrix)[:3, :3], basis, atol=1e-5, rtol=0
            )


def test_animation_modes_coexist() -> None:
    source = DATA / "cubic_camera_pivot.fbx"
    default = open_fbx(source)
    adaptive = open_fbx(source, "adaptive")
    keyframes = open_fbx(source, "keyframes")
    assert adaptive.GetRootLayer().identifier != keyframes.GetRootLayer().identifier
    assert (
        default.GetRootLayer().ExportToString()
        == adaptive.GetRootLayer().ExportToString()
    )
    camera_path = next(
        p.GetPath() for p in keyframes.Traverse() if p.GetName() == "CubicCamera"
    )
    sparse = UsdGeom.Xformable(keyframes.GetPrimAtPath(camera_path))
    dense = UsdGeom.Xformable(adaptive.GetPrimAtPath(camera_path))
    assert sparse.GetTimeSamples() == [0, 48]
    assert len(dense.GetTimeSamples()) > 2
    # Half a second: linear interpolation gives 2.5; the source cubic gives 1.5625.
    assert sparse.ComputeLocalToWorldTransform(12).ExtractTranslation()[
        0
    ] == pytest.approx(2.5)
    assert dense.ComputeLocalToWorldTransform(12).ExtractTranslation()[
        0
    ] == pytest.approx(1.5625, abs=5e-4)
    for time in (0, 48):
        np.testing.assert_allclose(
            sparse.ComputeLocalToWorldTransform(time),
            dense.ComputeLocalToWorldTransform(time),
            atol=1e-6,
        )
    for stage in (keyframes, adaptive):
        assert stage.GetTimeCodesPerSecond() == stage.GetFramesPerSecond() == 24
        assert stage.GetEndTimeCode() == 48


@pytest.mark.parametrize("name", ["Held", "HeldNext"])
def test_keyframes_does_not_add_held_samples(name: str) -> None:
    stage = open_fbx(DATA / "cubic_camera_pivot.fbx", "keyframes")
    prim = next(p for p in stage.Traverse() if p.GetName() == name)
    transform = UsdGeom.Xformable(prim)
    assert transform.GetTimeSamples() == [0, 24, 48]
    assert transform.ComputeLocalToWorldTransform(12).ExtractTranslation()[
        0
    ] == pytest.approx(5)


def test_keyframes_does_not_reconstruct_full_revolutions() -> None:
    stage = open_fbx(DATA / "cubic_camera_pivot.fbx", "keyframes")
    prim = next(p for p in stage.Traverse() if p.GetName() == "OrbitingTriangle")
    transform = UsdGeom.Xformable(prim)
    np.testing.assert_allclose(
        transform.ComputeLocalToWorldTransform(12).ExtractTranslation(),
        (6, 0, 0),
        atol=1e-5,
    )


@pytest.mark.parametrize("mode", ["splines", "unknown", "", "Adaptive"])
def test_invalid_animation_mode(mode: str) -> None:
    with pytest.raises(Tf.ErrorException, match="animationMode.*keyframes.*adaptive"):
        open_fbx(DATA / "cubic_camera_pivot.fbx", mode)


def test_animation_mode_payload_recomposition() -> None:
    source = open_fbx(DATA / "cubic_camera_pivot.fbx", "adaptive")
    stage = Usd.Stage.CreateInMemory()
    prim = stage.DefinePrim("/Asset")
    prim.SetMetadata("animationMode", "keyframes")
    prim.GetPayloads().AddPayload(
        str(DATA / "cubic_camera_pivot.fbx"), source.GetDefaultPrim().GetPath()
    )
    camera = next(p for p in stage.Traverse() if p.GetName() == "CubicCamera")
    path = camera.GetPath()
    sparse = UsdGeom.Xformable(camera)
    assert sparse.GetTimeSamples() == [0, 48]
    assert sparse.ComputeLocalToWorldTransform(12).ExtractTranslation()[
        0
    ] == pytest.approx(2.5)
    prim.SetMetadata("animationMode", "adaptive")
    dense = UsdGeom.Xformable(stage.GetPrimAtPath(path))
    assert len(dense.GetTimeSamples()) > 2
    assert dense.ComputeLocalToWorldTransform(12).ExtractTranslation()[
        0
    ] == pytest.approx(1.5625, abs=5e-4)
