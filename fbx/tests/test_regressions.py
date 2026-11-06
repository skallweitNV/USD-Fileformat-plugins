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


def open_fbx(
    path: Path, animation_mode: str | None = None, **arguments: str
) -> Usd.Stage:
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
    args.update(arguments)
    layer = Sdf.Layer.FindOrOpen(str(path.resolve()), args)
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


def test_reduces_channels_independently() -> None:
    source = DATA / "cubic_camera_pivot.fbx"
    stage = open_fbx(source, animationMaxSampleRate="960")
    pivot = next(p for p in stage.Traverse() if p.GetName() == "Pivot")
    counts = {
        op.GetOpType(): op.GetNumTimeSamples()
        for op in UsdGeom.Xformable(pivot).GetOrderedXformOps()
    }
    assert counts[UsdGeom.XformOp.TypeOrient] < 30
    assert counts[UsdGeom.XformOp.TypeTranslate] > counts[UsdGeom.XformOp.TypeOrient]
    assert counts.get(UsdGeom.XformOp.TypeScale, 0) <= 2
    # Full revolutions must remain visible between equal endpoints.
    transform = UsdGeom.Xformable(pivot)
    for seconds in np.linspace(0, 2, 257):
        matrix = transform.GetLocalTransformation(float(seconds * 24))
        angle = 2 * np.pi * seconds
        np.testing.assert_allclose(
            matrix.ExtractTranslation(),
            (4 - 4 * np.cos(angle), -4 * np.sin(angle), 0),
            atol=1e-4,
            rtol=0,
        )


def test_small_rotation_preserves_both_interpolations(tmp_path: Path) -> None:
    source = DATA / "cubic_camera_pivot.fbx"
    text = source.read_text().replace("a: 0,720\n", "a: -1,2\n")
    text = text.replace("a: 8452,8452\n", "a: 1032,1032\n")
    text = text.replace(
        '"RotationPivot", "Vector3D", "Vector", "",4,0,0',
        '"RotationPivot", "Vector3D", "Vector", "",0,0,0',
    )
    source = tmp_path / "small_rotation.fbx"
    source.write_text(text)
    stage = open_fbx(source)
    pivot = next(p for p in stage.Traverse() if p.GetName() == "Pivot")
    orientation = pivot.GetAttribute("xformOp:orient")
    transform = UsdGeom.Xformable(pivot)
    for seconds in np.linspace(0, 2, 257):
        t = seconds / 2
        expected = -1 + 3 * (3 * t * t - 2 * t * t * t)
        time = float(seconds * 24)
        quat = orientation.Get(time).GetNormalized()
        normalized_angle = np.degrees(
            2 * np.arctan2(quat.GetImaginary()[2], quat.GetReal())
        )
        matrix = transform.GetLocalTransformation(time)
        usd_angle = np.degrees(np.arctan2(matrix[0][1], matrix[0][0]))
        assert normalized_angle == pytest.approx(expected, abs=0.01)
        assert usd_angle == pytest.approx(expected, abs=0.01)


def test_collapses_translation_noise(tmp_path: Path) -> None:
    source = DATA / "cubic_camera_pivot.fbx"
    text = source.read_text()
    assert "a: 0,10\n" in text
    source = tmp_path / "noise.fbx"
    source.write_text(text.replace("a: 0,10\n", "a: 0,0.00000001\n"))
    stage = open_fbx(source)
    camera = next(p for p in stage.Traverse() if p.GetName() == "CubicCamera")
    op = next(
        op
        for op in UsdGeom.Xformable(camera).GetOrderedXformOps()
        if op.GetOpType() == UsdGeom.XformOp.TypeTranslate
    )
    assert op.GetNumTimeSamples() <= 2
    values = [op.Get(t) for t in op.GetTimeSamples()]
    assert all(value == values[0] for value in values)
    assert (stage.GetStartTimeCode(), stage.GetEndTimeCode()) == (0, 48)


@pytest.mark.parametrize("scale", ["0.01,0.01,0.01", "0.01,0.03,0.02"])
def test_translation_tolerance_accounts_for_parent_scale(
    tmp_path: Path, scale: str
) -> None:
    source = DATA / "cubic_camera_pivot.fbx"
    unscaled = open_fbx(source)
    camera = next(p for p in unscaled.Traverse() if p.GetName() == "CubicCamera")
    original_count = len(UsdGeom.Xformable(camera).GetTimeSamples())
    text = source.read_text()
    parent = f"""\n\tModel: 1000, "Model::ScaledParent", "Null" {{
        Version: 232
        Properties70: {{
            P: "Lcl Scaling", "Lcl Scaling", "", "A",{scale}
        }}
    }}\n"""
    text = text.replace("Objects:  {", "Objects:  {" + parent)
    text = text.replace('C: "OO",2261461967184,0', 'C: "OO",2261461967184,1000')
    text = text.replace("Connections:  {", 'Connections:  {\n\tC: "OO",1000,0')
    source = tmp_path / "scaled.fbx"
    source.write_text(text)
    scaled = open_fbx(source)
    camera = UsdGeom.Xformable(
        next(p for p in scaled.Traverse() if p.GetName() == "CubicCamera")
    )
    assert 2 < len(camera.GetTimeSamples()) < original_count
    for seconds in np.linspace(0, 2, 257):
        t = seconds / 2
        expected = (0.1 * (3 * t * t - 2 * t * t * t), 0, 0)
        np.testing.assert_allclose(
            camera.ComputeLocalToWorldTransform(
                float(seconds * 24)
            ).ExtractTranslation(),
            expected,
            atol=1e-4,
            rtol=0,
        )


def test_sampling_options_control_density(capfd: pytest.CaptureFixture[str]) -> None:
    source = DATA / "cubic_camera_pivot.fbx"
    stages = [
        open_fbx(source, animationPositionTolerance=tolerance)
        for tolerance in ("0.0001", "0.01")
    ]
    counts = []
    for stage in stages:
        camera = next(p for p in stage.Traverse() if p.GetName() == "CubicCamera")
        counts.append(
            sum(
                op.GetNumTimeSamples()
                for op in UsdGeom.Xformable(camera).GetOrderedXformOps()
            )
        )
    assert counts[0] > counts[1]
    limited = open_fbx(source, animationMaxSampleRate="0.5")
    assert "maximum rate 0.5 Hz" in capfd.readouterr().err
    camera = next(p for p in limited.Traverse() if p.GetName() == "CubicCamera")
    assert (
        max(
            op.GetNumTimeSamples()
            for op in UsdGeom.Xformable(camera).GetOrderedXformOps()
        )
        <= 49
    )
    held = UsdGeom.Xformable(
        next(p for p in limited.Traverse() if p.GetName() == "Held")
    )
    for seconds, expected in ((0.999, 0), (1, 10), (1.001, 10), (2, 20)):
        assert held.GetLocalTransformation(seconds * 24).ExtractTranslation()[
            0
        ] == pytest.approx(expected)


def test_sampling_metadata_recomposes_payload() -> None:
    source = DATA / "cubic_camera_pivot.fbx"
    imported = open_fbx(source)
    stage = Usd.Stage.CreateInMemory()
    asset = stage.DefinePrim("/Asset")
    asset.SetMetadata("animationPositionTolerance", "0.01")
    asset.GetPayloads().AddPayload(
        str(source.resolve()), imported.GetDefaultPrim().GetPath()
    )
    camera = next(p for p in stage.Traverse() if p.GetName() == "CubicCamera")
    path = camera.GetPath()
    loose_count = len(UsdGeom.Xformable(camera).GetTimeSamples())
    asset.SetMetadata("animationPositionTolerance", "0.0001")
    assert (
        len(UsdGeom.Xformable(stage.GetPrimAtPath(path)).GetTimeSamples()) > loose_count
    )
    asset.SetMetadata("animationMode", "keyframes")
    assert UsdGeom.Xformable(stage.GetPrimAtPath(path)).GetTimeSamples() == [0, 48]


@pytest.mark.parametrize(
    "name",
    [
        "animationPositionTolerance",
        "animationRotationTolerance",
        "animationScaleTolerance",
        "animationMaxSampleRate",
    ],
)
@pytest.mark.parametrize("value", ["invalid", "-1", "nan", "inf", "1junk"])
def test_rejects_invalid_sampling_options(name: str, value: str) -> None:
    source = DATA / "cubic_camera_pivot.fbx"
    with pytest.raises(Tf.ErrorException, match=name):
        open_fbx(source, **{name: value})


@pytest.mark.parametrize(
    "name",
    [
        "animationPositionTolerance",
        "animationRotationTolerance",
        "animationScaleTolerance",
    ],
)
def test_rejects_zero_tolerance(name: str) -> None:
    source = DATA / "cubic_camera_pivot.fbx"
    with pytest.raises(Tf.ErrorException, match=name):
        open_fbx(source, **{name: "0"})


@pytest.mark.parametrize("value", ["180", "181"])
def test_rejects_ambiguous_rotation_tolerance(value: str) -> None:
    with pytest.raises(Tf.ErrorException, match="animationRotationTolerance"):
        open_fbx(DATA / "cubic_camera_pivot.fbx", animationRotationTolerance=value)
