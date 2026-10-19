# FBX regression tests

With USD Python bindings and the built FBX plugin on `PXR_PLUGINPATH_NAME`, run:

    python -m pytest fbx/tests/test_regressions.py -q

The tests require pytest and numpy. A missing plugin is a failure, not a skip.
Fixtures are synthetic ASCII FBX files generated independently with Autodesk FBX
SDK 2020.3.11, not exports from the plugin under test. `empty_texture.fbx`
contains an unnamed texture reference and a mesh with large nonuniform scale.

`duplicate_texture_names.fbx` references `a/shared.png` and `b/shared.png`;
the test generates red and green images in those locations.

`cubic_camera_pivot.fbx` uses meters and 24 fps. It contains a cubic camera
translation, a linear two-revolution Euler rotation around a pivot, and both
constant interpolation modes. Its clip spans two seconds.

Adaptive baking evaluates the source frame grid and refines subframes against
SDK transforms. It preserves full Euler revolutions and pivot-induced motion,
but does not preserve source curve tangents. Translation refinement targets
0.1 mm locally, subject to float precision; finite probes are not a universal
world-space error bound. Node orientations use normalized double samples.
Dense joint hierarchies can be expensive: a production Sol asset took about
30 minutes. The skeletal curve sampler is unchanged. Arbitrary affine/sheared
inheritance is outside these regression checks.
