# FBX regression tests

With USD Python bindings and the built FBX plugin on `PXR_PLUGINPATH_NAME`, run:

    python -m pytest fbx/tests/test_regressions.py -q

The tests require pytest and numpy. A missing plugin is a failure, not a skip.
Fixtures are synthetic ASCII FBX files generated independently with Autodesk FBX
SDK 2020.3.11, not exports from the plugin under test. `empty_texture.fbx`
contains an unnamed texture reference and a mesh with large nonuniform scale.

`duplicate_texture_names.fbx` references `a/shared.png` and `b/shared.png`;
the test generates red and green images in those locations.
