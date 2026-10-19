# Image I/O regressions

Build `imageIoTests` with `USD_FILEFORMATS_BUILD_TESTS=ON`, then run
`ctest -R "^imageIoTests$" --output-on-failure` in the build tree. The test creates
and cleans up its own temporary directory through a CMake runner. Alternatively,
pass an existing disposable directory to the executable and remove it afterward.

The integration branch replaces Hio image helpers with OpenImageIO and ArResolver
asset reads. Tests cover package-relative images, unassociated alpha, orientation,
stored color values, HDR EXR, memory encoding and input/encoder failures.
Hio-backed procedural image providers such as SBSAR are not supported by this
replacement. Making the backend selectable is separate upstream integration work.

DDS coverage constructs a BC1 red block and verifies format mapping, decoded
size and pixel values through the material image helpers.
