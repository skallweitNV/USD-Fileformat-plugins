# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

string(RANDOM LENGTH 16 ALPHABET 0123456789abcdef suffix)
set(scratch "${CMAKE_CURRENT_BINARY_DIR}/image-io-${suffix}")
file(MAKE_DIRECTORY "${scratch}")
execute_process(COMMAND "${TEST_EXECUTABLE}" "${scratch}" RESULT_VARIABLE result)
file(REAL_PATH "${CMAKE_CURRENT_BINARY_DIR}" build_root)
file(REAL_PATH "${scratch}" scratch_root)
get_filename_component(scratch_parent "${scratch_root}" DIRECTORY)
if(NOT scratch_parent STREQUAL build_root)
    message(FATAL_ERROR "Refusing to remove artifacts outside the test build directory")
endif()
file(REMOVE_RECURSE "${scratch_root}")
if(NOT result STREQUAL "0")
    message(FATAL_ERROR "Image I/O regression failed: ${result}")
endif()
