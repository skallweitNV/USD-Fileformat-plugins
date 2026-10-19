// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <OpenImageIO/filesystem.h>
#include <OpenImageIO/imageio.h>
#include <fileformatutils/images.h>
#include <pxr/usd/sdf/zipFile.h>

#include <array>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#ifdef _WIN32
#include <windows.h>
#endif

static void
check(bool condition, const char* message)
{
    if (!condition)
        throw std::runtime_error(message);
}

int
main(int argc, char** argv)
{
#ifdef _WIN32
    SetErrorMode(SEM_FAILCRITICALERRORS | SEM_NOGPFAULTERRORBOX | SEM_NOOPENFILEERRORBOX);
    _set_abort_behavior(0, _WRITE_ABORT_MSG | _CALL_REPORTFAULT);
#endif
    try {
        check(argc == 2, "Pass an existing temporary directory for test artifacts");
        const std::filesystem::path scratch(argv[1]);
        const auto png_path = scratch / "rgba.png";
        const std::array<uint8_t, 16> pixels{ 255, 0, 0,   0,   0,  255, 0,  128,
                                              0,   0, 255, 255, 64, 32,  16, 192 };
        auto encoder = OIIO::ImageOutput::create(png_path.string());
        OIIO::ImageSpec spec(2, 2, 4, OIIO::TypeDesc::UINT8);
        spec.attribute("oiio:UnassociatedAlpha", 1);
        check(bool(encoder) && encoder->open(png_path.string(), spec), "Cannot create PNG fixture");
        check(encoder->write_image(OIIO::TypeDesc::UINT8, pixels.data()),
              "Cannot write PNG fixture");
        check(encoder->close(), "Cannot finalize PNG fixture");

        const auto archive_path = scratch / "images.usdz";
        auto archive = pxr::SdfZipFileWriter::CreateNew(archive_path.string());
        check(archive.AddFile(png_path.string(), "rgba.png") == "rgba.png",
              "Cannot package fixture");
        check(archive.Save(), "Cannot save package");

        for (const auto& source : { png_path.string(), archive_path.string() + "[rgba.png]" }) {
            check(adobe::usd::isImageFileSupported(source), "PNG should be supported");
            check(adobe::usd::getSbsarImageExtension(source) == "png",
                  "Integer image must select PNG");
            std::vector<uint8_t> encoded;
            check(adobe::usd::transcodeImageAssetToMemory(
                    source, (scratch / "converted.png").string(), encoded),
                  "PNG transcode failed");
            OIIO::Filesystem::IOMemReader reader(encoded.data(), encoded.size());
            OIIO::ImageSpec config;
            void* proxy = &reader;
            config.attribute("oiio:ioproxy", OIIO::TypeDesc::PTR, &proxy);
            config.attribute("oiio:UnassociatedAlpha", 1);
            auto decoder = OIIO::ImageInput::open("converted.png", &config);
            check(bool(decoder), "Cannot decode transcoded PNG");
            check(decoder->spec().width == 2 && decoder->spec().height == 2 &&
                    decoder->spec().nchannels == 4,
                  "Wrong shape");
            std::array<uint8_t, 16> result{};
            check(decoder->read_image(0, 0, 0, 4, OIIO::TypeDesc::UINT8, result.data()),
                  "Cannot read transcoded PNG");
            check(decoder->close(), "Cannot close transcoded PNG");
            check(result == pixels,
                  "PNG transcode changed orientation, color, or unassociated alpha");
        }

        const auto exr_path = scratch / "float.exr";
        const std::array<float, 3> hdr{ 0.25f, 2.0f, -0.5f };
        encoder = OIIO::ImageOutput::create(exr_path.string());
        check(bool(encoder) &&
                encoder->open(exr_path.string(), OIIO::ImageSpec(1, 1, 3, OIIO::TypeDesc::FLOAT)),
              "Cannot create EXR");
        check(encoder->write_image(OIIO::TypeDesc::FLOAT, hdr.data()) && encoder->close(),
              "Cannot write EXR");
        check(adobe::usd::getSbsarImageExtension(exr_path.string()) == "exr",
              "Float image must select EXR");
        std::vector<uint8_t> encoded;
        check(adobe::usd::transcodeImageAssetToMemory(
                exr_path.string(), (scratch / "converted.exr").string(), encoded),
              "EXR transcode failed");
        OIIO::Filesystem::IOMemReader reader(encoded.data(), encoded.size());
        OIIO::ImageSpec config;
        void* proxy = &reader;
        config.attribute("oiio:ioproxy", OIIO::TypeDesc::PTR, &proxy);
        auto decoder = OIIO::ImageInput::open("converted.exr", &config);
        check(bool(decoder), "Cannot decode transcoded EXR");
        std::array<float, 3> result{};
        check(decoder->read_image(0, 0, 0, 3, OIIO::TypeDesc::FLOAT, result.data()),
              "Cannot read transcoded EXR");
        check(decoder->close() && result == hdr, "EXR transcode changed HDR values");

        check(!adobe::usd::isImageFileSupported("image.unsupported_format"),
              "Unsupported format accepted");
        check(!adobe::usd::transcodeImageAssetToMemory(
                png_path.string(), "image.unsupported_format", encoded),
              "Unknown encoder accepted");
        check(encoded.empty(), "Failure retained stale output");
        check(!adobe::usd::transcodeImageAssetToMemory(
                (scratch / "missing.png").string(), "out.png", encoded),
              "Missing input accepted");
        const auto corrupt_path = scratch / "corrupt.png";
        std::ofstream(corrupt_path) << "not a PNG";
        check(!adobe::usd::transcodeImageAssetToMemory(corrupt_path.string(), "out.png", encoded),
              "Corrupt input accepted");
        check(adobe::usd::getSbsarImageExtension(corrupt_path.string()).empty(),
              "Corrupt image classified");
        check(!std::filesystem::exists(scratch / "converted.png"),
              "Transcode unexpectedly wrote a file");
        check(!std::filesystem::exists(scratch / "converted.exr"),
              "EXR transcode unexpectedly wrote a file");
        std::cout << "PASS: file/package PNG, alpha/orientation, HDR EXR, invalid inputs, memory "
                     "encoding\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
