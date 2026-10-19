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
        // One BC1 block: every pixel selects the red RGB565 endpoint.
        std::array<uint32_t, 32> dds_header{};
        dds_header[0] = 0x20534444;
        dds_header[1] = 124;
        dds_header[2] = 0x81007;
        dds_header[3] = dds_header[4] = 4;
        dds_header[5] = 8;
        dds_header[19] = 32;
        dds_header[20] = 4;
        dds_header[21] = 0x31545844;
        dds_header[27] = 0x1000;
        adobe::usd::ImageAsset dds;
        dds.format = adobe::usd::getFormat("dds");
        dds.uri = "red.dds";
        dds.image.resize(136);
        std::memcpy(dds.image.data(), dds_header.data(), 128);
        dds.image[129] = 0xf8;
        adobe::usd::Image decoded_dds;
        check(adobe::usd::getFormatExtension(dds.format) == "dds", "DDS format mapping missing");
        check(decoded_dds.read(dds), "Cannot decode BC1 DDS in Adobe material helpers");
        check(decoded_dds.width == 4 && decoded_dds.height == 4 && decoded_dds.channels >= 3,
              "Wrong DDS shape");
        check(decoded_dds.pixels[0] == 1 && decoded_dds.pixels[1] == 0 &&
                decoded_dds.pixels[2] == 0,
              "Wrong DDS color");
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
