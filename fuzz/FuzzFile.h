#ifndef NEURACRAFT_FUZZ_FILE_H
#define NEURACRAFT_FUZZ_FILE_H

// The readers take file paths: each input is written to a per-process file first.

#include <cstddef>
#include <cstdint>
#include <fstream>
#include <string>
#include <unistd.h>

inline std::string fuzzFile(const std::uint8_t* data, std::size_t size, const char* ext) {
    static const std::string dir = [] {
        const char* shm = "/dev/shm";
        return std::string(access(shm, W_OK) == 0 ? shm : "/tmp");
    }();
    const std::string path = dir + "/neuracraft_fuzz_" + std::to_string(getpid()) + ext;
    std::ofstream(path, std::ios::binary | std::ios::trunc).write(reinterpret_cast<const char*>(data), static_cast<std::streamsize>(size));
    return path;
}

#endif
