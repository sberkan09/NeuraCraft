#include "FuzzFile.h"
#include "Image.h"

#include <stdexcept>

extern "C" int LLVMFuzzerTestOneInput(const std::uint8_t* data, std::size_t size) {
    nc::setMaxDecodePixels(1 << 22);
    const std::string path = fuzzFile(data, size, size && data[0] == 'P' ? ".ppm" : ".png");
    try {
        const nc::Image im = nc::loadImage(path);
        if (im.pixels.size() != im.width * im.height * im.channels) __builtin_trap();
        nc::toRgb(im);
    } catch (const std::exception&) {
    }
    return 0;
}
