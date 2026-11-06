#include "Json.h"

#include <cstdint>
#include <stdexcept>

extern "C" int LLVMFuzzerTestOneInput(const std::uint8_t* data, std::size_t size) {
    try {
        const nc::json::Value v = nc::json::parse(std::string(reinterpret_cast<const char*>(data), size));
        const std::string again = nc::json::dump(v);
        nc::json::parse(again);  // what dump writes must parse
    } catch (const std::runtime_error&) {
    }
    return 0;
}
