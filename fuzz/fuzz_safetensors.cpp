#include "FuzzFile.h"
#include "SafeTensors.h"

#include <stdexcept>

extern "C" int LLVMFuzzerTestOneInput(const std::uint8_t* data, std::size_t size) {
    const std::string path = fuzzFile(data, size, ".safetensors");
    try {
        const nc::SafeTensors st(path);
        for (const std::string& name : st.names()) {
            const auto& t = st.info(name);
            if (t.elements() > (1u << 20)) continue;
            try {
                st.load(name);
            } catch (const std::exception&) {
            }
        }
    } catch (const std::exception&) {
    }
    return 0;
}
