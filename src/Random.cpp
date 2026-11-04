#include "Random.h"

namespace nc {

std::mt19937& rng() {
    static std::mt19937 generator{std::random_device{}()};
    return generator;
}

void setSeed(unsigned int seed) {
    rng().seed(seed);
}

} // namespace nc
