#ifndef NEURACRAFT_RANDOM_H
#define NEURACRAFT_RANDOM_H

#include <random>

namespace nc {

// Library-wide random generator. Seeded from std::random_device by default;
// call setSeed() for reproducible weight init, shuffling and dropout.
std::mt19937& rng();
void setSeed(unsigned int seed);

} // namespace nc

#endif // NEURACRAFT_RANDOM_H
