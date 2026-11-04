#ifndef NEURACRAFT_UNICODE_TABLES_H
#define NEURACRAFT_UNICODE_TABLES_H

#include <cstdint>

namespace nc {

// Unicode general categories L* (\p{L}) and N* (\p{N}); see scripts/gen_unicode_tables.py.
bool unicodeIsLetter(std::uint32_t c);
bool unicodeIsNumber(std::uint32_t c);

} // namespace nc

#endif // NEURACRAFT_UNICODE_TABLES_H
