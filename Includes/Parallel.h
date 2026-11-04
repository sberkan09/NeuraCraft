#ifndef NEURACRAFT_PARALLEL_H
#define NEURACRAFT_PARALLEL_H

#include <cstddef>
#include <functional>

namespace nc {

// Number of CPU threads used by the library (default: all hardware threads).
// Call setThreadCount() between operations, not while one is running.
unsigned threadCount();
void setThreadCount(unsigned threads);

// Runs fn(chunkBegin, chunkEnd) over [begin, end) split into chunks of at least
// `grain` items, on the library thread pool. The calling thread participates.
// Nested calls from inside a worker run serially. The first exception thrown by
// fn is rethrown in the caller after all chunks finish.
void parallelFor(std::size_t begin, std::size_t end, std::size_t grain,
                 const std::function<void(std::size_t, std::size_t)>& fn);

} // namespace nc

#endif // NEURACRAFT_PARALLEL_H
