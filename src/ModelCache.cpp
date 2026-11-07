#include "ModelCache.h"

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <filesystem>
#include <map>
#include <mutex>
#include <vector>

namespace nc {

namespace {
struct Entry {
    std::string tag;
    std::shared_ptr<void> value;
    std::uint64_t used = 0;  // last use (gClock)
};
std::atomic<bool> gEnabled{false};
std::atomic<std::int64_t> gLoadMicros{0};
thread_local int tLoadDepth = 0;
std::atomic<std::size_t> gKeep{0};
std::uint64_t gClock = 0;  // under gMutex
std::mutex gMutex;
std::map<std::string, Entry>& entries() {
    static std::map<std::string, Entry> e;
    return e;
}
} // namespace

void setModelCaching(bool enabled) {
    gEnabled = enabled;
    if (!enabled) clearModelCache();
}
bool modelCaching() { return gEnabled.load(); }

double modelLoadSeconds() { return static_cast<double>(gLoadMicros.load()) / 1e6; }

detail::LoadTimer::LoadTimer() : start_(std::chrono::steady_clock::now()), outer_(tLoadDepth++ == 0) {}
detail::LoadTimer::~LoadTimer() {
    --tLoadDepth;
    if (outer_) gLoadMicros += std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - start_).count();
}

void clearModelCache() {
    std::map<std::string, Entry> dropped;  // freed outside the lock
    std::lock_guard<std::mutex> lock(gMutex);
    dropped.swap(entries());
}

std::size_t modelCacheEntries() {
    std::lock_guard<std::mutex> lock(gMutex);
    return entries().size();
}

void setModelCacheKeep(std::size_t models) { gKeep = models; }
std::size_t modelCacheKeep() { return gKeep.load(); }

void releaseModelsExcept(const std::set<std::string>& keep) {
    std::vector<std::shared_ptr<void>> dropped;
    std::lock_guard<std::mutex> lock(gMutex);
    // The most recently used tags outside `keep` that stay anyway.
    std::set<std::string> recent;
    if (gKeep.load()) {
        std::vector<std::pair<std::uint64_t, std::string>> uses;
        for (const auto& [_, e] : entries())
            if (!keep.count(e.tag)) uses.push_back({e.used, e.tag});
        std::sort(uses.rbegin(), uses.rend());
        for (const auto& [_, tag] : uses) {
            if (recent.size() >= gKeep.load()) break;
            recent.insert(tag);
        }
    }
    for (auto it = entries().begin(); it != entries().end();) {
        if (keep.count(it->second.tag) || recent.count(it->second.tag)) {
            ++it;
        } else {
            dropped.push_back(std::move(it->second.value));
            it = entries().erase(it);
        }
    }
}

std::string modelTag(const std::string& path, std::size_t width, std::size_t height) {
    std::string tag = std::filesystem::path(path).lexically_normal().string();
    while (tag.size() > 1 && (tag.back() == '/' || tag.back() == '\\')) tag.pop_back();
    if (width > 0 && height > 0) tag += "@" + std::to_string(width) + "x" + std::to_string(height);
    return tag;
}

namespace detail {
std::shared_ptr<void> cacheFind(const std::string& key) {
    std::lock_guard<std::mutex> lock(gMutex);
    auto it = entries().find(key);
    if (it == entries().end()) return nullptr;
    it->second.used = ++gClock;
    return it->second.value;
}
void cacheStore(const std::string& key, const std::string& tag, std::shared_ptr<void> value) {
    std::lock_guard<std::mutex> lock(gMutex);
    entries()[key] = Entry{tag, std::move(value), ++gClock};
}
} // namespace detail

ModelCacheScope::ModelCacheScope() : previous_(modelCaching()) { gEnabled = true; }
ModelCacheScope::~ModelCacheScope() {
    if (!previous_) setModelCaching(false);
}

} // namespace nc
