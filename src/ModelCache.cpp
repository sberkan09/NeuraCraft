#include "ModelCache.h"

#include <atomic>
#include <filesystem>
#include <map>
#include <mutex>
#include <vector>

namespace nc {

namespace {
struct Entry {
    std::string tag;
    std::shared_ptr<void> value;
};
std::atomic<bool> gEnabled{false};
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

void clearModelCache() {
    std::map<std::string, Entry> dropped;  // freed outside the lock
    std::lock_guard<std::mutex> lock(gMutex);
    dropped.swap(entries());
}

std::size_t modelCacheEntries() {
    std::lock_guard<std::mutex> lock(gMutex);
    return entries().size();
}

void releaseModelsExcept(const std::set<std::string>& keep) {
    std::vector<std::shared_ptr<void>> dropped;
    std::lock_guard<std::mutex> lock(gMutex);
    for (auto it = entries().begin(); it != entries().end();) {
        if (keep.count(it->second.tag)) {
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
    return it == entries().end() ? nullptr : it->second.value;
}
void cacheStore(const std::string& key, const std::string& tag, std::shared_ptr<void> value) {
    std::lock_guard<std::mutex> lock(gMutex);
    entries()[key] = Entry{tag, std::move(value)};
}
} // namespace detail

ModelCacheScope::ModelCacheScope() : previous_(modelCaching()) { gEnabled = true; }
ModelCacheScope::~ModelCacheScope() {
    if (!previous_) setModelCaching(false);
}

} // namespace nc
