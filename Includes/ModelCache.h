#ifndef NEURACRAFT_MODEL_CACHE_H
#define NEURACRAFT_MODEL_CACHE_H

#include <cstddef>
#include <functional>
#include <memory>
#include <set>
#include <string>

namespace nc {

// Loaded models kept between pipeline calls, so several generations with the
// same weights (a workflow's steps, a script's loop) read and quantize them
// once. Off by default: one-shot commands load each model once anyway and
// should not hold memory afterwards. runWorkflow switches it on for its run
// and releases each model after the last step that needs it.
//
// Entries are keyed by everything that shapes the loaded object (files, size,
// int8, LoRAs, adapters merged in) and tagged with their source, a normalized
// path plus "@WxH" for size-dependent models (see modelTag), which is what
// releaseModelsExcept matches.
void setModelCaching(bool enabled);
bool modelCaching();
void clearModelCache();
std::size_t modelCacheEntries();
// Drops every entry whose tag is not in `keep`.
void releaseModelsExcept(const std::set<std::string>& keep);
// "data/sd/../sd" and "data/sd/" -> "data/sd"; width / height > 0 append "@WxH".
std::string modelTag(const std::string& path, std::size_t width = 0, std::size_t height = 0);

namespace detail {
std::shared_ptr<void> cacheFind(const std::string& key);
void cacheStore(const std::string& key, const std::string& tag, std::shared_ptr<void> value);
} // namespace detail

// The cached model under `key`, or load() (which returns a shared_ptr<T>)
// stored under it. Without caching, just load().
template <typename T>
std::shared_ptr<T> cachedModel(const std::string& key, const std::string& tag, const std::function<std::shared_ptr<T>()>& load) {
    if (!modelCaching()) return load();
    if (std::shared_ptr<void> hit = detail::cacheFind(key)) return std::static_pointer_cast<T>(hit);
    std::shared_ptr<T> model = load();
    detail::cacheStore(key, tag, model);
    return model;
}

// Turns caching on for a scope and restores the previous state (and, if it
// was off, empties the cache) when the scope ends.
class ModelCacheScope {
public:
    ModelCacheScope();
    ~ModelCacheScope();
    ModelCacheScope(const ModelCacheScope&) = delete;
    ModelCacheScope& operator=(const ModelCacheScope&) = delete;

private:
    bool previous_;
};

} // namespace nc

#endif // NEURACRAFT_MODEL_CACHE_H
