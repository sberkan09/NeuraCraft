#include "Profile.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <iomanip>
#include <map>
#include <mutex>
#include <ostream>

namespace nc {

namespace {
std::atomic<bool> gEnabled{false};
std::mutex gMutex;
std::map<std::string, ProfileEntry>& table() {
    static std::map<std::string, ProfileEntry> t;
    return t;
}
thread_local ProfileScope* tCurrent = nullptr;

double now() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}
} // namespace

void setProfiling(bool enabled) { gEnabled.store(enabled, std::memory_order_relaxed); }
bool profiling() { return gEnabled.load(std::memory_order_relaxed); }

void resetProfile() {
    std::lock_guard<std::mutex> lock(gMutex);
    table().clear();
}

std::vector<ProfileEntry> profileEntries() {
    std::vector<ProfileEntry> out;
    {
        std::lock_guard<std::mutex> lock(gMutex);
        for (const auto& kv : table()) out.push_back(kv.second);
    }
    std::sort(out.begin(), out.end(), [](const ProfileEntry& a, const ProfileEntry& b) { return a.seconds > b.seconds; });
    return out;
}

void printProfile(std::ostream& out) {
    const std::vector<ProfileEntry> entries = profileEntries();
    double total = 0.0;
    for (const auto& e : entries) total += e.seconds;
    out << "profile (self time; " << std::fixed << std::setprecision(2) << total << " s profiled):\n";
    for (const auto& e : entries)
        out << "  " << std::left << std::setw(28) << e.name << std::right << std::setw(10) << std::setprecision(3) << e.seconds
            << " s" << std::setw(7) << std::setprecision(1) << (total > 0 ? 100.0 * e.seconds / total : 0.0) << " %"
            << std::setw(10) << e.calls << " calls\n";
    out << std::defaultfloat;
}

ProfileScope::ProfileScope(const char* name) : name_(name), active_(profiling()) {
    if (!active_) return;
    parent_ = tCurrent;
    tCurrent = this;
    start_ = now();
}

ProfileScope::~ProfileScope() {
    if (!active_) return;
    const double elapsed = now() - start_;
    tCurrent = parent_;
    if (parent_) parent_->inner_ += elapsed;
    std::lock_guard<std::mutex> lock(gMutex);
    ProfileEntry& e = table()[name_];
    if (e.name.empty()) e.name = name_;
    e.seconds += std::max(0.0, elapsed - inner_);
    ++e.calls;
}

} // namespace nc
