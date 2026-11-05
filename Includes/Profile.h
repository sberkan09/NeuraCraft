#ifndef NEURACRAFT_PROFILE_H
#define NEURACRAFT_PROFILE_H

#include <cstddef>
#include <iosfwd>
#include <string>
#include <vector>

namespace nc {

// Lightweight wall-clock profiler for finding where inference time goes
// (`--profile` on the image commands). Off by default; a disabled scope costs
// one relaxed atomic load. Scopes nest: each one is charged its own (self)
// time, with time spent in inner scopes charged to those instead, per thread.
void setProfiling(bool enabled);
bool profiling();
void resetProfile();

struct ProfileEntry {
    std::string name;
    double seconds = 0.0;  // self time
    std::size_t calls = 0;
};
// Entries sorted by time, largest first.
std::vector<ProfileEntry> profileEntries();
// Table of the entries with their share of the profiled time.
void printProfile(std::ostream& out);

class ProfileScope {
public:
    explicit ProfileScope(const char* name);
    ~ProfileScope();
    ProfileScope(const ProfileScope&) = delete;
    ProfileScope& operator=(const ProfileScope&) = delete;

private:
    const char* name_;
    bool active_;
    double start_ = 0.0;
    double inner_ = 0.0;  // time of nested scopes
    ProfileScope* parent_ = nullptr;
};

} // namespace nc

#define NC_PROFILE_CAT2(a, b) a##b
#define NC_PROFILE_CAT(a, b) NC_PROFILE_CAT2(a, b)
#define NC_PROFILE(name) ::nc::ProfileScope NC_PROFILE_CAT(ncProfileScope_, __LINE__)(name)

#endif // NEURACRAFT_PROFILE_H
