#include "ImageTo3D.h"

#include "HttpClient.h"
#include "Json.h"

#include <cstdlib>
#include <mutex>
#include <stdexcept>

namespace nc {

namespace {
std::mutex gMutex;
std::string gService;
bool gSet = false;
} // namespace

std::string imageTo3DService() {
    std::lock_guard<std::mutex> lock(gMutex);
    if (gSet) return gService;
    const char* env = std::getenv("NEURACRAFT_3D_URL");
    return env ? env : "";
}

void setImageTo3DService(const std::string& url) {
    std::lock_guard<std::mutex> lock(gMutex);
    gService = url;
    gSet = true;
}

bool imageTo3DAvailable() {
    const std::string url = imageTo3DService();
    if (url.empty()) return false;
    try {
        return http::request("GET", url, "/health", "", {}, 5).status == 200;
    } catch (const std::exception&) {
        return false;
    }
}

Mesh imageTo3D(const Image& picture, const ImageTo3DOptions& o) {
    const std::string url = imageTo3DService();
    if (url.empty()) throw std::runtime_error("image_to_3d: no 3D service (set NEURACRAFT_3D_URL, see services/trellis2)");
    if (o.quality != "fast" && o.quality != "standard" && o.quality != "high")
        throw std::invalid_argument("image_to_3d: quality is fast, standard or high");
    const std::string path = "/v1/image-to-3d?quality=" + o.quality + "&seed=" + std::to_string(o.seed) + "&texture=" + std::to_string(o.texture) +
                             "&faces=" + std::to_string(o.faces);
    const http::Response r = http::request("POST", url, path, "image/png", encodePng(picture), o.timeoutSeconds);
    if (r.status != 200) {
        std::string why = std::string(r.body.begin(), r.body.end()).substr(0, 500);
        try {
            const json::Value v = json::parse(why);
            if (const json::Value* e = v.find("error"); e && e->isString()) why = e->string;
        } catch (const std::exception&) {
        }
        throw std::runtime_error("image_to_3d: the 3D service answered " + std::to_string(r.status) + ": " + why);
    }
    Mesh m = decodeGlb(r.body, "the 3D service's answer");
    m.glb = r.body;
    return m;
}

} // namespace nc
