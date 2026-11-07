#include "ImageTo3D.h"

#include "HttpClient.h"
#include "Json.h"

#include <cstdlib>
#include <mutex>
#include <stdexcept>

namespace nc {

namespace {
// A service's address: set by the program, else from the environment.
struct ServiceAddress {
    const char* env;
    std::mutex mutex;
    std::string url;
    bool set = false;
    std::string get() {
        std::lock_guard<std::mutex> lock(mutex);
        if (set) return url;
        const char* v = std::getenv(env);
        return v ? v : "";
    }
    void put(const std::string& u) {
        std::lock_guard<std::mutex> lock(mutex);
        url = u;
        set = true;
    }
};
ServiceAddress gImageTo3D{"NEURACRAFT_3D_URL", {}, {}, false};
ServiceAddress gRig{"NEURACRAFT_RIG_URL", {}, {}, false};

bool healthy(const std::string& url) {
    if (url.empty()) return false;
    try {
        return http::request("GET", url, "/health", "", {}, 5).status == 200;
    } catch (const std::exception&) {
        return false;
    }
}

// The service's answer as a mesh (its glTF kept), or its error message as an exception.
Mesh glbAnswer(const http::Response& r, const std::string& what, const std::string& service) {
    if (r.status != 200) {
        std::string why = std::string(r.body.begin(), r.body.end()).substr(0, 500);
        try {
            const json::Value v = json::parse(why);
            if (const json::Value* e = v.find("error"); e && e->isString()) why = e->string;
        } catch (const std::exception&) {
        }
        throw std::runtime_error(what + ": the " + service + " answered " + std::to_string(r.status) + ": " + why);
    }
    Mesh m = decodeGlb(r.body, "the " + service + "'s answer");
    m.glb = r.body;
    return m;
}
} // namespace

std::string imageTo3DService() { return gImageTo3D.get(); }
void setImageTo3DService(const std::string& url) { gImageTo3D.put(url); }
bool imageTo3DAvailable() { return healthy(imageTo3DService()); }
std::string rigService() { return gRig.get(); }
void setRigService(const std::string& url) { gRig.put(url); }
bool rigAvailable() { return healthy(rigService()); }

Mesh imageTo3D(const Image& picture, const ImageTo3DOptions& o) {
    const std::string url = imageTo3DService();
    if (url.empty()) throw std::runtime_error("image_to_3d: no 3D service (set NEURACRAFT_3D_URL, see services/trellis2)");
    if (o.quality != "fast" && o.quality != "standard" && o.quality != "high")
        throw std::invalid_argument("image_to_3d: quality is fast, standard or high");
    const std::string path = "/v1/image-to-3d?quality=" + o.quality + "&seed=" + std::to_string(o.seed) + "&texture=" + std::to_string(o.texture) +
                             "&faces=" + std::to_string(o.faces);
    return glbAnswer(http::request("POST", url, path, "image/png", encodePng(picture), o.timeoutSeconds), "image_to_3d", "3D service");
}

Mesh rigMesh(const Mesh& mesh, const RigOptions& o) {
    const std::string url = rigService();
    if (url.empty()) throw std::runtime_error("rig: no rigging service (set NEURACRAFT_RIG_URL, see services/rigging)");
    checkMesh(mesh, "rig");
    if (mesh.empty()) throw std::invalid_argument("rig: the model has no triangles");
    const std::string path = "/v1/rig?seed=" + std::to_string(o.seed) + "&postprocess=" + (o.postprocess ? "1" : "0");
    const std::vector<std::uint8_t> body = mesh.glb.empty() ? encodeGlb(mesh) : mesh.glb;
    return glbAnswer(http::request("POST", url, path, "model/gltf-binary", body, o.timeoutSeconds), "rig", "rigging service");
}

} // namespace nc
