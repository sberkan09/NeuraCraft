#ifndef NEURACRAFT_IMAGE_TO_3D_H
#define NEURACRAFT_IMAGE_TO_3D_H

// A whole 3D model from one picture, made by a 3D service: a separate process next to the engine
// (services/trellis2: TRELLIS.2 on a GPU) reached over plain HTTP on the local machine or network.
// The service's address comes from NEURACRAFT_3D_URL ("http://127.0.0.1:8188") or setImageTo3DService;
// workflows cannot choose it (no request ever goes where a workflow says).
//
// Protocol: POST <url>/v1/image-to-3d?quality=fast|standard|high&seed=N&texture=N&faces=N with a PNG
// body (RGBA: the alpha is the object's mask; RGB: the service removes the background) answers 200
// with a glTF binary, or an error status with {"error": "..."}. GET <url>/health answers 200.

#include "Image.h"
#include "Mesh.h"

#include <cstddef>
#include <string>

namespace nc {

struct ImageTo3DOptions {
    std::string quality = "standard";  // fast (512^3), standard (1024^3), high (1536^3)
    long seed = 42;
    std::size_t texture = 2048;        // texture side, pixels
    std::size_t faces = 500000;        // triangle budget of the result
    double timeoutSeconds = 900;
};

// The service in use: setImageTo3DService's, else NEURACRAFT_3D_URL, else "" (none).
std::string imageTo3DService();
void setImageTo3DService(const std::string& url);
// Whether the service answers its health check (false without one).
bool imageTo3DAvailable();
// The model, its glTF kept as the service sent it (Mesh::glb; saved as is to .glb). Throws
// std::runtime_error without a service, or with the service's message when it fails.
Mesh imageTo3D(const Image& picture, const ImageTo3DOptions& options = {});

} // namespace nc

#endif // NEURACRAFT_IMAGE_TO_3D_H
