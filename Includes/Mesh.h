#ifndef NEURACRAFT_MESH_H
#define NEURACRAFT_MESH_H

// Triangle meshes: the 3D values of workflows. Read and written as glTF binary (.glb), Wavefront OBJ
// and PLY; drawn by a small CPU renderer (stills and turntables); made from a picture and its depth.

#include "Image.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace nc {

struct Mesh {
    std::vector<float> positions;         // x, y, z per vertex (y up, the front faces +z)
    std::vector<float> normals;           // x, y, z per vertex, or empty
    std::vector<float> uvs;               // u, v per vertex (v = 0 at the texture's top row), or empty
    std::vector<std::uint8_t> colors;     // r, g, b per vertex, or empty
    std::vector<std::uint32_t> indices;   // three per triangle
    Image texture;                        // base colour (RGB or RGBA) sampled through uvs; empty = none
    // The glTF binary this mesh was read from, when it came from one: saveMesh(.glb) writes it as it is,
    // so materials this struct does not hold (metallic / roughness / normal maps) survive.
    std::vector<std::uint8_t> glb;

    std::size_t vertexCount() const { return positions.size() / 3; }
    std::size_t triangleCount() const { return indices.size() / 3; }
    bool empty() const { return indices.empty(); }
    std::size_t bytes() const;
};

// Throws std::invalid_argument when the arrays disagree (sizes, indices out of range).
void checkMesh(const Mesh& mesh, const std::string& what = "mesh");
// Smooth vertex normals (area-weighted face normals).
void computeNormals(Mesh& mesh);

// The readers refuse files with more vertices or triangles than this (per thread; default 2^24 each).
void setMaxMeshElements(std::size_t elements);
std::size_t maxMeshElements();

// By extension: .glb, .obj (with a .mtl and a PNG next to it when there is a texture), .ply (binary,
// texture colours baked into vertex colours).
void saveMesh(const std::string& path, const Mesh& mesh);
// .glb (triangles of every mesh in the default scene, node transforms applied; the first material's
// PNG base colour texture), .obj (positions, uvs, normals, vertex colours), .ply (ASCII or binary).
Mesh loadMesh(const std::string& path);
bool isMeshFile(const std::string& path);  // by extension

std::vector<std::uint8_t> encodeGlb(const Mesh& mesh);
Mesh decodeGlb(const std::vector<std::uint8_t>& data, const std::string& name = "GLB data");
Mesh parseObj(const std::string& text, const std::string& name = "OBJ data");
Mesh decodePly(const std::vector<std::uint8_t>& data, const std::string& name = "PLY data");

// A view of a mesh: the camera orbits its centre at a distance that fits it in the frame.
struct MeshRenderOptions {
    std::size_t width = 512, height = 512;
    float yaw = 0.0f;           // degrees around the vertical axis (0: looking at the front, along -z)
    float pitch = 15.0f;        // degrees above the horizon
    float fov = 35.0f;          // vertical field of view, degrees
    float margin = 1.1f;        // > 1 leaves space around the mesh
    bool lighting = true;       // shade by a light near the camera; false: flat colours
    std::uint8_t background[3] = {255, 255, 255};
    std::size_t supersample = 2;  // rendered this many times larger, then averaged
};
Image renderMesh(const Mesh& mesh, const MeshRenderOptions& options = {});
// frames views turning `degrees` around the vertical axis from options.yaw.
std::vector<Image> renderTurntable(const Mesh& mesh, std::size_t frames, float degrees = 360.0f, const MeshRenderOptions& options = {});

// A relief mesh of a picture: a grid over it raised by its depth (relative inverse depth, larger =
// nearer, one value per pixel), textured with the picture. Triangles across depth edges are left out
// so foreground and background come apart instead of being joined by stretched walls.
struct DepthMeshOptions {
    std::size_t grid = 256;  // vertices along the longer side
    float depth = 0.4f;      // relief height, as a fraction of the picture's height
    float edges = 0.08f;     // largest depth step (of the 0..1 range) inside a triangle; 0 keeps all
};
Mesh meshFromDepth(const Image& picture, const std::vector<float>& depth, const DepthMeshOptions& options = {});

} // namespace nc

#endif // NEURACRAFT_MESH_H
