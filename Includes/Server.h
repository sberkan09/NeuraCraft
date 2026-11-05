#ifndef NEURACRAFT_SERVER_H
#define NEURACRAFT_SERVER_H

// NeuraCraft Studio's engine: a small local HTTP server that runs workflows
// (Workflow.h) one after another and streams their progress.
//
//   GET  /api/info                      version, device, data folder
//   GET  /api/ops                       workflow operations and fields (workflowSchemaJson)
//   GET  /api/models                    models under the data folder (findModels)
//   POST /api/check                     {problems: [...]} for a workflow
//   POST /api/run                       queue a workflow -> {job: id}
//   POST /api/upload                    {name, data: base64 PNG} -> {path} (saved in <base>/uploads)
//   GET  /api/jobs, /api/jobs/<id>      job state, progress and outputs
//   POST /api/jobs/<id>/cancel
//   GET  /api/jobs/<id>/preview.jpg     latest live preview of a running job
//   GET  /api/jobs/<id>/clips/<clip>/<frame>.jpg
//   GET  /api/cache, POST /api/cache/clear   results kept for reuse (WorkflowResultCache)
//   GET  /api/events                    server-sent events: job, progress, preview, output, log
//   GET  /...                           files of the web UI folder (ServerOptions::uiDir), if set
//
// It listens on 127.0.0.1 unless told otherwise. Requests that change
// anything must be application/json, and a request from a web page is only
// answered for http://localhost or http://127.0.0.1 origins, so other sites
// cannot drive it from a browser. POSIX (Linux, macOS) only.

#include <functional>
#include <memory>
#include <string>

namespace nc {

struct ServerOptions {
    std::string host = "127.0.0.1";
    int port = 8188;          // 0 = any free port (see StudioServer::port)
    std::string dataDir = "data";
    std::string baseDir = ".";  // relative paths in workflows are taken from here
    std::string uiDir;          // static files served at /, if not empty
    std::size_t keepJobs = 8;   // finished jobs whose clips stay in memory
    // Step results reused by later jobs when a step and its inputs are unchanged; 0 turns it off.
    std::size_t cacheBytes = std::size_t(1) << 30;
};

class StudioServer {
public:
    explicit StudioServer(ServerOptions options, std::function<void(const std::string&)> log = {});
    ~StudioServer();  // stops it
    StudioServer(const StudioServer&) = delete;
    StudioServer& operator=(const StudioServer&) = delete;

    void start();         // binds and serves on background threads; throws if the port is taken
    void stop();          // cancels the running job, closes the sockets, joins the threads
    void wait();          // blocks until stop() (or a signal handler calling it)
    int port() const;     // the bound port

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace nc

#endif // NEURACRAFT_SERVER_H
