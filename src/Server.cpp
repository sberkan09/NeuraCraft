#include "Server.h"

#include "Backend.h"
#include "Image.h"
#include "Json.h"
#include "Parallel.h"
#include "Workflow.h"

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <deque>
#include <filesystem>
#include <fstream>
#include <list>
#include <map>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <thread>
#include <vector>

#ifndef _WIN32
#include <arpa/inet.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <unistd.h>
#endif

namespace nc {

#ifdef _WIN32

struct StudioServer::Impl {};
StudioServer::StudioServer(ServerOptions, std::function<void(const std::string&)>) {}
StudioServer::~StudioServer() = default;
void StudioServer::start() { throw std::runtime_error("NeuraCraft serve: not supported on Windows yet"); }
void StudioServer::stop() {}
void StudioServer::wait() {}
int StudioServer::port() const { return 0; }

#else

namespace {

using json::makeArray;
using json::makeBool;
using json::makeNumber;
using json::makeObject;
using json::makeString;

double nowSeconds() {
    return std::chrono::duration<double>(std::chrono::system_clock::now().time_since_epoch()).count();
}

// Writes all of data; false when the peer went away.
bool sendAll(int fd, const char* data, std::size_t n) {
    while (n > 0) {
#ifdef MSG_NOSIGNAL
        const ssize_t w = ::send(fd, data, n, MSG_NOSIGNAL);
#else
        const ssize_t w = ::send(fd, data, n, 0);
#endif
        if (w <= 0) return false;
        data += w;
        n -= static_cast<std::size_t>(w);
    }
    return true;
}
bool sendAll(int fd, const std::string& s) { return sendAll(fd, s.data(), s.size()); }

struct Request {
    std::string method, path, query, body;
    std::map<std::string, std::string> headers;  // lower-case names
    std::string header(const std::string& name) const {
        auto it = headers.find(name);
        return it == headers.end() ? std::string() : it->second;
    }
};

struct Response {
    int status = 200;
    std::string type = "application/json";
    std::string body;
    std::vector<std::pair<std::string, std::string>> headers;
};

const char* statusText(int s) {
    switch (s) {
    case 200: return "OK";
    case 204: return "No Content";
    case 400: return "Bad Request";
    case 403: return "Forbidden";
    case 404: return "Not Found";
    case 405: return "Method Not Allowed";
    case 409: return "Conflict";
    case 413: return "Payload Too Large";
    case 415: return "Unsupported Media Type";
    default: return "Internal Server Error";
    }
}

Response jsonResponse(const json::Value& v, int status = 200) {
    Response r;
    r.status = status;
    r.body = json::dump(v);
    return r;
}
Response errorResponse(int status, const std::string& message) { return jsonResponse(makeObject({{"error", makeString(message)}}), status); }

// Reads one request; false on a malformed or oversized one.
bool readRequest(int fd, Request& req) {
    std::string data;
    char buf[8192];
    std::size_t headerEnd = std::string::npos;
    while ((headerEnd = data.find("\r\n\r\n")) == std::string::npos) {
        if (data.size() > 65536) return false;
        const ssize_t r = ::recv(fd, buf, sizeof buf, 0);
        if (r <= 0) return false;
        data.append(buf, static_cast<std::size_t>(r));
    }
    std::istringstream head(data.substr(0, headerEnd));
    std::string line;
    if (!std::getline(head, line)) return false;
    if (!line.empty() && line.back() == '\r') line.pop_back();
    std::istringstream first(line);
    std::string target, version;
    first >> req.method >> target >> version;
    if (req.method.empty() || target.empty() || target[0] != '/') return false;
    const std::size_t q = target.find('?');
    req.path = target.substr(0, q);
    if (q != std::string::npos) req.query = target.substr(q + 1);
    while (std::getline(head, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        const std::size_t colon = line.find(':');
        if (colon == std::string::npos) continue;
        std::string name = line.substr(0, colon), value = line.substr(colon + 1);
        std::transform(name.begin(), name.end(), name.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        value.erase(0, value.find_first_not_of(" \t"));
        req.headers[name] = value;
    }
    std::size_t length = 0;
    const std::string cl = req.header("content-length");
    if (!cl.empty()) {
        try {
            length = static_cast<std::size_t>(std::stoull(cl));
        } catch (const std::exception&) {
            return false;
        }
    }
    if (length > (std::size_t(64) << 20)) return false;
    req.body = data.substr(headerEnd + 4);
    while (req.body.size() < length) {
        const ssize_t r = ::recv(fd, buf, sizeof buf, 0);
        if (r <= 0) return false;
        req.body.append(buf, static_cast<std::size_t>(r));
    }
    req.body.resize(length);
    return true;
}

// Origins allowed to call the API from a browser: the machine itself.
bool localOrigin(const std::string& origin) {
    for (const char* p : {"http://localhost", "http://127.0.0.1", "http://[::1]"}) {
        const std::size_t n = std::strlen(p);
        if (origin.compare(0, n, p) == 0 && (origin.size() == n || origin[n] == ':')) return true;
    }
    return false;
}

std::string contentType(const std::string& path) {
    const std::string ext = std::filesystem::path(path).extension().string();
    if (ext == ".html") return "text/html; charset=utf-8";
    if (ext == ".js" || ext == ".mjs") return "text/javascript; charset=utf-8";
    if (ext == ".css") return "text/css; charset=utf-8";
    if (ext == ".json") return "application/json";
    if (ext == ".svg") return "image/svg+xml";
    if (ext == ".png") return "image/png";
    if (ext == ".jpg" || ext == ".jpeg") return "image/jpeg";
    if (ext == ".gif") return "image/gif";
    if (ext == ".woff2") return "font/woff2";
    if (ext == ".ico") return "image/x-icon";
    return "application/octet-stream";
}

std::vector<std::string> split(const std::string& path) {
    std::vector<std::string> parts;
    std::string cur;
    for (char c : path) {
        if (c == '/') {
            if (!cur.empty()) parts.push_back(cur);
            cur.clear();
        } else {
            cur += c;
        }
    }
    if (!cur.empty()) parts.push_back(cur);
    return parts;
}

std::string urlDecode(const std::string& s) {
    std::string out;
    for (std::size_t i = 0; i < s.size(); ++i) {
        if (s[i] == '%' && i + 2 < s.size() && std::isxdigit(static_cast<unsigned char>(s[i + 1])) &&
            std::isxdigit(static_cast<unsigned char>(s[i + 2]))) {
            out += static_cast<char>(std::stoi(s.substr(i + 1, 2), nullptr, 16));
            i += 2;
        } else {
            out += s[i] == '+' ? ' ' : s[i];
        }
    }
    return out;
}

std::string base64Decode(const std::string& in) {
    static int table[256];
    static bool ready = false;
    if (!ready) {
        for (int& t : table) t = -1;
        const char* a = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
        for (int i = 0; i < 64; ++i) table[static_cast<unsigned char>(a[i])] = i;
        table[static_cast<unsigned char>('-')] = 62;  // URL-safe alphabet too
        table[static_cast<unsigned char>('_')] = 63;
        ready = true;
    }
    std::string out;
    out.reserve(in.size() * 3 / 4);
    unsigned acc = 0;
    int bits = 0;
    for (unsigned char c : in) {
        if (c == '=') break;
        const int v = table[c];
        if (v < 0) continue;  // whitespace, line breaks
        acc = (acc << 6) | static_cast<unsigned>(v);
        bits += 6;
        if (bits >= 8) {
            bits -= 8;
            out += static_cast<char>((acc >> bits) & 0xFF);
        }
    }
    return out;
}

struct OutputInfo {
    std::size_t step = 0;
    std::string op, clip;
    std::size_t frames = 0, width = 0, height = 0;
    double seconds = 0.0;
};

struct Job {
    int id = 0;
    std::string workflow;
    std::string state = "queued";  // queued, running, done, failed, cancelled
    std::size_t step = 0, steps = 0;
    std::string op, stage, error;
    double fraction = -1.0;
    double queued = 0.0, started = 0.0, finished = 0.0;
    std::vector<OutputInfo> outputs;
    std::map<std::string, Clip> clips;  // dropped for older finished jobs
    Image preview;
    std::size_t previews = 0;
    std::atomic<bool> cancel{false};
};

json::Value jobJson(const Job& j) {
    std::vector<json::Value> outs;
    for (const OutputInfo& o : j.outputs)
        outs.push_back(makeObject({{"step", makeNumber(static_cast<double>(o.step))},
                                   {"op", makeString(o.op)},
                                   {"clip", makeString(o.clip)},
                                   {"frames", makeNumber(static_cast<double>(o.frames))},
                                   {"width", makeNumber(static_cast<double>(o.width))},
                                   {"height", makeNumber(static_cast<double>(o.height))},
                                   {"seconds", makeNumber(o.seconds)},
                                   {"url", makeString(o.clip.empty() ? "" : "/api/jobs/" + std::to_string(j.id) + "/clips/" + o.clip + "/")}}));
    return makeObject({{"id", makeNumber(j.id)},
                       {"state", makeString(j.state)},
                       {"step", makeNumber(static_cast<double>(j.step))},
                       {"steps", makeNumber(static_cast<double>(j.steps))},
                       {"op", makeString(j.op)},
                       {"stage", makeString(j.stage)},
                       {"fraction", makeNumber(j.fraction)},
                       {"error", makeString(j.error)},
                       {"queued", makeNumber(j.queued)},
                       {"started", makeNumber(j.started)},
                       {"finished", makeNumber(j.finished)},
                       {"previews", makeNumber(static_cast<double>(j.previews))},
                       {"clipsKept", makeBool(!j.clips.empty())},
                       {"outputs", makeArray(outs)}});
}

} // namespace

struct StudioServer::Impl {
    ServerOptions opt;
    std::function<void(const std::string&)> log;
    int listenFd = -1;
    int boundPort = 0;
    std::atomic<bool> running{false};
    std::thread acceptThread, workerThread, pingThread;

    std::mutex mutex;  // jobs, queue, clients
    std::condition_variable queueCv, stopCv;
    std::map<int, std::shared_ptr<Job>> jobs;
    std::deque<int> queue;
    int nextId = 1;
    std::unique_ptr<WorkflowResultCache> results;  // shared by all jobs (has its own lock)
    int nextUpload = 1;
    std::list<int> clients;  // event-stream sockets
    std::mutex sendMutex;    // serializes writes to clients

    std::atomic<int> openConnections{0};

    void say(const std::string& s) {
        if (log) log(s);
    }

    // ---- Events ----
    void broadcast(const std::string& type, const json::Value& data) {
        const std::string msg = "event: " + type + "\ndata: " + json::dump(data) + "\n\n";
        std::lock_guard<std::mutex> lock(sendMutex);
        std::vector<int> dead;
        std::vector<int> fds;
        {
            std::lock_guard<std::mutex> l(mutex);
            fds.assign(clients.begin(), clients.end());
        }
        for (int fd : fds)
            if (!sendAll(fd, msg)) dead.push_back(fd);
        if (!dead.empty()) {
            std::lock_guard<std::mutex> l(mutex);
            for (int fd : dead) {
                clients.remove(fd);
                ::close(fd);
            }
        }
    }
    bool hasClients() {
        std::lock_guard<std::mutex> l(mutex);
        return !clients.empty();
    }

    // ---- Jobs ----
    void worker() {
        while (true) {
            std::shared_ptr<Job> job;
            {
                std::unique_lock<std::mutex> lock(mutex);
                queueCv.wait(lock, [&] { return !running || !queue.empty(); });
                if (!running) return;
                job = jobs[queue.front()];
                queue.pop_front();
                if (job->cancel) {
                    job->state = "cancelled";
                    job->finished = nowSeconds();
                    continue;
                }
                job->state = "running";
                job->started = nowSeconds();
            }
            broadcast("job", jobSnapshot(*job));
            run(*job);
            broadcast("job", jobSnapshot(*job));
            trimJobs();
        }
    }

    json::Value jobSnapshot(Job& j) {
        std::lock_guard<std::mutex> l(mutex);
        return jobJson(j);
    }

    void run(Job& job) {
        WorkflowEvents ev;
        const int id = job.id;
        ev.log = [&](const std::string& line) {
            say("job " + std::to_string(id) + ": " + line);
            broadcast("log", makeObject({{"job", makeNumber(id)}, {"line", makeString(line)}}));
        };
        auto lastProgress = std::make_shared<double>(0.0);
        ev.progress = [&, lastProgress](std::size_t step, std::size_t steps, const std::string& op, const std::string& stage, double f) {
            {
                std::lock_guard<std::mutex> l(mutex);
                job.step = step;
                job.steps = steps;
                job.op = op;
                job.stage = stage;
                job.fraction = f;
            }
            broadcast("progress", makeObject({{"job", makeNumber(id)},
                                              {"step", makeNumber(static_cast<double>(step))},
                                              {"steps", makeNumber(static_cast<double>(steps))},
                                              {"op", makeString(op)},
                                              {"stage", makeString(stage)},
                                              {"fraction", makeNumber(f)}}));
        };
        if (hasClients())
            ev.preview = [&](std::size_t step, const Image& picture) {
                std::size_t n;
                {
                    std::lock_guard<std::mutex> l(mutex);
                    job.preview = picture;
                    n = ++job.previews;
                }
                broadcast("preview", makeObject({{"job", makeNumber(id)},
                                                 {"step", makeNumber(static_cast<double>(step))},
                                                 {"url", makeString("/api/jobs/" + std::to_string(id) + "/preview.jpg?n=" + std::to_string(n))}}));
            };
        ev.output = [&](std::size_t step, const std::string& op, const std::string& clip, const Clip& frames, double seconds) {
            OutputInfo o;
            o.step = step;
            o.op = op;
            o.clip = clip;
            o.frames = frames.size();
            if (!frames.empty()) {
                o.width = frames.front().width;
                o.height = frames.front().height;
            }
            o.seconds = seconds;
            {
                std::lock_guard<std::mutex> l(mutex);
                job.outputs.push_back(o);
                if (!clip.empty()) job.clips[clip] = frames;
            }
            broadcast("output", makeObject({{"job", makeNumber(id)},
                                            {"step", makeNumber(static_cast<double>(step))},
                                            {"op", makeString(op)},
                                            {"clip", makeString(clip)},
                                            {"frames", makeNumber(static_cast<double>(o.frames))},
                                            {"seconds", makeNumber(seconds)},
                                            {"url", makeString(clip.empty() ? "" : "/api/jobs/" + std::to_string(id) + "/clips/" + clip + "/")}}));
        };
        ev.cancelled = [&] { return job.cancel.load() || !running.load(); };
        std::string state = "done", error;
        try {
            runWorkflow(job.workflow, opt.baseDir, ev, results.get());
        } catch (const WorkflowCancelled&) {
            state = "cancelled";
        } catch (const std::exception& e) {
            state = "failed";
            error = e.what();
        }
        std::lock_guard<std::mutex> l(mutex);
        job.state = state;
        job.error = error;
        job.finished = nowSeconds();
    }

    // Older finished jobs give their clips back (their summaries stay).
    void trimJobs() {
        std::vector<json::Value> trimmed;  // told to the clients, so they stop showing those frames
        {
            std::lock_guard<std::mutex> l(mutex);
            std::size_t kept = 0;
            for (auto it = jobs.rbegin(); it != jobs.rend(); ++it) {
                Job& j = *it->second;
                if (j.state == "queued" || j.state == "running") continue;
                if (++kept > opt.keepJobs && !j.clips.empty()) {
                    j.clips.clear();
                    j.preview = Image();
                    trimmed.push_back(jobJson(j));
                }
            }
        }
        for (const json::Value& v : trimmed) broadcast("job", v);
    }

    // ---- HTTP ----
    Response handle(const Request& req) {
        const std::vector<std::string> p = split(req.path);
        const bool get = req.method == "GET" || req.method == "HEAD", post = req.method == "POST";
        if (post) {
            const std::string type = req.header("content-type");
            if (type.rfind("application/json", 0) != 0) return errorResponse(415, "POST bodies must be application/json");
        }
        if (p.size() >= 1 && p[0] == "api") {
            if (p.size() == 2 && p[1] == "info" && get) {
                return jsonResponse(makeObject({{"name", makeString("NeuraCraft Studio engine")},
                                                {"api", makeNumber(1)},
                                                {"device", makeString(deviceDescription())},
                                                {"gpu", makeString(gpuName())},
                                                {"gpuAttention", makeBool(gpuAttentionEnabled())},
                                                {"threads", makeNumber(static_cast<double>(threadCount()))},
                                                {"dataDir", makeString(opt.dataDir)},
                                                {"baseDir", makeString(std::filesystem::absolute(opt.baseDir).string())}}));
            }
            if (p.size() >= 2 && p[1] == "cache") {
                if (p.size() == 3 && p[2] == "clear" && post) {
                    if (results) results->clear();
                } else if (!(p.size() == 2 && get)) {
                    return errorResponse(404, "not found");
                }
                return jsonResponse(makeObject({{"enabled", makeBool(results != nullptr)},
                                                {"entries", makeNumber(results ? static_cast<double>(results->size()) : 0)},
                                                {"bytes", makeNumber(results ? static_cast<double>(results->bytes()) : 0)},
                                                {"hits", makeNumber(results ? static_cast<double>(results->hits()) : 0)}}));
            }
            if (p.size() == 2 && p[1] == "ops" && get) {
                Response r;
                r.body = workflowSchemaJson();
                return r;
            }
            if (p.size() == 2 && p[1] == "models" && get) {
                Response r;
                r.body = modelsJson(findModels(opt.dataDir));
                return r;
            }
            if (p.size() == 2 && p[1] == "check" && post) {
                std::vector<json::Value> problems;
                for (const std::string& s : checkWorkflow(req.body)) problems.push_back(makeString(s));
                return jsonResponse(makeObject({{"problems", makeArray(problems)}}));
            }
            if (p.size() == 2 && p[1] == "upload" && post) {
                // {"name": "photo.png", "data": "<base64 PNG>"} -> saved under <base>/uploads, path for workflows.
                const json::Value body = json::parse(req.body);
                const json::Value* data = body.find("data");
                if (!data || !data->isString()) return errorResponse(400, "upload needs \"data\" (base64 PNG)");
                const std::string bytes = base64Decode(data->string);
                static const unsigned char sig[8] = {0x89, 'P', 'N', 'G', '\r', '\n', 0x1A, '\n'};
                if (bytes.size() < 8 || std::memcmp(bytes.data(), sig, 8) != 0) return errorResponse(400, "upload: not a PNG");
                std::string name = body.find("name") && body.at("name").isString() ? body.at("name").string : "upload";
                std::string clean;
                for (char c : std::filesystem::path(name).stem().string())
                    clean += (std::isalnum(static_cast<unsigned char>(c)) || c == '-' || c == '_') ? c : '_';
                if (clean.empty()) clean = "upload";
                if (clean.size() > 40) clean.resize(40);
                const std::filesystem::path dir = std::filesystem::path(opt.baseDir) / "uploads";
                std::filesystem::create_directories(dir);
                std::string file;
                {
                    std::lock_guard<std::mutex> l(mutex);
                    file = std::to_string(static_cast<long long>(nowSeconds() * 1000)) + "-" + std::to_string(nextUpload++) + "-" + clean + ".png";
                }
                std::ofstream out(dir / file, std::ios::binary);
                out.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
                if (!out) return errorResponse(500, "upload: cannot write " + (dir / file).string());
                return jsonResponse(makeObject({{"path", makeString("uploads/" + file)}}));
            }
            if (p.size() == 2 && p[1] == "run" && post) {
                const std::vector<std::string> problems = checkWorkflow(req.body);
                if (!problems.empty()) {
                    std::vector<json::Value> list;
                    for (const std::string& s : problems) list.push_back(makeString(s));
                    return jsonResponse(makeObject({{"error", makeString("workflow has problems")}, {"problems", makeArray(list)}}), 400);
                }
                auto job = std::make_shared<Job>();
                job->workflow = req.body;
                job->queued = nowSeconds();
                {
                    std::lock_guard<std::mutex> l(mutex);
                    job->id = nextId++;
                    jobs[job->id] = job;
                    queue.push_back(job->id);
                }
                queueCv.notify_all();
                broadcast("job", jobSnapshot(*job));
                return jsonResponse(makeObject({{"job", makeNumber(job->id)}}));
            }
            if (p.size() >= 2 && p[1] == "jobs") {
                if (p.size() == 2 && get) {
                    std::vector<json::Value> list;
                    std::lock_guard<std::mutex> l(mutex);
                    for (auto it = jobs.rbegin(); it != jobs.rend(); ++it) list.push_back(jobJson(*it->second));
                    return jsonResponse(makeArray(list));
                }
                std::shared_ptr<Job> job;
                try {
                    std::lock_guard<std::mutex> l(mutex);
                    auto it = jobs.find(std::stoi(p[2]));
                    if (it != jobs.end()) job = it->second;
                } catch (const std::exception&) {
                }
                if (!job) return errorResponse(404, "no such job");
                if (p.size() == 3 && get) return jsonResponse(jobSnapshot(*job));
                if (p.size() == 4 && p[3] == "cancel" && post) {
                    job->cancel = true;
                    return jsonResponse(makeObject({{"job", makeNumber(job->id)}, {"cancelling", makeBool(true)}}));
                }
                if (p.size() == 4 && p[3] == "preview.jpg" && get) {
                    Image im;
                    {
                        std::lock_guard<std::mutex> l(mutex);
                        im = job->preview;
                    }
                    if (im.pixels.empty()) return errorResponse(404, "no preview yet");
                    const auto jpg = encodeJpeg(toRgb(im), 85);
                    Response r;
                    r.type = "image/jpeg";
                    r.body.assign(jpg.begin(), jpg.end());
                    r.headers.emplace_back("Cache-Control", "no-store");
                    return r;
                }
                if (p.size() == 6 && p[3] == "clips" && get) {
                    const std::string clip = urlDecode(p[4]), file = p[5];
                    const std::size_t dot = file.rfind('.');
                    if (dot == std::string::npos || file.substr(dot) != ".jpg") return errorResponse(404, "frames are .jpg");
                    std::size_t frame = 0;
                    try {
                        frame = static_cast<std::size_t>(std::stoul(file.substr(0, dot)));
                    } catch (const std::exception&) {
                        return errorResponse(404, "bad frame number");
                    }
                    Image im;
                    {
                        std::lock_guard<std::mutex> l(mutex);
                        auto it = job->clips.find(clip);
                        if (it == job->clips.end()) return errorResponse(404, "no clip named " + clip + " (kept for the last jobs only)");
                        if (frame >= it->second.size()) return errorResponse(404, "frame out of range");
                        im = it->second[frame];
                    }
                    std::size_t maxSide = 0;
                    if (req.query.rfind("max=", 0) == 0) maxSide = static_cast<std::size_t>(std::atoi(req.query.c_str() + 4));
                    im = toRgb(im);
                    if (maxSide > 0 && std::max(im.width, im.height) > maxSide) {
                        const double s = static_cast<double>(maxSide) / static_cast<double>(std::max(im.width, im.height));
                        im = resizeImage(im, std::max<std::size_t>(1, static_cast<std::size_t>(im.width * s)),
                                         std::max<std::size_t>(1, static_cast<std::size_t>(im.height * s)));
                    }
                    const auto jpg = encodeJpeg(im, 90);
                    Response r;
                    r.type = "image/jpeg";
                    r.body.assign(jpg.begin(), jpg.end());
                    r.headers.emplace_back("Cache-Control", "max-age=3600");
                    return r;
                }
            }
            return errorResponse(get || post ? 404 : 405, "no such endpoint: " + req.method + " " + req.path);
        }
        // Static UI files.
        if (!get) return errorResponse(405, "method not allowed");
        if (opt.uiDir.empty()) {
            Response r;
            r.type = "text/plain; charset=utf-8";
            r.body = "NeuraCraft Studio engine. API under /api (see Server.h); start with --ui <folder> to serve the web UI.\n";
            return r;
        }
        for (const std::string& part : p)
            if (part == ".." || part.find('\\') != std::string::npos) return errorResponse(403, "bad path");
        std::filesystem::path file = std::filesystem::path(opt.uiDir);
        for (const std::string& part : p) file /= urlDecode(part);
        std::error_code ec;
        if (p.empty() || std::filesystem::is_directory(file, ec)) file /= "index.html";
        if (!std::filesystem::is_regular_file(file, ec)) file = std::filesystem::path(opt.uiDir) / "index.html";  // single-page app routes
        std::ifstream in(file, std::ios::binary);
        if (!in) return errorResponse(404, "not found");
        std::stringstream ss;
        ss << in.rdbuf();
        Response r;
        r.type = contentType(file.string());
        r.body = ss.str();
        return r;
    }

    void serveConnection(int fd) {
        Request req;
        if (!readRequest(fd, req)) {
            ::close(fd);
            return;
        }
        const std::string origin = req.header("origin");
        if (!origin.empty() && !localOrigin(origin)) {
            const Response r = errorResponse(403, "requests from " + origin + " are not allowed");
            writeResponse(fd, req, r, "");
            ::close(fd);
            return;
        }
        const std::string allow = origin.empty() ? "" : origin;
        if (req.method == "OPTIONS") {
            Response r;
            r.status = 204;
            r.type = "";
            r.headers.emplace_back("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
            r.headers.emplace_back("Access-Control-Allow-Headers", "Content-Type");
            r.headers.emplace_back("Access-Control-Max-Age", "600");
            writeResponse(fd, req, r, allow);
            ::close(fd);
            return;
        }
        if (req.method == "GET" && req.path == "/api/events") {
            std::string head = "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-store\r\nConnection: keep-alive\r\n";
            if (!allow.empty()) head += "Access-Control-Allow-Origin: " + allow + "\r\nVary: Origin\r\n";
            head += "\r\n: connected\n\n";
            {
                std::lock_guard<std::mutex> lock(sendMutex);
                if (!sendAll(fd, head)) {
                    ::close(fd);
                    return;
                }
                std::lock_guard<std::mutex> l(mutex);
                clients.push_back(fd);  // the broadcaster owns it from here
            }
            return;
        }
        Response r;
        try {
            r = handle(req);
        } catch (const std::exception& e) {
            r = errorResponse(500, e.what());
        }
        writeResponse(fd, req, r, allow);
        ::close(fd);
    }

    void writeResponse(int fd, const Request& req, const Response& r, const std::string& allowOrigin) {
        std::string head = "HTTP/1.1 " + std::to_string(r.status) + " " + statusText(r.status) + "\r\n";
        if (!r.type.empty()) head += "Content-Type: " + r.type + "\r\n";
        head += "Content-Length: " + std::to_string(r.body.size()) + "\r\nConnection: close\r\n";
        if (!allowOrigin.empty()) head += "Access-Control-Allow-Origin: " + allowOrigin + "\r\nVary: Origin\r\n";
        for (const auto& h : r.headers) head += h.first + ": " + h.second + "\r\n";
        head += "\r\n";
        if (!sendAll(fd, head)) return;
        if (req.method != "HEAD") sendAll(fd, r.body);
    }

    void acceptLoop() {
        while (running) {
            sockaddr_in addr{};
            socklen_t len = sizeof addr;
            const int fd = ::accept(listenFd, reinterpret_cast<sockaddr*>(&addr), &len);
            if (fd < 0) {
                if (!running) break;
                continue;
            }
#ifdef SO_NOSIGPIPE
            int one = 1;
            ::setsockopt(fd, SOL_SOCKET, SO_NOSIGPIPE, &one, sizeof one);
#endif
            timeval tv{30, 0};
            ::setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof tv);
            ++openConnections;
            std::thread([this, fd] {
                serveConnection(fd);
                --openConnections;
            }).detach();
        }
    }

    void pingLoop() {
        std::unique_lock<std::mutex> lock(mutex);
        while (running) {
            stopCv.wait_for(lock, std::chrono::seconds(15));
            if (!running) break;
            lock.unlock();
            // A comment line keeps idle event streams (and proxies) open and finds closed ones.
            std::lock_guard<std::mutex> s(sendMutex);
            std::vector<int> fds, dead;
            {
                std::lock_guard<std::mutex> l(mutex);
                fds.assign(clients.begin(), clients.end());
            }
            for (int fd : fds)
                if (!sendAll(fd, ": ping\n\n")) dead.push_back(fd);
            {
                std::lock_guard<std::mutex> l(mutex);
                for (int fd : dead) {
                    clients.remove(fd);
                    ::close(fd);
                }
            }
            lock.lock();
        }
    }
};

StudioServer::StudioServer(ServerOptions options, std::function<void(const std::string&)> log) : impl_(std::make_unique<Impl>()) {
    impl_->opt = std::move(options);
    impl_->log = std::move(log);
    if (impl_->opt.cacheBytes) impl_->results = std::make_unique<WorkflowResultCache>(impl_->opt.cacheBytes);
}

StudioServer::~StudioServer() { stop(); }

void StudioServer::start() {
    Impl& m = *impl_;
    if (m.running) return;
    m.listenFd = ::socket(AF_INET, SOCK_STREAM, 0);
    if (m.listenFd < 0) throw std::runtime_error("serve: cannot create a socket");
    int one = 1;
    ::setsockopt(m.listenFd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof one);
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(static_cast<std::uint16_t>(m.opt.port));
    const std::string host = m.opt.host == "localhost" ? "127.0.0.1" : m.opt.host;
    if (::inet_pton(AF_INET, host.c_str(), &addr.sin_addr) != 1) {
        ::close(m.listenFd);
        throw std::invalid_argument("serve: host must be an IPv4 address, not " + m.opt.host);
    }
    if (::bind(m.listenFd, reinterpret_cast<sockaddr*>(&addr), sizeof addr) != 0 || ::listen(m.listenFd, 64) != 0) {
        ::close(m.listenFd);
        throw std::runtime_error("serve: cannot listen on " + m.opt.host + ":" + std::to_string(m.opt.port) + " (" +
                                 std::strerror(errno) + ")");
    }
    socklen_t len = sizeof addr;
    ::getsockname(m.listenFd, reinterpret_cast<sockaddr*>(&addr), &len);
    m.boundPort = ntohs(addr.sin_port);
    m.running = true;
    m.acceptThread = std::thread([&m] { m.acceptLoop(); });
    m.workerThread = std::thread([&m] { m.worker(); });
    m.pingThread = std::thread([&m] { m.pingLoop(); });
    m.say("NeuraCraft Studio engine on http://" + m.opt.host + ":" + std::to_string(m.boundPort));
}

void StudioServer::stop() {
    Impl& m = *impl_;
    if (!m.running.exchange(false)) return;
    {
        std::lock_guard<std::mutex> l(m.mutex);
        for (auto& kv : m.jobs) kv.second->cancel = true;
    }
    m.queueCv.notify_all();
    m.stopCv.notify_all();
    ::shutdown(m.listenFd, SHUT_RDWR);
    ::close(m.listenFd);
    if (m.acceptThread.joinable()) m.acceptThread.join();
    if (m.workerThread.joinable()) m.workerThread.join();
    if (m.pingThread.joinable()) m.pingThread.join();
    {
        std::lock_guard<std::mutex> s(m.sendMutex);
        std::lock_guard<std::mutex> l(m.mutex);
        for (int fd : m.clients) {
            ::shutdown(fd, SHUT_RDWR);
            ::close(fd);
        }
        m.clients.clear();
    }
    // Requests still being answered finish on their own threads; give them a moment.
    for (int i = 0; i < 200 && m.openConnections > 0; ++i) std::this_thread::sleep_for(std::chrono::milliseconds(10));
}

void StudioServer::wait() {
    Impl& m = *impl_;
    std::unique_lock<std::mutex> lock(m.mutex);
    m.stopCv.wait(lock, [&] { return !m.running.load(); });
}

int StudioServer::port() const { return impl_->boundPort; }

#endif

} // namespace nc
