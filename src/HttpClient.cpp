#include "HttpClient.h"

#include <cerrno>
#include <cstring>
#include <stdexcept>

#include <netdb.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <unistd.h>

namespace nc::http {

namespace {
struct Socket {
    int fd = -1;
    ~Socket() {
        if (fd >= 0) ::close(fd);
    }
};

std::string lower(std::string s) {
    for (char& c : s) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return s;
}
} // namespace

Response request(const std::string& method, const std::string& url, const std::string& path, const std::string& contentType,
                 const std::vector<std::uint8_t>& body, double timeoutSeconds, std::size_t maxBytes) {
    if (url.rfind("http://", 0) != 0) throw std::runtime_error("http: only http:// addresses are supported: " + url);
    std::string rest = url.substr(7);
    const std::size_t slash = rest.find('/');
    std::string base = slash == std::string::npos ? "" : rest.substr(slash);
    std::string hostPort = rest.substr(0, slash);
    while (!base.empty() && base.back() == '/') base.pop_back();
    std::string host = hostPort, port = "80";
    if (const std::size_t colon = hostPort.rfind(':'); colon != std::string::npos && hostPort.find(']') == std::string::npos) {
        host = hostPort.substr(0, colon);
        port = hostPort.substr(colon + 1);
    }
    if (host.empty()) throw std::runtime_error("http: no host in " + url);

    addrinfo hints{};
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;
    addrinfo* found = nullptr;
    if (const int e = getaddrinfo(host.c_str(), port.c_str(), &hints, &found); e != 0)
        throw std::runtime_error("http: cannot resolve " + host + ": " + gai_strerror(e));
    Socket s;
    std::string lastError = "no address";
    for (addrinfo* a = found; a; a = a->ai_next) {
        s.fd = ::socket(a->ai_family, a->ai_socktype, a->ai_protocol);
        if (s.fd < 0) continue;
        timeval tv{};
        tv.tv_sec = static_cast<long>(timeoutSeconds);
        tv.tv_usec = static_cast<long>((timeoutSeconds - static_cast<double>(tv.tv_sec)) * 1e6);
        setsockopt(s.fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
        setsockopt(s.fd, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv));
        if (::connect(s.fd, a->ai_addr, a->ai_addrlen) == 0) break;
        lastError = std::strerror(errno);
        ::close(s.fd);
        s.fd = -1;
    }
    freeaddrinfo(found);
    if (s.fd < 0) throw std::runtime_error("http: cannot connect to " + url + ": " + lastError);

    std::string head = method + " " + base + path + " HTTP/1.0\r\nHost: " + hostPort + "\r\nConnection: close\r\n";
    if (method == "POST") head += "Content-Type: " + contentType + "\r\nContent-Length: " + std::to_string(body.size()) + "\r\n";
    head += "\r\n";
    auto sendAll = [&](const char* p, std::size_t n) {
        while (n) {
            const ssize_t k = ::send(s.fd, p, n, MSG_NOSIGNAL);
            if (k <= 0) throw std::runtime_error(std::string("http: send failed: ") + (k == 0 ? "closed" : std::strerror(errno)));
            p += k;
            n -= static_cast<std::size_t>(k);
        }
    };
    sendAll(head.data(), head.size());
    if (method == "POST" && !body.empty()) sendAll(reinterpret_cast<const char*>(body.data()), body.size());

    std::vector<std::uint8_t> raw;
    char buf[1 << 16];
    for (;;) {
        const ssize_t k = ::recv(s.fd, buf, sizeof(buf), 0);
        if (k == 0) break;
        if (k < 0) {
            if (errno == EINTR) continue;
            throw std::runtime_error(errno == EAGAIN || errno == EWOULDBLOCK ? "http: no answer from " + url + " in time"
                                                                              : std::string("http: receive failed: ") + std::strerror(errno));
        }
        raw.insert(raw.end(), buf, buf + k);
        if (raw.size() > maxBytes + 65536) throw std::runtime_error("http: answer too large");
    }
    const std::string text(raw.begin(), raw.begin() + static_cast<long>(std::min<std::size_t>(raw.size(), 65536)));
    const std::size_t end = text.find("\r\n\r\n");
    if (end == std::string::npos || text.rfind("HTTP/", 0) != 0) throw std::runtime_error("http: malformed answer from " + url);
    Response r;
    const std::size_t sp = text.find(' ');
    try {
        r.status = std::stoi(text.substr(sp + 1, 3));
    } catch (const std::exception&) {
        throw std::runtime_error("http: malformed status from " + url);
    }
    bool chunked = false;
    long length = -1;
    std::size_t at = text.find("\r\n") + 2;
    while (at < end) {
        const std::size_t eol = text.find("\r\n", at);
        const std::string line = text.substr(at, eol - at);
        at = eol + 2;
        const std::size_t colon = line.find(':');
        if (colon == std::string::npos) continue;
        const std::string name = lower(line.substr(0, colon));
        std::string value = line.substr(colon + 1);
        while (!value.empty() && value.front() == ' ') value.erase(value.begin());
        if (name == "content-type") r.contentType = value;
        else if (name == "transfer-encoding" && lower(value).find("chunked") != std::string::npos) chunked = true;
        else if (name == "content-length") {
            try {
                length = std::stol(value);
            } catch (const std::exception&) {
                throw std::runtime_error("http: bad Content-Length from " + url);
            }
        }
    }
    std::vector<std::uint8_t> payload(raw.begin() + static_cast<long>(end + 4), raw.end());
    if (chunked) {
        std::size_t p = 0;
        for (;;) {
            std::size_t eol = p;
            while (eol + 1 < payload.size() && !(payload[eol] == '\r' && payload[eol + 1] == '\n')) ++eol;
            if (eol + 1 >= payload.size()) throw std::runtime_error("http: truncated chunked answer");
            std::size_t n = 0;
            try {
                n = std::stoul(std::string(payload.begin() + static_cast<long>(p), payload.begin() + static_cast<long>(eol)), nullptr, 16);
            } catch (const std::exception&) {
                throw std::runtime_error("http: bad chunk size");
            }
            p = eol + 2;
            if (n == 0) break;
            if (p + n > payload.size()) throw std::runtime_error("http: truncated chunked answer");
            r.body.insert(r.body.end(), payload.begin() + static_cast<long>(p), payload.begin() + static_cast<long>(p + n));
            p += n + 2;
        }
    } else {
        if (length >= 0) {
            if (payload.size() < static_cast<std::size_t>(length)) throw std::runtime_error("http: truncated answer from " + url);
            payload.resize(static_cast<std::size_t>(length));
        }
        r.body = std::move(payload);
    }
    return r;
}

} // namespace nc::http
