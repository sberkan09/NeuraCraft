#ifndef NEURACRAFT_HTTP_CLIENT_H
#define NEURACRAFT_HTTP_CLIENT_H

// A minimal HTTP/1.0 client for plain http:// services next to the engine (no TLS, no redirects).

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace nc::http {

struct Response {
    int status = 0;
    std::string contentType;
    std::vector<std::uint8_t> body;
};

// method: GET or POST. url: http://host[:port][/base]; path is appended. Throws std::runtime_error
// on connection failures, timeouts, malformed answers or bodies over maxBytes.
Response request(const std::string& method, const std::string& url, const std::string& path, const std::string& contentType,
                 const std::vector<std::uint8_t>& body, double timeoutSeconds, std::size_t maxBytes = std::size_t(1) << 30);

} // namespace nc::http

#endif
