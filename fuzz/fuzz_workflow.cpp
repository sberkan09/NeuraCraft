#include "Workflow.h"

#include <cstdint>
#include <stdexcept>

// Workflow JSON as a user would send it: checked, checked against limits, file names mapped.
extern "C" int LLVMFuzzerTestOneInput(const std::uint8_t* data, std::size_t size) {
    const std::string text(reinterpret_cast<const char*>(data), size);
    try {
        nc::checkWorkflow(text);
    } catch (const std::exception&) {
    }
    nc::WorkflowLimits limits;
    limits.readRoots = {"/nonexistent/in"};
    limits.writeRoot = "/nonexistent/out";
    limits.maxPixels = 1 << 20;
    limits.maxFrames = 64;
    limits.maxSteps = 16;
    limits.maxDenoiseSteps = 50;
    try {
        nc::checkWorkflowLimits(text, "/nonexistent", limits);
    } catch (const std::exception&) {
    }
    try {
        nc::mapWorkflowFiles(text, [](const nc::WorkflowField&, const std::string& v) { return "/cache/" + v; });
    } catch (const std::exception&) {
    }
    return 0;
}
