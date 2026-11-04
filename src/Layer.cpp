#include "Layer.h"

namespace nc {

namespace {
thread_local int tlSkipInitDepth = 0;
} // namespace

SkipParameterInit::SkipParameterInit() { ++tlSkipInitDepth; }
SkipParameterInit::~SkipParameterInit() { --tlSkipInitDepth; }
bool SkipParameterInit::active() { return tlSkipInitDepth > 0; }

} // namespace nc
