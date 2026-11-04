#include "Lora.h"
#include "Backend.h"
#include "SafeTensors.h"

#include <algorithm>
#include <stdexcept>

namespace nc {

namespace {
std::string underscored(std::string s) {
    std::replace(s.begin(), s.end(), '.', '_');
    return s;
}
bool startsWith(const std::string& s, const std::string& p) { return s.compare(0, p.size(), p) == 0; }
bool endsWith(const std::string& s, const std::string& p) {
    return s.size() >= p.size() && s.compare(s.size() - p.size(), p.size(), p) == 0;
}
} // namespace

LoraMerger::LoraMerger(const std::vector<Lora>& loras, const std::string& component) {
    const std::string kohya = component == "unet" ? "lora_unet_" : "lora_te_";
    const std::string peft = component + ".";
    for (const Lora& lora : loras) {
        const SafeTensors st(lora.file);
        std::size_t found = 0;
        for (const std::string& name : st.names()) {
            // The down / A half names the layer; the up / B half and alpha sit next to it.
            std::string module, upName, alphaName;
            if (startsWith(name, kohya) && endsWith(name, ".lora_down.weight")) {
                const std::string base = name.substr(0, name.size() - std::string(".lora_down.weight").size());
                module = base.substr(kohya.size());
                upName = base + ".lora_up.weight";
                alphaName = base + ".alpha";
            } else if (startsWith(name, peft) && endsWith(name, ".lora_A.weight")) {
                const std::string base = name.substr(0, name.size() - std::string(".lora_A.weight").size());
                module = underscored(base.substr(peft.size()));
                upName = base + ".lora_B.weight";
                alphaName = base + ".alpha";
            } else {
                continue;
            }
            if (!st.has(upName)) throw std::runtime_error("LoRA " + lora.file + ": " + name + " has no matching up weight");
            const auto& di = st.info(name);
            const auto& ui = st.info(upName);
            if (di.shape.size() < 2 || ui.shape.size() < 2) throw std::runtime_error("LoRA " + lora.file + ": bad shape for " + name);
            const std::size_t rank = di.shape[0], out = ui.shape[0];
            if (ui.elements() / out != rank) throw std::runtime_error("LoRA " + lora.file + ": " + upName + " is not 1x1 / rank " + std::to_string(rank));
            Delta d;
            const Matrix down = st.load(name), up = st.load(upName);
            d.down = Matrix(rank, di.elements() / rank, std::vector<float>(down.data(), down.data() + down.size()));
            d.up = Matrix(out, rank, std::vector<float>(up.data(), up.data() + up.size()));
            float alpha = static_cast<float>(rank);
            if (st.has(alphaName)) alpha = st.load(alphaName).data()[0];
            d.scale = lora.scale * alpha / static_cast<float>(rank);
            deltas_[module].push_back(std::move(d));
            ++total_;
            ++found;
        }
        (void)found;
    }
}

void LoraMerger::apply(const std::string& name, Matrix& weight) {
    if (deltas_.empty() || !endsWith(name, ".weight")) return;
    auto it = deltas_.find(underscored(name.substr(0, name.size() - 7)));
    if (it == deltas_.end()) return;
    for (Delta& d : it->second) {
        const std::size_t out = d.up.rows(), rank = d.up.cols(), in = d.down.cols();
        if (weight.size() != out * in)
            throw std::runtime_error("LoRA: " + name + " has " + std::to_string(weight.size()) + " values, the adapter makes " +
                                     std::to_string(out) + " x " + std::to_string(in));
        Matrix up = d.up;
        for (float& v : up.values()) v *= d.scale;
        // weight (out x in, row-major PyTorch layout) += up * down
        gemm(false, false, out, in, rank, up.data(), rank, d.down.data(), in, weight.data(), in, true);
        if (!d.used) ++merged_;
        d.used = true;
    }
}

std::vector<std::string> LoraMerger::unmerged() const {
    std::vector<std::string> names;
    for (const auto& [module, list] : deltas_)
        for (const Delta& d : list)
            if (!d.used) names.push_back(module);
    return names;
}

} // namespace nc
