#include "Checkpoint.h"

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>

namespace nc {

namespace {

namespace fs = std::filesystem;
using json::makeArray;
using json::makeBool;
using json::makeNumber;
using json::makeObject;
using json::makeString;

const std::string kUnet = "model.diffusion_model.";
const std::string kControl = "control_model.";                 // a ControlNet (lllyasviel's control_v11*, ...)
const std::string kVae = "first_stage_model.";
const std::string kClip = "cond_stage_model.transformer.";     // OpenAI CLIP (SD 1.x)
const std::string kOpenClip = "cond_stage_model.model.";       // OpenCLIP (SD 2.x)

bool startsWith(const std::string& s, const std::string& p) { return s.compare(0, p.size(), p) == 0; }

std::string readText(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("cannot open " + path);
    std::ostringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

// Element counts work whatever the stored shapes (some tools flatten them): an output size is a bias's
// or norm's length, an input size the weight's count divided by that.
std::size_t count(const SafeTensors& st, const std::string& name) {
    if (!st.has(name)) throw std::runtime_error("checkpoint: no tensor " + name);
    return st.info(name).elements();
}

// Largest N for which `prefix` N `suffix` names a tensor (keys like "...input_blocks.11.0..."), or -1.
int maxIndex(const std::vector<std::string>& names, const std::string& prefix) {
    int best = -1;
    for (const std::string& n : names) {
        if (!startsWith(n, prefix)) continue;
        std::size_t i = prefix.size(), j = i;
        while (j < n.size() && std::isdigit(static_cast<unsigned char>(n[j]))) ++j;
        if (j > i && j < n.size() && n[j] == '.') best = std::max(best, std::stoi(n.substr(i, j - i)));
    }
    return best;
}

bool anyWithPrefix(const std::vector<std::string>& names, const std::string& prefix) {
    auto it = std::lower_bound(names.begin(), names.end(), prefix);
    return it != names.end() && startsWith(*it, prefix);
}

// ---- UNet ----

// The denoiser's prefix: the UNet of a checkpoint, or a ControlNet file's control model.
std::string unetPrefix(const SafeTensors& st) {
    if (st.has(kUnet + "input_blocks.0.0.weight")) return kUnet;
    if (st.has(kControl + "input_blocks.0.0.weight")) return kControl;
    return kUnet;
}

struct UnetLayout {
    std::size_t levels = 0, layers = 0;  // resolution levels, resnets per down block
};

UnetLayout unetLayout(const std::vector<std::string>& names, const std::string& kUnet) {
    UnetLayout l;
    const int inputs = maxIndex(names, kUnet + "input_blocks.") + 1;  // 1 + levels * layers + (levels - 1)
    const int outputs = maxIndex(names, kUnet + "output_blocks.") + 1;  // levels * (layers + 1); none in a ControlNet
    const bool control = kUnet == kControl;
    if (inputs <= 1 || (outputs <= 0 && !control)) throw std::runtime_error("checkpoint: no UNet blocks");
    // inputs = levels * (layers + 1), outputs = levels * (layers + 1): levels from the downsamplers.
    std::size_t down = 0;
    for (int i = 1; i < inputs; ++i)
        if (anyWithPrefix(names, kUnet + "input_blocks." + std::to_string(i) + ".0.op.")) ++down;
    l.levels = down + 1;
    l.layers = static_cast<std::size_t>(inputs) / l.levels - 1;
    if (l.levels * (l.layers + 1) != static_cast<std::size_t>(inputs) || (!control && static_cast<std::size_t>(outputs) != l.levels * (l.layers + 1)))
        throw std::runtime_error("checkpoint: unexpected UNet block layout");
    return l;
}

std::string resnetPart(const std::string& rest) {
    static const std::vector<std::pair<std::string, std::string>> map = {
        {"in_layers.0.", "norm1."}, {"in_layers.2.", "conv1."}, {"emb_layers.1.", "time_emb_proj."},
        {"out_layers.0.", "norm2."}, {"out_layers.3.", "conv2."}, {"skip_connection.", "conv_shortcut."}};
    for (const auto& [from, to] : map)
        if (startsWith(rest, from)) return to + rest.substr(from.size());
    return "";
}

std::vector<SafeTensors::Rename> unetRenames(const SafeTensors& st) {
    const std::vector<std::string> names = st.names();
    const std::string kUnet = unetPrefix(st);
    const UnetLayout l = unetLayout(names, kUnet);
    std::vector<SafeTensors::Rename> out;
    auto add = [&](const std::string& to, const std::string& from) { out.push_back({to, from}); };
    // "<block>.<k>.<rest>" -> index k and rest
    auto split = [](const std::string& s, std::size_t& k, std::string& rest) {
        const std::size_t dot = s.find('.');
        if (dot == std::string::npos) return false;
        k = static_cast<std::size_t>(std::stoul(s.substr(0, dot)));
        rest = s.substr(dot + 1);
        return true;
    };
    for (const std::string& n : names) {
        if (!startsWith(n, kUnet)) continue;
        const std::string key = n.substr(kUnet.size());
        std::string to;
        if (startsWith(key, "time_embed.0.")) to = "time_embedding.linear_1." + key.substr(13);
        else if (startsWith(key, "time_embed.2.")) to = "time_embedding.linear_2." + key.substr(13);
        else if (startsWith(key, "out.0.")) to = "conv_norm_out." + key.substr(6);
        else if (startsWith(key, "out.2.")) to = "conv_out." + key.substr(6);
        else if (startsWith(key, "input_blocks.0.0.")) to = "conv_in." + key.substr(17);
        else if (startsWith(key, "input_blocks.")) {
            std::size_t i, part;
            std::string rest, inner;
            if (!split(key.substr(13), i, rest) || !split(rest, part, inner)) continue;
            const std::size_t b = (i - 1) / (l.layers + 1), j = (i - 1) % (l.layers + 1);
            const std::string p = "down_blocks." + std::to_string(b) + ".";
            if (j == l.layers) {
                if (startsWith(inner, "op.")) to = p + "downsamplers.0.conv." + inner.substr(3);
            } else if (part == 0) {
                const std::string r = resnetPart(inner);
                if (!r.empty()) to = p + "resnets." + std::to_string(j) + "." + r;
            } else if (part == 1) {
                to = p + "attentions." + std::to_string(j) + "." + inner;
            }
        } else if (startsWith(key, "middle_block.")) {
            std::size_t part;
            std::string inner;
            if (!split(key.substr(13), part, inner)) continue;
            if (part == 1) to = "mid_block.attentions.0." + inner;
            else {
                const std::string r = resnetPart(inner);
                if (!r.empty()) to = "mid_block.resnets." + std::to_string(part / 2) + "." + r;
            }
        } else if (startsWith(key, "output_blocks.")) {
            std::size_t i, part;
            std::string rest, inner;
            if (!split(key.substr(14), i, rest) || !split(rest, part, inner)) continue;
            const std::size_t b = i / (l.layers + 1), j = i % (l.layers + 1);
            const std::string p = "up_blocks." + std::to_string(b) + ".";
            const std::string here = kUnet + "output_blocks." + std::to_string(i) + "." + std::to_string(part) + ".";
            if (part == 0) {
                const std::string r = resnetPart(inner);
                if (!r.empty()) to = p + "resnets." + std::to_string(j) + "." + r;
            } else if (startsWith(inner, "conv.") && !anyWithPrefix(names, here + "proj_in.")) {
                to = p + "upsamplers.0.conv." + inner.substr(5);  // the block's upsampler (after the attention, if any)
            } else if (part == 1) {
                to = p + "attentions." + std::to_string(j) + "." + inner;
            }
        }
        else if (startsWith(key, "input_hint_block.")) {
            // conv, SiLU, conv, SiLU, ...: convs 0, 2, ..., 12 then the zero conv 14 -> conv_in, blocks.0-5, conv_out.
            std::size_t k;
            std::string inner;
            if (split(key.substr(17), k, inner) && k % 2 == 0) {
                const std::size_t last = static_cast<std::size_t>(maxIndex(names, kUnet + "input_hint_block."));
                to = k == 0 ? "controlnet_cond_embedding.conv_in." + inner
                     : k == last ? "controlnet_cond_embedding.conv_out." + inner
                                 : "controlnet_cond_embedding.blocks." + std::to_string(k / 2 - 1) + "." + inner;
            }
        } else if (startsWith(key, "zero_convs.")) {
            std::size_t k;
            std::string inner;
            if (split(key.substr(11), k, inner) && startsWith(inner, "0.")) to = "controlnet_down_blocks." + std::to_string(k) + "." + inner.substr(2);
        } else if (startsWith(key, "middle_block_out.0.")) {
            to = "controlnet_mid_block." + key.substr(19);
        }
        if (!to.empty()) add(to, n);
    }
    return out;
}

json::Value unetConfig(const SafeTensors& st, const CheckpointInfo& info) {
    const std::vector<std::string> names = st.names();
    const std::string kUnet = unetPrefix(st);
    const bool control = kUnet == kControl;
    const UnetLayout l = unetLayout(names, kUnet);
    std::vector<json::Value> channels, down, up, heads;
    for (std::size_t b = 0; b < l.levels; ++b) {
        const std::string first = kUnet + "input_blocks." + std::to_string(1 + b * (l.layers + 1)) + ".";
        const std::size_t ch = count(st, first + "0.out_layers.3.bias");
        channels.push_back(makeNumber(static_cast<double>(ch)));
        const bool attn = anyWithPrefix(names, first + "1.proj_in.");
        down.push_back(makeString(attn ? "CrossAttnDownBlock2D" : "DownBlock2D"));
        // diffusers' attention_head_dim is the number of heads: 8 for SD 1.x, 64-channel heads for SD 2.x.
        heads.push_back(makeNumber(info.family == "sd1" ? 8.0 : static_cast<double>(std::max<std::size_t>(1, ch / 64))));
    }
    for (std::size_t b = 0; b < l.levels && !control; ++b) {
        const bool attn = anyWithPrefix(names, kUnet + "output_blocks." + std::to_string(b * (l.layers + 1)) + ".1.proj_in.");
        up.push_back(makeString(attn ? "CrossAttnUpBlock2D" : "UpBlock2D"));
    }
    const std::string projIn = kUnet + "input_blocks.1.1.proj_in.weight";
    const bool linearProj = st.has(projIn) && st.info(projIn).shape.size() == 2;
    json::Value config = makeObject({{"_class_name", makeString(control ? "ControlNetModel" : "UNet2DConditionModel")},
                       {"in_channels", makeNumber(static_cast<double>(info.inChannels))},
                       {"out_channels", makeNumber(control ? 4.0 : static_cast<double>(count(st, kUnet + "out.2.bias")))},
                       {"block_out_channels", makeArray(channels)},
                       {"layers_per_block", makeNumber(static_cast<double>(l.layers))},
                       {"down_block_types", makeArray(down)},
                       {"attention_head_dim", makeArray(heads)},
                       {"cross_attention_dim", makeNumber(static_cast<double>(info.contextDim))},
                       {"use_linear_projection", makeBool(linearProj)},
                       {"norm_num_groups", makeNumber(32)},
                       {"norm_eps", makeNumber(1e-5)},
                       {"flip_sin_to_cos", makeBool(true)},
                       {"freq_shift", makeNumber(0)}});
    if (control) {
        // The hint convs: conv_in, then pairs (same size, stride 2) up to the last before the zero conv.
        const int last = maxIndex(names, kUnet + "input_hint_block.");
        std::vector<json::Value> cc;
        for (int k = 0; k < last; k += 4) cc.push_back(makeNumber(static_cast<double>(count(st, kUnet + "input_hint_block." + std::to_string(k) + ".bias"))));
        config.object.emplace_back("conditioning_embedding_out_channels", makeArray(cc));
        config.object.emplace_back("conditioning_channels",
                                   makeNumber(static_cast<double>(count(st, kUnet + "input_hint_block.0.weight") / (count(st, kUnet + "input_hint_block.0.bias") * 9))));
    } else {
        config.object.emplace_back("up_block_types", makeArray(up));
    }
    return config;
}

// ---- VAE ----

std::string vaePart(const std::string& rest) {
    if (startsWith(rest, "nin_shortcut.")) return "conv_shortcut." + rest.substr(13);
    return rest;  // norm1, conv1, norm2, conv2
}
std::string vaeAttention(const std::string& rest) {
    static const std::vector<std::pair<std::string, std::string>> map = {
        {"norm.", "group_norm."}, {"q.", "to_q."}, {"k.", "to_k."}, {"v.", "to_v."}, {"proj_out.", "to_out.0."}};
    for (const auto& [from, to] : map)
        if (startsWith(rest, from)) return to + rest.substr(from.size());
    return "";
}

// The VAE's prefix: first_stage_model. inside a checkpoint, none in a VAE file of its own.
std::string vaePrefix(const SafeTensors& st) { return st.has(kVae + "encoder.conv_in.weight") ? kVae : ""; }

std::vector<SafeTensors::Rename> vaeRenames(const SafeTensors& st) {
    const std::vector<std::string> names = st.names();
    const std::string kVae = vaePrefix(st);
    const int ups = maxIndex(names, kVae + "decoder.up.") + 1;
    std::vector<SafeTensors::Rename> out;
    for (const std::string& n : names) {
        if (!startsWith(n, kVae)) continue;
        const std::string key = n.substr(kVae.size());
        std::string to;
        for (const char* half : {"encoder.", "decoder."}) {
            if (!startsWith(key, half)) continue;
            const std::string h = half, rest = key.substr(h.size());
            if (startsWith(rest, "conv_in.") || startsWith(rest, "conv_out.")) to = key;
            else if (startsWith(rest, "norm_out.")) to = h + "conv_norm_out." + rest.substr(9);
            else if (startsWith(rest, "mid.block_1.")) to = h + "mid_block.resnets.0." + vaePart(rest.substr(12));
            else if (startsWith(rest, "mid.block_2.")) to = h + "mid_block.resnets.1." + vaePart(rest.substr(12));
            else if (startsWith(rest, "mid.attn_1.")) {
                const std::string a = vaeAttention(rest.substr(11));
                if (!a.empty()) to = h + "mid_block.attentions.0." + a;
            } else if (startsWith(rest, "down.") || startsWith(rest, "up.")) {
                const bool isDown = startsWith(rest, "down.");
                const std::string r = rest.substr(isDown ? 5 : 3);
                const std::size_t dot = r.find('.');
                std::size_t b = static_cast<std::size_t>(std::stoul(r.substr(0, dot)));
                if (!isDown) b = static_cast<std::size_t>(ups - 1) - b;  // the original counts decoder levels from the bottom
                const std::string inner = r.substr(dot + 1);
                const std::string p = h + (isDown ? "down_blocks." : "up_blocks.") + std::to_string(b) + ".";
                if (startsWith(inner, "block.")) {
                    const std::size_t d2 = inner.find('.', 6);
                    to = p + "resnets." + inner.substr(6, d2 - 6) + "." + vaePart(inner.substr(d2 + 1));
                } else if (startsWith(inner, "downsample.conv.")) {
                    to = p + "downsamplers.0.conv." + inner.substr(16);
                } else if (startsWith(inner, "upsample.conv.")) {
                    to = p + "upsamplers.0.conv." + inner.substr(14);
                }
            }
        }
        if (startsWith(key, "quant_conv.") || startsWith(key, "post_quant_conv.")) to = key;
        if (startsWith(key, "decoder.") || startsWith(key, "encoder.") || startsWith(key, "quant_conv.") || startsWith(key, "post_quant_conv."))
            if (!to.empty()) out.push_back({to, n});
    }
    return out;
}

json::Value vaeConfig(const SafeTensors& st) {
    const std::vector<std::string> names = st.names();
    const std::string kVae = vaePrefix(st);
    const int levels = maxIndex(names, kVae + "encoder.down.") + 1;
    if (levels <= 0) throw std::runtime_error("checkpoint: no VAE encoder blocks");
    std::vector<json::Value> channels;
    for (int b = 0; b < levels; ++b)
        channels.push_back(makeNumber(static_cast<double>(count(st, kVae + "encoder.down." + std::to_string(b) + ".block.0.conv1.bias"))));
    const int layers = maxIndex(names, kVae + "encoder.down.0.block.") + 1;
    const std::size_t inC = count(st, kVae + "encoder.conv_in.weight") / (count(st, kVae + "encoder.conv_in.bias") * 9);
    return makeObject({{"_class_name", makeString("AutoencoderKL")},
                       {"in_channels", makeNumber(static_cast<double>(inC))},
                       {"out_channels", makeNumber(static_cast<double>(count(st, kVae + "decoder.conv_out.bias")))},
                       {"latent_channels", makeNumber(static_cast<double>(count(st, kVae + "post_quant_conv.bias")))},
                       {"block_out_channels", makeArray(channels)},
                       {"layers_per_block", makeNumber(static_cast<double>(layers))},
                       {"norm_num_groups", makeNumber(32)},
                       {"act_fn", makeString("silu")},
                       {"scaling_factor", makeNumber(0.18215)}});
}

// ---- CLIP text encoder ----

std::vector<SafeTensors::Rename> clipRenames(const SafeTensors& st) {
    std::vector<SafeTensors::Rename> out;
    const std::vector<std::string> names = st.names();
    if (anyWithPrefix(names, kClip)) {  // Hugging Face layout already, under a prefix (older files lack "text_model.")
        for (const std::string& n : names) {
            if (!startsWith(n, kClip)) continue;
            std::string key = n.substr(kClip.size());
            if (!startsWith(key, "text_model.")) key = "text_model." + key;
            out.push_back({key, n});
        }
        return out;
    }
    // OpenCLIP: fused attention projections, other names.
    const std::size_t D = count(st, kOpenClip + "ln_final.weight");
    const std::string t = "text_model.";
    for (const std::string& n : names) {
        if (!startsWith(n, kOpenClip)) continue;
        const std::string key = n.substr(kOpenClip.size());
        if (key == "token_embedding.weight") out.push_back({t + "embeddings.token_embedding.weight", n});
        else if (key == "positional_embedding") out.push_back({t + "embeddings.position_embedding.weight", n});
        else if (startsWith(key, "ln_final.")) out.push_back({t + "final_layer_norm." + key.substr(9), n});
        else if (startsWith(key, "transformer.resblocks.")) {
            const std::string r = key.substr(22);
            const std::size_t dot = r.find('.');
            const std::string p = t + "encoder.layers." + r.substr(0, dot) + ".", inner = r.substr(dot + 1);
            if (inner == "attn.in_proj_weight" || inner == "attn.in_proj_bias") {
                const bool weight = inner == "attn.in_proj_weight";
                const std::size_t part = weight ? D * D : D;  // q, k, v stacked by rows
                const char* qkv[] = {"q_proj", "k_proj", "v_proj"};
                for (std::size_t k = 0; k < 3; ++k)
                    out.push_back({p + "self_attn." + qkv[k] + (weight ? ".weight" : ".bias"), n, k * part, part});
            } else {
                static const std::vector<std::pair<std::string, std::string>> map = {
                    {"ln_1.", "layer_norm1."}, {"ln_2.", "layer_norm2."}, {"attn.out_proj.", "self_attn.out_proj."},
                    {"mlp.c_fc.", "mlp.fc1."}, {"mlp.c_proj.", "mlp.fc2."}};
                for (const auto& [from, to] : map)
                    if (startsWith(inner, from)) out.push_back({p + to + inner.substr(from.size()), n});
            }
        }
    }
    return out;
}

json::Value clipConfig(const SafeTensors& st) {
    const std::vector<std::string> names = st.names();
    auto config = [](const char* act, std::size_t vocabElems, std::size_t D, std::size_t intermediate, std::size_t layers, std::size_t posElems) {
        return makeObject({{"hidden_act", makeString(act)},
                           {"vocab_size", makeNumber(static_cast<double>(vocabElems / D))},
                           {"hidden_size", makeNumber(static_cast<double>(D))},
                           {"num_attention_heads", makeNumber(static_cast<double>(std::max<std::size_t>(1, D / 64)))},
                           {"intermediate_size", makeNumber(static_cast<double>(intermediate))},
                           {"num_hidden_layers", makeNumber(static_cast<double>(layers))},
                           {"max_position_embeddings", makeNumber(static_cast<double>(posElems / D))},
                           {"layer_norm_eps", makeNumber(1e-5)}});
    };
    if (anyWithPrefix(names, kClip)) {
        const std::string p = st.has(kClip + "text_model.embeddings.token_embedding.weight") ? kClip + "text_model." : kClip;
        const std::size_t D = count(st, p + "final_layer_norm.weight");
        return config("quick_gelu", count(st, p + "embeddings.token_embedding.weight"), D, count(st, p + "encoder.layers.0.mlp.fc1.bias"),
                      static_cast<std::size_t>(maxIndex(names, p + "encoder.layers.") + 1), count(st, p + "embeddings.position_embedding.weight"));
    }
    const std::size_t D = count(st, kOpenClip + "ln_final.weight");
    const int layers = maxIndex(names, kOpenClip + "transformer.resblocks.") + 1;
    // Stable Diffusion 2 conditions on the penultimate layer; diffusers' SD 2 text encoder simply has one layer less.
    return config("gelu", count(st, kOpenClip + "token_embedding.weight"), D, count(st, kOpenClip + "transformer.resblocks.0.mlp.c_fc.bias"),
                  static_cast<std::size_t>(std::max(1, layers - 1)), count(st, kOpenClip + "positional_embedding"));
}

// Settings a file cannot tell (or that differ from the usual ones), from "<file>.json" next to it:
// {"unet": {...}, "vae": {...}, "text_encoder": {...}, "scheduler": {...}}; keys given replace the derived ones.
json::Value sidecar(const std::string& model) {
    const fs::path p = fs::path(model).replace_extension(".json");
    if (!fs::exists(p)) return makeObject({});
    json::Value v = json::parse(readText(p.string()));
    if (!v.isObject()) throw std::runtime_error(p.string() + ": expected a JSON object");
    return v;
}
void overlay(json::Value& config, const json::Value& extra) {
    if (!extra.isObject()) return;
    for (const auto& [k, v] : extra.object) {
        bool replaced = false;
        for (auto& [ck, cv] : config.object)
            if (ck == k) {
                cv = v;
                replaced = true;
            }
        if (!replaced) config.object.emplace_back(k, v);
    }
}

std::string firstExisting(const std::vector<std::string>& paths) {
    for (const std::string& p : paths)
        if (std::ifstream(p)) return p;
    return "";
}

} // namespace

bool isSingleFileCheckpoint(const std::string& path) {
    std::error_code ec;
    if (!fs::is_regular_file(path, ec) || fs::path(path).extension() != ".safetensors") return false;
    try {
        const SafeTensors st(path);
        return st.has(kUnet + "input_blocks.0.0.weight");
    } catch (const std::exception&) {
        return false;
    }
}

CheckpointInfo inspectCheckpoint(const std::string& path) {
    const SafeTensors st(path);
    const std::vector<std::string> names = st.names();
    const std::string kUnet = unetPrefix(st);
    if (!st.has(kUnet + "input_blocks.0.0.weight"))
        throw std::runtime_error(path + ": not a Stable Diffusion checkpoint (no " + kUnet + "input_blocks.0.0.weight)");
    CheckpointInfo info;
    info.controlNet = kUnet == kControl;
    info.inChannels = count(st, kUnet + "input_blocks.0.0.weight") / (count(st, kUnet + "input_blocks.0.0.bias") * 9);
    for (const std::string& n : names) {
        const std::size_t at = n.find(".attn2.to_k.weight");
        if (startsWith(n, kUnet) && at != std::string::npos && at + 18 == n.size()) {
            // to_k maps the text (contextDim) to the block's channels (its norm's length).
            const std::string block = n.substr(0, n.rfind(".attn2."));
            info.contextDim = count(st, n) / count(st, block + ".norm1.weight");
            break;
        }
    }
    info.hasVae = st.has(kVae + "encoder.conv_in.weight");
    const bool openAi = anyWithPrefix(names, kClip), openClip = anyWithPrefix(names, kOpenClip);
    info.hasTextEncoder = openAi || openClip;
    if (const json::Value extra = sidecar(path); extra.find("family") && extra.at("family").isString()) info.family = extra.at("family").string;
    else if (anyWithPrefix(names, "conditioner.embedders.") || info.contextDim == 2048) info.family = "sdxl";
    else if (openClip || (!openAi && info.contextDim == 1024)) info.family = "sd2";
    else if (openAi || info.contextDim == 768) info.family = "sd1";
    else info.family = "unknown";
    return info;
}

std::vector<SafeTensors::Rename> checkpointRenames(const SafeTensors& file, const std::string& component) {
    if (component == "unet") return unetRenames(file);
    if (component == "vae") return vaeRenames(file);
    if (component == "text_encoder") return clipRenames(file);
    throw std::invalid_argument("checkpoint: unknown component '" + component + "' (unet, vae, text_encoder)");
}

ModelComponent modelComponent(const std::string& model, const std::string& component, const std::string& what) {
    ModelComponent c;
    std::error_code ec;
    if (fs::is_regular_file(model, ec) && component == "vae") {
        const SafeTensors st(model);
        if (!st.has(kUnet + "input_blocks.0.0.weight") && st.has("encoder.conv_in.weight")) {  // a VAE file of its own
            c.config = vaeConfig(st);
            const json::Value extra = sidecar(model);
            if (const json::Value* o = extra.find("vae")) overlay(c.config, *o);
            c.weights = model + "#vae";
            return c;
        }
    }
    if (fs::is_regular_file(model, ec)) {
        const CheckpointInfo info = inspectCheckpoint(model);
        if (info.family != "sd1" && info.family != "sd2")
            throw std::runtime_error(what + ": " + model + " is a " + info.family +
                                     " checkpoint; single files of SD 1.x and 2.x are supported");
        const SafeTensors st(model);
        if (component == "unet") c.config = unetConfig(st, info);
        else if (component == "vae") {
            if (!info.hasVae) throw std::runtime_error(what + ": " + model + " has no VAE; pass one with \"vae\"");
            c.config = vaeConfig(st);
        } else if (component == "text_encoder") {
            if (!info.hasTextEncoder) throw std::runtime_error(what + ": " + model + " has no text encoder");
            c.config = clipConfig(st);
        } else {
            throw std::invalid_argument(what + ": unknown component " + component);
        }
        const json::Value extra = sidecar(model);
        if (const json::Value* o = extra.find(component)) overlay(c.config, *o);
        c.weights = model + "#" + component;
        return c;
    }
    // A diffusers folder: the component's subfolder, or the model folder itself holding config.json.
    c.folder = std::ifstream(model + "/config.json") ? model : model + "/" + component;
    const std::string config = c.folder + "/config.json";
    if (!std::ifstream(config)) throw std::runtime_error(what + ": no " + config + " (nor a single-file checkpoint at " + model + ")");
    c.config = json::parse(readText(config));
    const std::string base = component == "text_encoder" || (c.folder == model && fs::exists(model + "/model.safetensors"))
                                 ? "model"
                                 : "diffusion_pytorch_model";
    c.weights = firstExisting({c.folder + "/" + base + ".safetensors", c.folder + "/" + base + ".fp16.safetensors"});
    if (c.weights.empty()) throw std::runtime_error(what + ": no " + base + "(.fp16).safetensors in " + c.folder);
    return c;
}

std::string tokenizerDirectory(const std::string& model) {
    if (const char* env = std::getenv("NEURACRAFT_TOKENIZER"); env && *env) return env;
    std::error_code ec;
    if (!fs::is_regular_file(model, ec)) return model + "/tokenizer";
    auto ok = [](const fs::path& d) { return fs::exists(d / "vocab.json") && fs::exists(d / "merges.txt"); };
    // Next to the file, then in a sibling model folder, going up a few levels (e.g. data/checkpoints/x.safetensors
    // finds data/sd-turbo/tokenizer).
    fs::path dir = fs::path(model).parent_path();
    if (dir.empty()) dir = ".";
    const fs::path stem = fs::path(model).stem();
    for (const fs::path& d : {dir / (stem.string() + ".tokenizer"), dir / "tokenizer"})
        if (ok(d)) return d.string();
    for (int up = 0; up < 3 && !dir.empty(); ++up, dir = dir.parent_path()) {
        std::vector<fs::path> found;
        for (const auto& e : fs::directory_iterator(dir, ec))
            if (e.is_directory(ec) && ok(e.path() / "tokenizer")) found.push_back(e.path() / "tokenizer");
        if (!found.empty()) {
            std::sort(found.begin(), found.end());
            return found.front().string();
        }
        if (dir == dir.parent_path()) break;
    }
    throw std::runtime_error("no CLIP tokenizer for " + model + ": put vocab.json and merges.txt in " +
                             (fs::path(model).parent_path() / "tokenizer").string() +
                             " (any SD 1.x / 2.x tokenizer folder works) or set NEURACRAFT_TOKENIZER");
}

bool padsWithExclamation(const std::string& model) {
    std::error_code ec;
    if (fs::is_regular_file(model, ec)) {
        const json::Value extra = sidecar(model);
        if (const json::Value* t = extra.find("tokenizer"))
            if (const json::Value* p = t->find("pad_token"); p && p->isString()) return p->string == "!";
        return inspectCheckpoint(model).family == "sd2";
    }
    const std::string config = model + "/tokenizer/tokenizer_config.json";
    if (!std::ifstream(config)) return false;
    const json::Value c = json::parse(readText(config));
    const json::Value* p = c.find("pad_token");
    return p && p->isString() && p->string == "!";
}

std::string schedulerConfig(const std::string& model) {
    std::error_code ec;
    if (fs::is_regular_file(model, ec)) {
        // The schedule SD 1.x / 2.x were trained and released with (diffusers' default for them), with
        // "scheduler" from the sidecar on top, written where EulerScheduler can read it.
        json::Value sched = makeObject({{"num_train_timesteps", makeNumber(1000)},
                                        {"beta_start", makeNumber(0.00085)},
                                        {"beta_end", makeNumber(0.012)},
                                        {"beta_schedule", makeString("scaled_linear")},
                                        {"timestep_spacing", makeString("leading")},
                                        {"steps_offset", makeNumber(1)}});
        const json::Value extra = sidecar(model);
        if (const json::Value* o = extra.find("scheduler")) overlay(sched, *o);
        const std::string text = json::dump(sched);
        std::size_t h = std::hash<std::string>{}(model + text);
        const fs::path p = fs::temp_directory_path(ec) / ("neuracraft-scheduler-" + std::to_string(h) + ".json");
        if (!fs::exists(p)) std::ofstream(p) << text;
        return p.string();
    }
    const std::string config = model + "/scheduler/scheduler_config.json";
    return std::ifstream(config) ? config : "";
}

} // namespace nc
