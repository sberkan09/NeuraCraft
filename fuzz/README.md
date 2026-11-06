# Fuzzing the file readers

A service runs workflows and models that other people send, so every reader of untrusted bytes is
fuzzed with libFuzzer under AddressSanitizer and UndefinedBehaviorSanitizer:

| target | reads |
|---|---|
| `fuzz_json` | `json::parse` / `json::dump` (configs, headers, workflows) |
| `fuzz_workflow` | `checkWorkflow`, `checkWorkflowLimits` (sandboxed), `mapWorkflowFiles` |
| `fuzz_png` | `loadImage` (PNG and PPM / PGM), with a 4 M pixel decode bound |
| `fuzz_safetensors` | the safetensors header and tensor conversion |
| `fuzz_onnx` | `onnx::Model::load` (protobuf, subgraphs, fusion passes) |
| `fuzz_tensorproto` | `onnx::loadTensorProto` |

```
fuzz/run.sh 60        # build with Clang (needs compiler-rt: libclang-rt-18-dev on Ubuntu), 60 s each
```

Seeds come from the test files and `fuzz/seeds/`; crashes are written to `fuzz/out/`. CI runs every
target for 30 s on each pull request. Running a model is not fuzzed: an ONNX graph can ask for any
amount of memory by design, so a service bounds that with the worker's memory and time limits.
