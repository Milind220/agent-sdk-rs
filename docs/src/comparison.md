# Comparison

| Capability | `agent-sdk-rs` | Abstraction-heavy frameworks |
|---|---|---|
| Loop visibility | explicit event stream | often hidden inside planners |
| Tooling | JSON-schema tool contracts | framework-specific wrappers |
| Completion control | optional explicit `done` path | often implicit stop logic |
| Provider swap | `ChatModel` adapter boundary | frequently runtime/provider coupled |
| Failure controls | retries + backoff + max iterations | varies per stack |

This crate stays intentionally small:

- easier to embed in existing binaries
- easier to reason about runtime behavior
- easier to test with deterministic mock providers
