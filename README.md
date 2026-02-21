# rust-sdk

Minimal Rust agent SDK (browser-use style) with:
- `Agent` loop
- `query` and `query_stream`
- Tool calling + JSON schema validation
- Dependency injection + overrides
- Anthropic adapter (`anthropic-ai-sdk`)

## Install

```toml
[dependencies]
rust-sdk = { path = "./rust-sdk" }
```

## Core API

- `Agent::builder()`
- `Agent::query(...)`
- `Agent::query_stream(...)`
- `ToolSpec`
- `DependencyMap`
- `ToolOutcome::{Text, Done}`

## Run examples

```bash
cargo run --example local_loop
cargo run --example di_override
```

`local_loop` demonstrates one normal tool (`add`) and one `done` tool through both `query` and `query_stream`.

## Anthropic setup

```bash
export ANTHROPIC_API_KEY=...
```

Then construct model via:

```rust
use rust_sdk::{AnthropicModel, AnthropicModelConfig};

let model = AnthropicModel::from_env("claude-sonnet-4-5").unwrap();
// or AnthropicModel::new(AnthropicModelConfig::new(api_key, model_name))
```
