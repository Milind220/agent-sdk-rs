# agent-sdk-rs

Minimal Rust port of the browser-use agent SDK loop.

Current scope:
- Anthropic-first provider (`anthropic-ai-sdk`)
- Agent loop with explicit done semantics (`ToolOutcome::Done`)
- `query` + `query_stream`
- Tool registry with JSON-schema validation
- Dependency injection + dependency overrides
- Streaming event model with message + step events
- Translated Claude-code style tool pack (sandbox/file/search/todo/done)

Not included:
- Multi-provider adapters (beyond Anthropic)
- Laminar integration

## Install

```toml
[dependencies]
agent-sdk-rs = { path = "." }
```

## Core API

- `Agent::builder()`
- `Agent::query(...)`
- `Agent::query_stream(...)`
- `AgentConfig`
- `AgentEvent`
- `ToolSpec`
- `DependencyMap`
- `ToolOutcome::{Text, Done}`

## Streaming Events

`query_stream` emits:
- `MessageStart` / `MessageComplete`
- `HiddenUserMessage`
- `StepStart` / `StepComplete`
- `Thinking`
- `Text`
- `ToolCall`
- `ToolResult`
- `FinalResponse`

## Examples

```bash
cargo run --example local_loop
cargo run --example di_override
```

## Anthropic Setup

```bash
export ANTHROPIC_API_KEY=...
export ANTHROPIC_MODEL=claude-sonnet-4-5 # optional in binary example
```

Rust usage:

```rust
use agent_sdk_rs::AnthropicModel;

let model = AnthropicModel::from_env("claude-sonnet-4-5")?;
```

## Claude-Code Tool Pack

Translated toolset lives at:
- `agent_sdk_rs::tools::claude_code`

Includes:
- `bash`
- `read`
- `write`
- `edit`
- `glob_search`
- `grep`
- `todo_read`
- `todo_write`
- `done`

Convenience registration:

```rust
use agent_sdk_rs::tools::claude_code::{all_tools, SandboxContext};
```

## Optional Fun Binary (`claude_code`)

Build/run only when feature is enabled:

```bash
cargo run --features claude-code --bin claude_code -- "list files and summarize"
```

Optional env:
- `CLAUDE_CODE_SANDBOX=/path/to/sandbox`

## License

MIT (`LICENSE`).
