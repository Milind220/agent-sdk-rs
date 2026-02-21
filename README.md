# agent-sdk-rs

`agent-sdk-rs` is a lightweight Rust agent framework inspired by browser-use's SDK design.

It is intentionally minimal:
- small surface area
- explicit control flow
- simple tool integration
- easy embedding into Rust binaries

This project is expected to stay lightweight by default. New features should preserve that core philosophy.

## What It Is

A Rust SDK for tool-using agents with:
- an `Agent` loop
- `query` + `query_stream`
- provider adapter boundary
- tool execution + dependency injection
- explicit completion semantics via `done`

## Coverage (v0.1.0 alpha)

Implemented:
- Anthropic provider adapter (`anthropic-ai-sdk`)
- `Agent` + builder API
- `query` and `query_stream`
- event stream model (`MessageStart`, `StepStart`, `ToolCall`, `ToolResult`, `FinalResponse`, etc.)
- tool registration with JSON schema
- dependency map + dependency overrides
- translated Claude-code-style tool set:
  - `bash`, `read`, `write`, `edit`
  - `glob_search`, `grep`
  - `todo_read`, `todo_write`
  - `done`
- optional `claude_code` binary target

Out of scope right now:
- non-Anthropic providers
- Laminar integration

## Roadmap

Near-term:
1. Add xAI Grok provider adapter (next)
2. Keep adapter trait stable across providers
3. Improve docs/examples while keeping core small

Non-goal:
- turning this into a heavy orchestration framework

## Install

Local path:

```toml
[dependencies]
agent-sdk-rs = { path = "." }
```

## Quick Usage

### 1. Basic agent query

```rust
use agent_sdk_rs::{Agent, AnthropicModel};

let model = AnthropicModel::from_env("claude-sonnet-4-5")?;
let mut agent = Agent::builder().model(model).build()?;

let answer = agent.query("Hello").await?;
println!("{answer}");
# Ok::<(), Box<dyn std::error::Error>>(())
```

### 2. Streaming events

```rust
use agent_sdk_rs::{Agent, AgentEvent, AnthropicModel};
use futures_util::StreamExt;

let model = AnthropicModel::from_env("claude-sonnet-4-5")?;
let mut agent = Agent::builder().model(model).build()?;

let stream = agent.query_stream("Solve this step by step");
futures_util::pin_mut!(stream);
while let Some(event) = stream.next().await {
    match event? {
        AgentEvent::ToolCall { tool, .. } => println!("tool: {tool}"),
        AgentEvent::FinalResponse { content } => println!("final: {content}"),
        _ => {}
    }
}
# Ok::<(), Box<dyn std::error::Error>>(())
```

### 3. Claude-code tool pack

```rust
use agent_sdk_rs::tools::claude_code::{SandboxContext, all_tools};
use agent_sdk_rs::{Agent, AnthropicModel};

let model = AnthropicModel::from_env("claude-sonnet-4-5")?;
let sandbox = SandboxContext::create::<std::path::PathBuf>(None)?;

let mut agent = Agent::builder()
    .model(model)
    .tools(all_tools())
    .dependency(sandbox)
    .require_done_tool(true)
    .build()?;
# Ok::<(), Box<dyn std::error::Error>>(())
```

## Optional Binary

Run the fun Claude-code-like binary:

```bash
cargo run --features claude-code --bin claude_code -- "list Rust files and summarize"
```

Environment:
- `ANTHROPIC_API_KEY` required
- `ANTHROPIC_MODEL` optional (default set in binary)
- `CLAUDE_CODE_SANDBOX` optional

## Examples

```bash
cargo run --example local_loop
cargo run --example di_override
```

## License

MIT. See `LICENSE`.
