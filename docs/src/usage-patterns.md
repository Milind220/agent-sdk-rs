# Usage Patterns

This section is tuned for fast copy/paste and common agent runtime pitfalls.

## Pattern: Explicit Completion for Autonomous Loops

Use `require_done_tool(true)` when agents should not stop just because the model emits plain text.

```rust,no_run
use agent_sdk_rs::{Agent, AnthropicModel};
use agent_sdk_rs::tools::claude_code::all_tools;

# async fn run() -> Result<(), Box<dyn std::error::Error>> {
let model = AnthropicModel::from_env("claude-sonnet-4-5")?;
let mut agent = Agent::builder()
    .model(model)
    .tools(all_tools())
    .require_done_tool(true)
    .max_iterations(64)
    .build()?;

let _ = agent.query("Inspect repository and return risks").await?;
# Ok(())
# }
```

## Pattern: Keep Tool Inputs Strict

Use `additionalProperties: false` and `required` fields in each tool schema.

Why:

- prevents silent typo args
- clearer model contract
- safer retries

## Common Pitfalls

| Pitfall | Symptom | Fix |
|---|---|---|
| No explicit stop tool for autonomous runs | early/ambiguous completion | add `done` tool + `require_done_tool(true)` |
| Loose tool schema | tool receives malformed args | tighten JSON schema + required keys |
| No iteration cap | infinite tool loops | set `max_iterations` |
| Mixed provider-specific assumptions | adapter swap breaks behavior | stay inside `ChatModel` + shared `Model*` types |

## Evidence

- `done` + max-iteration behavior: [`src/agent/tests.rs`](https://github.com/Milind220/agent-sdk-rs/blob/main/src/agent/tests.rs)
- argument validation logic: [`src/tools/mod.rs`](https://github.com/Milind220/agent-sdk-rs/blob/main/src/tools/mod.rs)
