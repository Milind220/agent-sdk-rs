# Quickstart

## Install

```toml
[dependencies]
agent-sdk-rs = "0.1"
```

## Basic Query

```rust,no_run
use agent_sdk_rs::{Agent, AnthropicModel};

# async fn run() -> Result<(), Box<dyn std::error::Error>> {
let model = AnthropicModel::from_env("claude-sonnet-4-5")?;
let mut agent = Agent::builder().model(model).build()?;

let answer = agent.query("Summarize this repo in one line").await?;
println!("{answer}");
# Ok(())
# }
```

## Streaming Query

```rust,no_run
use agent_sdk_rs::{Agent, AgentEvent, GoogleModel};
use futures_util::StreamExt;

# async fn run() -> Result<(), Box<dyn std::error::Error>> {
let model = GoogleModel::from_env("gemini-2.5-flash")?;
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
# Ok(())
# }
```

## Environment Variables

- Anthropic: `ANTHROPIC_API_KEY`
- Google: `GOOGLE_API_KEY` or `GEMINI_API_KEY`
- xAI: `XAI_API_KEY` or `GROK_API_KEY`
