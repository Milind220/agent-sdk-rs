//! # agent-sdk-rs
//!
//! **Pure-Rust SDK for tool-using agents with explicit control flow.**
//! Minimal by design: one agent loop, explicit completion, and provider adapters with a shared interface.
//!
//! ## Why this crate?
//! | Capability | `agent-sdk-rs` | Typical abstraction-heavy frameworks | Why this helps agents |
//! |---|---|---|---|
//! | Agent core | Explicit loop in [`Agent::query_stream`] | Hidden planners / wrappers | Fewer moving parts, easier debugging |
//! | Action space | User-defined tools via [`ToolSpec`] JSON schema | Fixed or opinionated primitives | Start broad, then restrict by policy |
//! | Completion semantics | Optional explicit `done` via [`ToolOutcome::Done`] + [`AgentBuilder::require_done_tool`] | Implicit stop when no tool calls | Prevents premature "done" |
//! | Provider interface | One trait ([`ChatModel`]) and swappable adapters | Provider-specific runtime behavior | Swap models without rewriting agent logic |
//! | Reliability guards | Retries/backoff + max-iteration limit + schema validation | Often ad-hoc in app code | Safer autonomous runs |
//!
//! ## Philosophy
//! This crate follows the "small loop, large action space, explicit exit" direction described by Browser Use:
//! - [The Bitter Lesson of Agent Frameworks](https://browser-use.com/posts/bitter-lesson-agent-frameworks)
//! - [browser-use/agent-sdk](https://github.com/browser-use/agent-sdk)
//!
//! In this crate, that maps to:
//! - Tools define capability surface ([`ToolSpec`]).
//! - The run loop is explicit and inspectable via events ([`AgentEvent`]).
//! - Completion can be explicit with `done` mode ([`ToolOutcome::Done`]).
//! - Model adapters stay thin and replaceable ([`ChatModel`], [`AnthropicModel`], [`GoogleModel`], [`GrokModel`]).
//!
//! ## Quickstart
//! ```rust,no_run
//! use agent_sdk_rs::{Agent, AnthropicModel};
//!
//! # async fn run() -> Result<(), Box<dyn std::error::Error>> {
//! let model = AnthropicModel::from_env("claude-sonnet-4-5")?;
//! let mut agent = Agent::builder().model(model).build()?;
//!
//! let answer = agent.query("Summarize the task in one line.").await?;
//! println!("{answer}");
//! # Ok(())
//! # }
//! ```
//!
//! ## Streaming events
//! ```rust,no_run
//! use agent_sdk_rs::{Agent, AgentEvent, GoogleModel};
//! use futures_util::StreamExt;
//!
//! # async fn run() -> Result<(), Box<dyn std::error::Error>> {
//! let model = GoogleModel::from_env("gemini-2.5-flash")?;
//! let mut agent = Agent::builder().model(model).build()?;
//! let stream = agent.query_stream("Solve this step by step.");
//! futures_util::pin_mut!(stream);
//!
//! while let Some(event) = stream.next().await {
//!     match event? {
//!         AgentEvent::ToolCall { tool, .. } => println!("tool: {tool}"),
//!         AgentEvent::FinalResponse { content } => println!("final: {content}"),
//!         _ => {}
//!     }
//! }
//! # Ok(())
//! # }
//! ```
//!
//! ## Explicit `done` mode
//! For autonomous runs, require an explicit completion signal:
//! ```rust,no_run
//! use agent_sdk_rs::{Agent, AnthropicModel};
//! use agent_sdk_rs::tools::claude_code::all_tools;
//!
//! # async fn run() -> Result<(), Box<dyn std::error::Error>> {
//! let model = AnthropicModel::from_env("claude-sonnet-4-5")?;
//! let mut agent = Agent::builder()
//!     .model(model)
//!     .tools(all_tools())
//!     .require_done_tool(true)
//!     .max_iterations(64)
//!     .build()?;
//!
//! let _ = agent.query("Inspect the repo and summarize open risks.").await?;
//! # Ok(())
//! # }
//! ```
//!
//! ## Evidence in this repository
//! - Done-tool stop semantics and max-iteration guard: `src/agent/tests.rs`
//! - Dependency override behavior for tools: `src/agent/tests.rs`
//! - Tool schema and argument validation: `src/tools/mod.rs`
//! - Provider adapters with the same core interface: `src/llm/`

/// Agent loop, config, event stream, and query helpers.
pub mod agent;
/// Error types returned by schema validation, tools, providers, and agent runtime.
pub mod error;
/// Provider abstraction and model adapters.
pub mod llm;
/// Tool specification, dependency injection, and built-in Claude-code-style tools.
pub mod tools;

/// Agent runtime API.
pub use agent::{
    Agent, AgentBuilder, AgentConfig, AgentEvent, AgentRole, AgentToolChoice, StepStatus, query,
    query_stream,
};
/// Error values exposed by the SDK.
pub use error::{AgentError, ProviderError, SchemaError, ToolError};
/// Model adapters and model-interface types.
pub use llm::{
    AnthropicModel, AnthropicModelConfig, ChatModel, GoogleModel, GoogleModelConfig, GrokModel,
    GrokModelConfig, ModelCompletion, ModelMessage, ModelToolCall, ModelToolChoice,
    ModelToolDefinition, ModelUsage,
};
/// Tool and dependency primitives.
pub use tools::{DependencyMap, ToolOutcome, ToolSpec};
