//! Minimal agent SDK for Rust.
//!
//! v0 surface:
//! - `Agent` loop with explicit tool-based completion support (`ToolOutcome::Done`)
//! - `query` and `query_stream` entry points
//! - Tool registry + JSON schema validation + dependency injection
//! - Anthropic adapter via `AnthropicModel`

pub mod agent;
pub mod error;
pub mod llm;
pub mod tools;

pub use agent::{Agent, AgentBuilder, AgentConfig, AgentEvent, query, query_stream};
pub use error::{AgentError, ProviderError, SchemaError, ToolError};
pub use llm::{
    AnthropicModel, AnthropicModelConfig, ChatModel, ModelCompletion, ModelMessage, ModelToolCall,
    ModelToolChoice, ModelToolDefinition,
};
pub use tools::{DependencyMap, ToolOutcome, ToolSpec};
