mod anthropic;
mod google;
mod grok;

use async_trait::async_trait;
use serde_json::Value;

use crate::error::ProviderError;

pub use anthropic::{AnthropicModel, AnthropicModelConfig};
pub use google::{GoogleModel, GoogleModelConfig};
pub use grok::{GrokModel, GrokModelConfig};

#[derive(Clone, Debug, PartialEq)]
#[doc(hidden)]
pub enum ModelMessage {
    System(String),
    User(String),
    Assistant {
        content: Option<String>,
        tool_calls: Vec<ModelToolCall>,
    },
    ToolResult {
        tool_call_id: String,
        tool_name: String,
        content: String,
        is_error: bool,
    },
}

#[derive(Clone, Debug, PartialEq)]
#[doc(hidden)]
pub struct ModelToolCall {
    pub id: String,
    pub name: String,
    pub arguments: Value,
}

#[derive(Clone, Debug)]
#[doc(hidden)]
pub struct ModelToolDefinition {
    pub name: String,
    pub description: String,
    pub parameters: Value,
}

#[derive(Clone, Debug, PartialEq, Eq)]
#[doc(hidden)]
pub enum ModelToolChoice {
    Auto,
    Required,
    None,
    Tool(String),
}

#[derive(Clone, Debug, Default, PartialEq)]
#[doc(hidden)]
pub struct ModelCompletion {
    pub text: Option<String>,
    pub thinking: Option<String>,
    pub tool_calls: Vec<ModelToolCall>,
    pub usage: Option<ModelUsage>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
#[doc(hidden)]
pub struct ModelUsage {
    pub input_tokens: u32,
    pub output_tokens: u32,
}

#[async_trait]
/// Provider abstraction used by [`crate::Agent`].
pub trait ChatModel: Send + Sync {
    /// Invokes one model completion step for the current message history.
    async fn invoke(
        &self,
        messages: &[ModelMessage],
        tools: &[ModelToolDefinition],
        tool_choice: ModelToolChoice,
    ) -> Result<ModelCompletion, ProviderError>;
}
