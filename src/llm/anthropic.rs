use anthropic_ai_sdk::client::AnthropicClient;
use anthropic_ai_sdk::types::message::{
    ContentBlock, CreateMessageParams, CreateMessageResponse, Message, MessageClient, MessageError,
    RequiredMessageParams, Role, Thinking, ThinkingType, Tool, ToolChoice,
};
use async_trait::async_trait;

use crate::error::ProviderError;
use crate::llm::{
    ChatModel, ModelCompletion, ModelMessage, ModelToolCall, ModelToolChoice, ModelToolDefinition,
    ModelUsage,
};

#[cfg(test)]
use anthropic_ai_sdk::types::message::ContentBlockDelta;
#[cfg(test)]
use anthropic_ai_sdk::types::message::{MessageStartContent, StopReason, StreamEvent};

#[derive(Debug, Clone)]
/// Runtime configuration for [`AnthropicModel`].
pub struct AnthropicModelConfig {
    /// Anthropic API key.
    pub api_key: String,
    /// Model id (for example `claude-sonnet-4-5`).
    pub model: String,
    /// Anthropic API version header value.
    pub api_version: String,
    /// Optional base URL override for proxies or compatible endpoints.
    pub api_base_url: Option<String>,
    /// Maximum output tokens per call.
    pub max_tokens: u32,
    /// Optional sampling temperature.
    pub temperature: Option<f32>,
    /// Optional nucleus sampling parameter.
    pub top_p: Option<f32>,
    /// Optional budget for extended thinking tokens.
    pub thinking_budget_tokens: Option<usize>,
}

impl AnthropicModelConfig {
    /// Creates a config with sensible defaults.
    pub fn new(api_key: impl Into<String>, model: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
            model: model.into(),
            api_version: AnthropicClient::DEFAULT_API_VERSION.to_string(),
            api_base_url: None,
            max_tokens: 4096,
            temperature: None,
            top_p: None,
            thinking_budget_tokens: None,
        }
    }
}

#[derive(Debug, Clone)]
/// Anthropic provider adapter implementing [`ChatModel`].
pub struct AnthropicModel {
    client: AnthropicClient,
    config: AnthropicModelConfig,
}

impl AnthropicModel {
    /// Creates a model adapter from explicit config.
    pub fn new(config: AnthropicModelConfig) -> Result<Self, ProviderError> {
        let mut builder =
            AnthropicClient::builder(config.api_key.clone(), config.api_version.clone());
        if let Some(url) = &config.api_base_url {
            builder = builder.with_api_base_url(url.clone());
        }

        let client = builder
            .build::<MessageError>()
            .map_err(|err| ProviderError::Request(err.to_string()))?;

        Ok(Self { client, config })
    }

    /// Creates a model adapter using `ANTHROPIC_API_KEY` from the environment.
    pub fn from_env(model: impl Into<String>) -> Result<Self, ProviderError> {
        let api_key = std::env::var("ANTHROPIC_API_KEY")
            .map_err(|_| ProviderError::Request("ANTHROPIC_API_KEY is not set".to_string()))?;
        Self::new(AnthropicModelConfig::new(api_key, model))
    }
}

#[async_trait]
impl ChatModel for AnthropicModel {
    async fn invoke(
        &self,
        messages: &[ModelMessage],
        tools: &[ModelToolDefinition],
        tool_choice: ModelToolChoice,
    ) -> Result<ModelCompletion, ProviderError> {
        let (history, system) = to_anthropic_messages(messages);

        let required = RequiredMessageParams {
            model: self.config.model.clone(),
            messages: history,
            max_tokens: self.config.max_tokens,
        };

        let mut request = CreateMessageParams::new(required).with_stream(false);

        if let Some(system_prompt) = system {
            request = request.with_system(system_prompt);
        }

        if let Some(temperature) = self.config.temperature {
            request = request.with_temperature(temperature);
        }

        if let Some(top_p) = self.config.top_p {
            request = request.with_top_p(top_p);
        }

        if let Some(budget_tokens) = self.config.thinking_budget_tokens {
            request = request.with_thinking(Thinking {
                budget_tokens,
                type_: ThinkingType::Enabled,
            });
        }

        if !tools.is_empty() {
            let anthropic_tools = tools
                .iter()
                .map(|tool| Tool {
                    name: tool.name.clone(),
                    description: Some(tool.description.clone()),
                    input_schema: tool.parameters.clone(),
                })
                .collect::<Vec<_>>();

            request = request.with_tools(anthropic_tools);
            request = request.with_tool_choice(match tool_choice {
                ModelToolChoice::Auto => ToolChoice::Auto,
                ModelToolChoice::Required => ToolChoice::Any,
                ModelToolChoice::None => ToolChoice::None,
                ModelToolChoice::Tool(name) => ToolChoice::Tool { name },
            });
        }

        let response = self
            .client
            .create_message(Some(&request))
            .await
            .map_err(|err| ProviderError::Request(err.to_string()))?;

        Ok(normalize_response(&response))
    }
}

fn to_anthropic_messages(messages: &[ModelMessage]) -> (Vec<Message>, Option<String>) {
    let mut system_lines = Vec::new();
    let mut anthropic_messages = Vec::new();

    for message in messages {
        match message {
            ModelMessage::System(content) => system_lines.push(content.clone()),
            ModelMessage::User(content) => {
                anthropic_messages.push(Message::new_text(Role::User, content.clone()));
            }
            ModelMessage::Assistant {
                content,
                tool_calls,
            } => {
                let mut blocks = Vec::new();
                if let Some(content) = content {
                    if !content.is_empty() {
                        blocks.push(ContentBlock::Text {
                            text: content.clone(),
                        });
                    }
                }
                for call in tool_calls {
                    blocks.push(ContentBlock::ToolUse {
                        id: call.id.clone(),
                        name: call.name.clone(),
                        input: call.arguments.clone(),
                    });
                }
                if !blocks.is_empty() {
                    anthropic_messages.push(Message::new_blocks(Role::Assistant, blocks));
                }
            }
            ModelMessage::ToolResult {
                tool_call_id,
                tool_name: _,
                content,
                is_error,
            } => {
                let rendered = if *is_error {
                    format!("Error: {content}")
                } else {
                    content.clone()
                };
                anthropic_messages.push(Message::new_blocks(
                    Role::User,
                    vec![ContentBlock::ToolResult {
                        tool_use_id: tool_call_id.clone(),
                        content: rendered,
                    }],
                ));
            }
        }
    }

    let system = if system_lines.is_empty() {
        None
    } else {
        Some(system_lines.join("\n\n"))
    };

    (anthropic_messages, system)
}

fn normalize_response(response: &CreateMessageResponse) -> ModelCompletion {
    let mut text_parts = Vec::new();
    let mut thinking_parts = Vec::new();
    let mut tool_calls = Vec::new();

    for block in &response.content {
        match block {
            ContentBlock::Text { text } => text_parts.push(text.clone()),
            ContentBlock::ToolUse { id, name, input } => tool_calls.push(ModelToolCall {
                id: id.clone(),
                name: name.clone(),
                arguments: input.clone(),
            }),
            ContentBlock::Thinking { thinking, .. } => thinking_parts.push(thinking.clone()),
            ContentBlock::RedactedThinking { data } => {
                thinking_parts.push(format!("[redacted:{} bytes]", data.len()))
            }
            _ => {}
        }
    }

    let text = if text_parts.is_empty() {
        None
    } else {
        Some(text_parts.join("\n"))
    };

    let thinking = if thinking_parts.is_empty() {
        None
    } else {
        Some(thinking_parts.join("\n"))
    };

    ModelCompletion {
        text,
        thinking,
        tool_calls,
        usage: Some(ModelUsage {
            input_tokens: response.usage.input_tokens,
            output_tokens: response.usage.output_tokens,
        }),
    }
}

#[cfg(test)]
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum AnthropicStreamChunk {
    Text {
        index: usize,
        text: String,
    },
    Thinking {
        index: usize,
        content: String,
    },
    ToolInputJson {
        index: usize,
        partial_json: String,
    },
    ToolCallStart {
        id: String,
        name: String,
        input: serde_json::Value,
    },
    Signature {
        index: usize,
        signature: String,
    },
    MessageStop {
        stop_reason: Option<String>,
    },
    Error {
        message: String,
    },
}

#[cfg(test)]
pub(crate) fn normalize_stream_event(event: &StreamEvent) -> Option<AnthropicStreamChunk> {
    match event {
        StreamEvent::ContentBlockStart {
            index: _,
            content_block,
        } => {
            if let ContentBlock::ToolUse { id, name, input } = content_block {
                Some(AnthropicStreamChunk::ToolCallStart {
                    id: id.clone(),
                    name: name.clone(),
                    input: input.clone(),
                })
            } else {
                None
            }
        }
        StreamEvent::ContentBlockDelta { index, delta } => match delta {
            ContentBlockDelta::TextDelta { text } => Some(AnthropicStreamChunk::Text {
                index: *index,
                text: text.clone(),
            }),
            ContentBlockDelta::ThinkingDelta { thinking } => Some(AnthropicStreamChunk::Thinking {
                index: *index,
                content: thinking.clone(),
            }),
            ContentBlockDelta::InputJsonDelta { partial_json } => {
                Some(AnthropicStreamChunk::ToolInputJson {
                    index: *index,
                    partial_json: partial_json.clone(),
                })
            }
            ContentBlockDelta::SignatureDelta { signature } => {
                Some(AnthropicStreamChunk::Signature {
                    index: *index,
                    signature: signature.clone(),
                })
            }
        },
        StreamEvent::MessageDelta { delta, usage: _ } => Some(AnthropicStreamChunk::MessageStop {
            stop_reason: delta.stop_reason.as_ref().map(stop_reason_name),
        }),
        StreamEvent::MessageStop => Some(AnthropicStreamChunk::MessageStop { stop_reason: None }),
        StreamEvent::Error { error } => Some(AnthropicStreamChunk::Error {
            message: error.message.clone(),
        }),
        StreamEvent::MessageStart {
            message: MessageStartContent { .. },
        }
        | StreamEvent::ContentBlockStop { .. }
        | StreamEvent::Ping => None,
    }
}

#[cfg(test)]
fn stop_reason_name(stop_reason: &StopReason) -> String {
    match stop_reason {
        StopReason::EndTurn => "end_turn",
        StopReason::MaxTokens => "max_tokens",
        StopReason::StopSequence => "stop_sequence",
        StopReason::ToolUse => "tool_use",
        StopReason::Refusal => "refusal",
    }
    .to_string()
}

#[cfg(test)]
mod tests {
    use anthropic_ai_sdk::types::message::MessageContent;
    use serde_json::json;

    use super::*;
    use crate::llm::ModelMessage;

    #[test]
    fn normalize_response_extracts_tool_calls_and_text() {
        let response = CreateMessageResponse {
            content: vec![
                ContentBlock::Text {
                    text: "Looking up".to_string(),
                },
                ContentBlock::ToolUse {
                    id: "call_1".to_string(),
                    name: "search".to_string(),
                    input: json!({"query": "rust"}),
                },
            ],
            id: "msg_1".to_string(),
            model: "claude-test".to_string(),
            role: Role::Assistant,
            stop_reason: Some(StopReason::ToolUse),
            stop_sequence: None,
            type_: "message".to_string(),
            usage: anthropic_ai_sdk::types::message::Usage {
                input_tokens: 1,
                output_tokens: 1,
            },
        };

        let completion = normalize_response(&response);
        assert_eq!(completion.text.as_deref(), Some("Looking up"));
        assert_eq!(completion.tool_calls.len(), 1);
        assert_eq!(completion.tool_calls[0].name, "search");
    }

    #[test]
    fn to_anthropic_messages_serializes_tool_result() {
        let history = vec![
            ModelMessage::System("sys".to_string()),
            ModelMessage::User("u1".to_string()),
            ModelMessage::ToolResult {
                tool_call_id: "call_1".to_string(),
                tool_name: "search".to_string(),
                content: "failed".to_string(),
                is_error: true,
            },
        ];

        let (messages, system) = to_anthropic_messages(&history);
        assert_eq!(system.as_deref(), Some("sys"));
        assert_eq!(messages.len(), 2);

        let MessageContent::Blocks { content } = &messages[1].content else {
            panic!("expected blocks")
        };
        assert_eq!(
            content[0],
            ContentBlock::ToolResult {
                tool_use_id: "call_1".to_string(),
                content: "Error: failed".to_string(),
            }
        );
    }

    #[test]
    fn normalize_stream_event_maps_deltas() {
        let text_event = StreamEvent::ContentBlockDelta {
            index: 0,
            delta: ContentBlockDelta::TextDelta {
                text: "hi".to_string(),
            },
        };
        let mapped_text = normalize_stream_event(&text_event);
        assert_eq!(
            mapped_text,
            Some(AnthropicStreamChunk::Text {
                index: 0,
                text: "hi".to_string(),
            })
        );

        let thinking_event = StreamEvent::ContentBlockDelta {
            index: 1,
            delta: ContentBlockDelta::ThinkingDelta {
                thinking: "plan".to_string(),
            },
        };
        let mapped_thinking = normalize_stream_event(&thinking_event);
        assert_eq!(
            mapped_thinking,
            Some(AnthropicStreamChunk::Thinking {
                index: 1,
                content: "plan".to_string(),
            })
        );
    }

    #[test]
    fn normalize_response_handles_thinking_without_text() {
        let response = CreateMessageResponse {
            content: vec![ContentBlock::Thinking {
                thinking: "I should call a tool".to_string(),
                signature: "sig".to_string(),
            }],
            id: "msg_2".to_string(),
            model: "claude-test".to_string(),
            role: Role::Assistant,
            stop_reason: Some(StopReason::EndTurn),
            stop_sequence: None,
            type_: "message".to_string(),
            usage: anthropic_ai_sdk::types::message::Usage {
                input_tokens: 1,
                output_tokens: 1,
            },
        };

        let completion = normalize_response(&response);
        assert!(completion.text.is_none());
        assert_eq!(
            completion.thinking,
            Some("I should call a tool".to_string())
        );
    }

    #[test]
    fn normalize_stream_event_extracts_tool_call_start() {
        let event = StreamEvent::ContentBlockStart {
            index: 0,
            content_block: ContentBlock::ToolUse {
                id: "tool_1".to_string(),
                name: "lookup".to_string(),
                input: json!({"x": 1}),
            },
        };

        let mapped = normalize_stream_event(&event);
        assert_eq!(
            mapped,
            Some(AnthropicStreamChunk::ToolCallStart {
                id: "tool_1".to_string(),
                name: "lookup".to_string(),
                input: json!({"x": 1}),
            })
        );
    }
}
