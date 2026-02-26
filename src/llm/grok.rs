use async_trait::async_trait;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::error::ProviderError;
use crate::llm::{
    ChatModel, ModelCompletion, ModelMessage, ModelToolCall, ModelToolChoice, ModelToolDefinition,
    ModelUsage,
};

const DEFAULT_API_BASE_URL: &str = "https://api.x.ai/v1";
const EMPTY_USER_CONTENT_FALLBACK: &str = " ";

#[derive(Debug, Clone)]
pub struct GrokModelConfig {
    pub api_key: String,
    pub model: String,
    pub api_base_url: Option<String>,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub max_tokens: Option<u32>,
}

impl GrokModelConfig {
    pub fn new(api_key: impl Into<String>, model: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
            model: model.into(),
            api_base_url: None,
            temperature: None,
            top_p: None,
            max_tokens: Some(4096),
        }
    }
}

#[derive(Debug, Clone)]
pub struct GrokModel {
    client: Client,
    config: GrokModelConfig,
}

impl GrokModel {
    pub fn new(config: GrokModelConfig) -> Result<Self, ProviderError> {
        let client = Client::builder()
            .build()
            .map_err(|err| ProviderError::Request(err.to_string()))?;

        Ok(Self { client, config })
    }

    pub fn from_env(model: impl Into<String>) -> Result<Self, ProviderError> {
        let api_key = std::env::var("XAI_API_KEY")
            .or_else(|_| std::env::var("GROK_API_KEY"))
            .map_err(|_| {
                ProviderError::Request("XAI_API_KEY (or GROK_API_KEY) is not set".to_string())
            })?;

        Self::new(GrokModelConfig::new(api_key, model))
    }

    fn endpoint(&self) -> String {
        let base = self
            .config
            .api_base_url
            .as_deref()
            .unwrap_or(DEFAULT_API_BASE_URL)
            .trim_end_matches('/');
        format!("{base}/chat/completions")
    }
}

#[async_trait]
impl ChatModel for GrokModel {
    async fn invoke(
        &self,
        messages: &[ModelMessage],
        tools: &[ModelToolDefinition],
        tool_choice: ModelToolChoice,
    ) -> Result<ModelCompletion, ProviderError> {
        let request = build_request(messages, tools, tool_choice, &self.config);

        let response = self
            .client
            .post(self.endpoint())
            .header("authorization", format!("Bearer {}", self.config.api_key))
            .header("content-type", "application/json")
            .json(&request)
            .send()
            .await
            .map_err(|err| ProviderError::Request(err.to_string()))?;

        if !response.status().is_success() {
            return Err(ProviderError::Request(extract_api_error(response).await));
        }

        let payload = response
            .json::<GrokChatCompletionResponse>()
            .await
            .map_err(|err| ProviderError::Response(err.to_string()))?;

        normalize_response(payload)
    }
}

#[derive(Debug, Serialize)]
struct GrokChatCompletionRequest {
    model: String,
    messages: Vec<GrokRequestMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<GrokToolDefinition>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_choice: Option<GrokToolChoicePayload>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<u32>,
}

#[derive(Debug, Serialize)]
#[serde(tag = "role", rename_all = "lowercase")]
enum GrokRequestMessage {
    System {
        content: String,
    },
    User {
        content: String,
    },
    Assistant {
        #[serde(skip_serializing_if = "Option::is_none")]
        content: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        tool_calls: Option<Vec<GrokToolCall>>,
    },
    Tool {
        tool_call_id: String,
        content: String,
    },
}

#[derive(Debug, Serialize)]
struct GrokToolDefinition {
    #[serde(rename = "type")]
    type_: String,
    function: GrokToolFunctionDefinition,
}

#[derive(Debug, Serialize)]
struct GrokToolFunctionDefinition {
    name: String,
    description: String,
    parameters: Value,
}

#[derive(Debug, Serialize)]
#[serde(untagged)]
enum GrokToolChoicePayload {
    Mode(String),
    Specific {
        #[serde(rename = "type")]
        type_: String,
        function: GrokToolChoiceFunction,
    },
}

#[derive(Debug, Serialize)]
struct GrokToolChoiceFunction {
    name: String,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
struct GrokToolCall {
    id: String,
    #[serde(rename = "type")]
    type_: String,
    function: GrokToolCallFunction,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
struct GrokToolCallFunction {
    name: String,
    arguments: String,
}

#[derive(Debug, Deserialize)]
struct GrokChatCompletionResponse {
    #[serde(default)]
    choices: Vec<GrokChoice>,
    usage: Option<GrokUsage>,
}

#[derive(Debug, Deserialize)]
struct GrokChoice {
    message: Option<GrokAssistantMessage>,
}

#[derive(Debug, Deserialize)]
struct GrokAssistantMessage {
    content: Option<String>,
    #[serde(default)]
    tool_calls: Vec<GrokToolCall>,
    #[serde(default)]
    reasoning_content: Option<String>,
}

#[derive(Debug, Deserialize)]
struct GrokUsage {
    prompt_tokens: Option<u32>,
    completion_tokens: Option<u32>,
    completion_tokens_details: Option<GrokCompletionTokenDetails>,
}

#[derive(Debug, Deserialize)]
struct GrokCompletionTokenDetails {
    reasoning_tokens: Option<u32>,
}

#[derive(Debug, Deserialize)]
struct GrokErrorEnvelope {
    error: GrokApiError,
}

#[derive(Debug, Deserialize)]
struct GrokApiError {
    message: Option<String>,
    #[serde(rename = "type")]
    type_: Option<String>,
    code: Option<Value>,
}

fn build_request(
    messages: &[ModelMessage],
    tools: &[ModelToolDefinition],
    tool_choice: ModelToolChoice,
    config: &GrokModelConfig,
) -> GrokChatCompletionRequest {
    let request_messages = ensure_non_empty_messages(to_grok_messages(messages));

    let tools_payload = if tools.is_empty() {
        None
    } else {
        Some(
            tools
                .iter()
                .map(|tool| GrokToolDefinition {
                    type_: "function".to_string(),
                    function: GrokToolFunctionDefinition {
                        name: tool.name.clone(),
                        description: tool.description.clone(),
                        parameters: tool.parameters.clone(),
                    },
                })
                .collect::<Vec<_>>(),
        )
    };

    let tool_choice_payload = if tools.is_empty() {
        None
    } else {
        Some(match tool_choice {
            ModelToolChoice::Auto => GrokToolChoicePayload::Mode("auto".to_string()),
            ModelToolChoice::Required => GrokToolChoicePayload::Mode("required".to_string()),
            ModelToolChoice::None => GrokToolChoicePayload::Mode("none".to_string()),
            ModelToolChoice::Tool(name) => GrokToolChoicePayload::Specific {
                type_: "function".to_string(),
                function: GrokToolChoiceFunction { name },
            },
        })
    };

    GrokChatCompletionRequest {
        model: config.model.clone(),
        messages: request_messages,
        tools: tools_payload,
        tool_choice: tool_choice_payload,
        temperature: config.temperature,
        top_p: config.top_p,
        max_tokens: config.max_tokens,
    }
}

fn to_grok_messages(messages: &[ModelMessage]) -> Vec<GrokRequestMessage> {
    let mut request_messages = Vec::new();

    for message in messages {
        match message {
            ModelMessage::System(content) => {
                if content.is_empty() {
                    continue;
                }
                request_messages.push(GrokRequestMessage::System {
                    content: content.clone(),
                });
            }
            ModelMessage::User(content) => {
                if content.is_empty() {
                    continue;
                }
                request_messages.push(GrokRequestMessage::User {
                    content: content.clone(),
                });
            }
            ModelMessage::Assistant {
                content,
                tool_calls,
            } => {
                let serialized_tool_calls = tool_calls
                    .iter()
                    .map(|tool_call| GrokToolCall {
                        id: tool_call.id.clone(),
                        type_: "function".to_string(),
                        function: GrokToolCallFunction {
                            name: tool_call.name.clone(),
                            arguments: tool_call.arguments.to_string(),
                        },
                    })
                    .collect::<Vec<_>>();

                let assistant_content = content.as_ref().filter(|text| !text.is_empty()).cloned();
                if assistant_content.is_none() && serialized_tool_calls.is_empty() {
                    continue;
                }

                request_messages.push(GrokRequestMessage::Assistant {
                    content: assistant_content,
                    tool_calls: if serialized_tool_calls.is_empty() {
                        None
                    } else {
                        Some(serialized_tool_calls)
                    },
                });
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

                request_messages.push(GrokRequestMessage::Tool {
                    tool_call_id: tool_call_id.clone(),
                    content: rendered,
                });
            }
        }
    }

    request_messages
}

fn ensure_non_empty_messages(mut messages: Vec<GrokRequestMessage>) -> Vec<GrokRequestMessage> {
    if messages.is_empty() {
        messages.push(GrokRequestMessage::User {
            content: EMPTY_USER_CONTENT_FALLBACK.to_string(),
        });
    }

    messages
}

fn normalize_response(
    response: GrokChatCompletionResponse,
) -> Result<ModelCompletion, ProviderError> {
    let choice = response
        .choices
        .into_iter()
        .next()
        .ok_or_else(|| ProviderError::Response("grok response missing choices".to_string()))?;

    let message = choice.message.ok_or_else(|| {
        ProviderError::Response("grok response missing choice message".to_string())
    })?;

    let mut tool_calls = Vec::new();
    for tool_call in message.tool_calls {
        let arguments = if tool_call.function.arguments.trim().is_empty() {
            json!({})
        } else {
            serde_json::from_str::<Value>(&tool_call.function.arguments).map_err(|err| {
                ProviderError::Response(format!(
                    "grok tool call arguments for '{}' are not valid JSON: {err}",
                    tool_call.function.name
                ))
            })?
        };

        tool_calls.push(ModelToolCall {
            id: tool_call.id,
            name: tool_call.function.name,
            arguments,
        });
    }

    let usage = response.usage.map(|usage| ModelUsage {
        input_tokens: usage.prompt_tokens.unwrap_or(0),
        output_tokens: usage.completion_tokens.unwrap_or(0).saturating_add(
            usage
                .completion_tokens_details
                .and_then(|details| details.reasoning_tokens)
                .unwrap_or(0),
        ),
    });

    Ok(ModelCompletion {
        text: message.content.filter(|text| !text.is_empty()),
        thinking: message.reasoning_content.filter(|text| !text.is_empty()),
        tool_calls,
        usage,
    })
}

async fn extract_api_error(response: reqwest::Response) -> String {
    let status = response.status();
    let body = response.text().await.unwrap_or_default();

    if let Ok(parsed) = serde_json::from_str::<GrokErrorEnvelope>(&body) {
        let code = parsed
            .error
            .code
            .map(|value| match value {
                Value::String(value) => value,
                other => other.to_string(),
            })
            .unwrap_or_else(|| status.as_u16().to_string());
        let error_type = parsed
            .error
            .type_
            .unwrap_or_else(|| status.to_string().to_uppercase());
        let message = parsed
            .error
            .message
            .unwrap_or_else(|| "unknown xai api error".to_string());

        return format!("xai api error {code} {error_type}: {message}");
    }

    if body.is_empty() {
        format!("xai api request failed ({status})")
    } else {
        format!("xai api request failed ({status}): {body}")
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    fn tool_definition() -> ModelToolDefinition {
        ModelToolDefinition {
            name: "lookup".to_string(),
            description: "Look up something".to_string(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "query": {"type": "string"}
                },
                "required": ["query"],
                "additionalProperties": false
            }),
        }
    }

    #[test]
    fn build_request_serializes_messages_tools_and_tool_choice() {
        let messages = vec![
            ModelMessage::System("You are helpful".to_string()),
            ModelMessage::User("Find docs".to_string()),
            ModelMessage::Assistant {
                content: Some("Calling tool".to_string()),
                tool_calls: vec![ModelToolCall {
                    id: "call_1".to_string(),
                    name: "lookup".to_string(),
                    arguments: json!({"query": "rust"}),
                }],
            },
            ModelMessage::ToolResult {
                tool_call_id: "call_1".to_string(),
                tool_name: "lookup".to_string(),
                content: "{\"result\":\"ok\"}".to_string(),
                is_error: false,
            },
        ];

        let mut config = GrokModelConfig::new("key", "grok-4-1-fast-reasoning");
        config.temperature = Some(0.2);
        config.max_tokens = Some(512);

        let request = build_request(
            &messages,
            &[tool_definition()],
            ModelToolChoice::Tool("lookup".to_string()),
            &config,
        );
        let value = serde_json::to_value(request).expect("serializes");

        assert_eq!(value["messages"][0]["role"], "system");
        assert_eq!(value["messages"][0]["content"], "You are helpful");
        assert_eq!(value["messages"][2]["role"], "assistant");
        assert_eq!(
            value["messages"][2]["tool_calls"][0]["function"]["name"],
            "lookup"
        );
        assert_eq!(
            value["messages"][2]["tool_calls"][0]["function"]["arguments"],
            "{\"query\":\"rust\"}"
        );
        assert_eq!(value["messages"][3]["role"], "tool");
        assert_eq!(value["messages"][3]["tool_call_id"], "call_1");
        assert_eq!(value["tools"][0]["function"]["name"], "lookup");
        assert_eq!(value["tool_choice"]["type"], "function");
        assert_eq!(value["tool_choice"]["function"]["name"], "lookup");
        assert!((value["temperature"].as_f64().unwrap_or_default() - 0.2).abs() < 1e-6);
        assert_eq!(value["max_tokens"], 512);
    }

    #[test]
    fn build_request_adds_fallback_content_for_empty_user_message() {
        let messages = vec![ModelMessage::User(String::new())];
        let config = GrokModelConfig::new("key", "grok-4-1-fast-reasoning");

        let request = build_request(&messages, &[], ModelToolChoice::Auto, &config);
        let value = serde_json::to_value(request).expect("serializes");

        assert_eq!(
            value["messages"].as_array().map(|values| values.len()),
            Some(1)
        );
        assert_eq!(value["messages"][0]["role"], "user");
        assert_eq!(value["messages"][0]["content"], " ");
        assert!(value.get("tools").is_none());
        assert!(value.get("tool_choice").is_none());
    }

    #[test]
    fn normalize_response_extracts_text_thinking_tool_calls_and_usage() {
        let response = GrokChatCompletionResponse {
            choices: vec![GrokChoice {
                message: Some(GrokAssistantMessage {
                    content: Some("answer".to_string()),
                    tool_calls: vec![GrokToolCall {
                        id: "call_x".to_string(),
                        type_: "function".to_string(),
                        function: GrokToolCallFunction {
                            name: "lookup".to_string(),
                            arguments: "{\"q\":\"rust\"}".to_string(),
                        },
                    }],
                    reasoning_content: Some("reasoning".to_string()),
                }),
            }],
            usage: Some(GrokUsage {
                prompt_tokens: Some(11),
                completion_tokens: Some(7),
                completion_tokens_details: Some(GrokCompletionTokenDetails {
                    reasoning_tokens: Some(3),
                }),
            }),
        };

        let completion = normalize_response(response).expect("response normalizes");

        assert_eq!(completion.text.as_deref(), Some("answer"));
        assert_eq!(completion.thinking.as_deref(), Some("reasoning"));
        assert_eq!(completion.tool_calls.len(), 1);
        assert_eq!(completion.tool_calls[0].name, "lookup");
        assert_eq!(completion.tool_calls[0].id, "call_x");
        assert_eq!(
            completion.usage,
            Some(ModelUsage {
                input_tokens: 11,
                output_tokens: 10,
            })
        );
    }

    #[test]
    fn normalize_response_requires_choices() {
        let err = normalize_response(GrokChatCompletionResponse {
            choices: Vec::new(),
            usage: None,
        })
        .expect_err("should fail");

        match err {
            ProviderError::Response(message) => {
                assert!(message.contains("missing choices"));
            }
            other => panic!("unexpected error: {other}"),
        }
    }

    #[test]
    fn normalize_response_fails_on_invalid_tool_arguments() {
        let err = normalize_response(GrokChatCompletionResponse {
            choices: vec![GrokChoice {
                message: Some(GrokAssistantMessage {
                    content: None,
                    tool_calls: vec![GrokToolCall {
                        id: "call_x".to_string(),
                        type_: "function".to_string(),
                        function: GrokToolCallFunction {
                            name: "lookup".to_string(),
                            arguments: "{not json}".to_string(),
                        },
                    }],
                    reasoning_content: None,
                }),
            }],
            usage: None,
        })
        .expect_err("should fail");

        match err {
            ProviderError::Response(message) => {
                assert!(message.contains("not valid JSON"));
            }
            other => panic!("unexpected error: {other}"),
        }
    }
}
