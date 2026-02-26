use async_trait::async_trait;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value, json};

use crate::error::ProviderError;
use crate::llm::{
    ChatModel, ModelCompletion, ModelMessage, ModelToolCall, ModelToolChoice, ModelToolDefinition,
    ModelUsage,
};

const DEFAULT_API_BASE_URL: &str = "https://generativelanguage.googleapis.com/v1beta";

#[derive(Debug, Clone)]
pub struct GoogleModelConfig {
    pub api_key: String,
    pub model: String,
    pub api_base_url: Option<String>,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub max_output_tokens: Option<u32>,
    pub thinking_budget_tokens: Option<u32>,
    pub include_thoughts: Option<bool>,
}

impl GoogleModelConfig {
    pub fn new(api_key: impl Into<String>, model: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
            model: model.into(),
            api_base_url: None,
            temperature: None,
            top_p: None,
            max_output_tokens: Some(4096),
            thinking_budget_tokens: None,
            include_thoughts: None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct GoogleModel {
    client: Client,
    config: GoogleModelConfig,
}

impl GoogleModel {
    pub fn new(config: GoogleModelConfig) -> Result<Self, ProviderError> {
        let client = Client::builder()
            .build()
            .map_err(|err| ProviderError::Request(err.to_string()))?;

        Ok(Self { client, config })
    }

    pub fn from_env(model: impl Into<String>) -> Result<Self, ProviderError> {
        let api_key = std::env::var("GOOGLE_API_KEY")
            .or_else(|_| std::env::var("GEMINI_API_KEY"))
            .map_err(|_| {
                ProviderError::Request("GOOGLE_API_KEY (or GEMINI_API_KEY) is not set".to_string())
            })?;

        Self::new(GoogleModelConfig::new(api_key, model))
    }

    fn endpoint(&self) -> String {
        let base = self
            .config
            .api_base_url
            .as_deref()
            .unwrap_or(DEFAULT_API_BASE_URL)
            .trim_end_matches('/');
        format!("{base}/models/{}:generateContent", self.config.model)
    }
}

#[async_trait]
impl ChatModel for GoogleModel {
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
            .header("x-goog-api-key", &self.config.api_key)
            .header("content-type", "application/json")
            .json(&request)
            .send()
            .await
            .map_err(|err| ProviderError::Request(err.to_string()))?;

        if !response.status().is_success() {
            return Err(ProviderError::Request(extract_api_error(response).await));
        }

        let payload = response
            .json::<GenerateContentResponse>()
            .await
            .map_err(|err| ProviderError::Response(err.to_string()))?;

        normalize_response(payload)
    }
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct GenerateContentRequest {
    contents: Vec<GoogleContent>,
    #[serde(skip_serializing_if = "Option::is_none")]
    system_instruction: Option<GoogleSystemInstruction>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<GoogleTool>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_config: Option<GoogleToolConfig>,
    #[serde(skip_serializing_if = "Option::is_none")]
    generation_config: Option<GoogleGenerationConfig>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(rename_all = "camelCase")]
struct GoogleContent {
    role: String,
    parts: Vec<GooglePart>,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct GoogleSystemInstruction {
    parts: Vec<GooglePart>,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct GoogleTool {
    function_declarations: Vec<GoogleFunctionDeclaration>,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct GoogleFunctionDeclaration {
    name: String,
    description: String,
    parameters: Value,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct GoogleToolConfig {
    function_calling_config: GoogleFunctionCallingConfig,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct GoogleFunctionCallingConfig {
    mode: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    allowed_function_names: Option<Vec<String>>,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct GoogleGenerationConfig {
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_output_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    thinking_config: Option<GoogleThinkingConfig>,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct GoogleThinkingConfig {
    thinking_budget: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    include_thoughts: Option<bool>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(rename_all = "camelCase")]
struct GooglePart {
    #[serde(skip_serializing_if = "Option::is_none")]
    text: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    thought: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    function_call: Option<GoogleFunctionCall>,
    #[serde(skip_serializing_if = "Option::is_none")]
    function_response: Option<GoogleFunctionResponse>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(rename_all = "camelCase")]
struct GoogleFunctionCall {
    id: Option<String>,
    name: Option<String>,
    args: Option<Value>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(rename_all = "camelCase")]
struct GoogleFunctionResponse {
    name: String,
    response: Value,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct GenerateContentResponse {
    #[serde(default)]
    candidates: Vec<GoogleCandidate>,
    usage_metadata: Option<GoogleUsageMetadata>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct GoogleCandidate {
    content: Option<GoogleContent>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct GoogleUsageMetadata {
    prompt_token_count: Option<u32>,
    candidates_token_count: Option<u32>,
    thoughts_token_count: Option<u32>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct GoogleErrorEnvelope {
    error: GoogleApiError,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct GoogleApiError {
    code: Option<u16>,
    status: Option<String>,
    message: Option<String>,
}

fn build_request(
    messages: &[ModelMessage],
    tools: &[ModelToolDefinition],
    tool_choice: ModelToolChoice,
    config: &GoogleModelConfig,
) -> GenerateContentRequest {
    let (contents, system_instruction) = to_google_contents(messages);

    let tools_payload = if tools.is_empty() {
        None
    } else {
        let declarations = tools
            .iter()
            .map(|tool| GoogleFunctionDeclaration {
                name: tool.name.clone(),
                description: tool.description.clone(),
                parameters: clean_gemini_schema(tool.parameters.clone()),
            })
            .collect::<Vec<_>>();
        Some(vec![GoogleTool {
            function_declarations: declarations,
        }])
    };

    let tool_config = if tools.is_empty() {
        None
    } else {
        Some(match tool_choice {
            ModelToolChoice::Auto => GoogleToolConfig {
                function_calling_config: GoogleFunctionCallingConfig {
                    mode: "AUTO".to_string(),
                    allowed_function_names: None,
                },
            },
            ModelToolChoice::Required => GoogleToolConfig {
                function_calling_config: GoogleFunctionCallingConfig {
                    mode: "ANY".to_string(),
                    allowed_function_names: None,
                },
            },
            ModelToolChoice::None => GoogleToolConfig {
                function_calling_config: GoogleFunctionCallingConfig {
                    mode: "NONE".to_string(),
                    allowed_function_names: None,
                },
            },
            ModelToolChoice::Tool(name) => GoogleToolConfig {
                function_calling_config: GoogleFunctionCallingConfig {
                    mode: "ANY".to_string(),
                    allowed_function_names: Some(vec![name]),
                },
            },
        })
    };

    let thinking_config = config
        .thinking_budget_tokens
        .map(|budget| GoogleThinkingConfig {
            thinking_budget: budget,
            include_thoughts: config.include_thoughts,
        });

    let generation_config = GoogleGenerationConfig {
        temperature: config.temperature,
        top_p: config.top_p,
        max_output_tokens: config.max_output_tokens,
        thinking_config,
    };

    GenerateContentRequest {
        contents,
        system_instruction: system_instruction.map(|instruction| GoogleSystemInstruction {
            parts: vec![GooglePart {
                text: Some(instruction),
                thought: None,
                function_call: None,
                function_response: None,
            }],
        }),
        tools: tools_payload,
        tool_config,
        generation_config: Some(generation_config),
    }
}

fn to_google_contents(messages: &[ModelMessage]) -> (Vec<GoogleContent>, Option<String>) {
    let mut system_lines = Vec::new();
    let mut contents = Vec::new();

    for message in messages {
        match message {
            ModelMessage::System(content) => {
                if !content.is_empty() {
                    system_lines.push(content.clone());
                }
            }
            ModelMessage::User(content) => {
                if content.is_empty() {
                    continue;
                }
                contents.push(GoogleContent {
                    role: "user".to_string(),
                    parts: vec![GooglePart {
                        text: Some(content.clone()),
                        thought: None,
                        function_call: None,
                        function_response: None,
                    }],
                });
            }
            ModelMessage::Assistant {
                content,
                tool_calls,
            } => {
                let mut parts = Vec::new();

                if let Some(text) = content
                    && !text.is_empty()
                {
                    parts.push(GooglePart {
                        text: Some(text.clone()),
                        thought: None,
                        function_call: None,
                        function_response: None,
                    });
                }

                for call in tool_calls {
                    parts.push(GooglePart {
                        text: None,
                        thought: None,
                        function_call: Some(GoogleFunctionCall {
                            id: Some(call.id.clone()),
                            name: Some(call.name.clone()),
                            args: Some(call.arguments.clone()),
                        }),
                        function_response: None,
                    });
                }

                if !parts.is_empty() {
                    contents.push(GoogleContent {
                        role: "model".to_string(),
                        parts,
                    });
                }
            }
            ModelMessage::ToolResult {
                tool_call_id: _,
                tool_name,
                content,
                is_error,
            } => contents.push(GoogleContent {
                role: "user".to_string(),
                parts: vec![GooglePart {
                    text: None,
                    thought: None,
                    function_call: None,
                    function_response: Some(GoogleFunctionResponse {
                        name: tool_name.clone(),
                        response: tool_result_payload(content, *is_error),
                    }),
                }],
            }),
        }
    }

    let system = if system_lines.is_empty() {
        None
    } else {
        Some(system_lines.join("\n\n"))
    };

    (contents, system)
}

fn tool_result_payload(content: &str, is_error: bool) -> Value {
    if is_error {
        return json!({"error": content});
    }

    if let Ok(parsed) = serde_json::from_str::<Value>(content) {
        parsed
    } else {
        json!({"result": content})
    }
}

fn normalize_response(response: GenerateContentResponse) -> Result<ModelCompletion, ProviderError> {
    let Some(candidate) = response.candidates.into_iter().next() else {
        return Err(ProviderError::Response(
            "google response missing candidates".to_string(),
        ));
    };

    let mut text_parts = Vec::new();
    let mut thinking_parts = Vec::new();
    let mut tool_calls = Vec::new();

    if let Some(content) = candidate.content {
        for (index, part) in content.parts.into_iter().enumerate() {
            if let Some(text) = part.text {
                if part.thought.unwrap_or(false) {
                    thinking_parts.push(text);
                } else {
                    text_parts.push(text);
                }
            }

            if let Some(function_call) = part.function_call {
                let Some(name) = function_call.name else {
                    return Err(ProviderError::Response(
                        "google functionCall missing name".to_string(),
                    ));
                };

                tool_calls.push(ModelToolCall {
                    id: function_call
                        .id
                        .unwrap_or_else(|| format!("call_{}", index + 1)),
                    name,
                    arguments: function_call.args.unwrap_or_else(|| json!({})),
                });
            }
        }
    }

    let usage = response.usage_metadata.map(|usage| ModelUsage {
        input_tokens: usage.prompt_token_count.unwrap_or(0),
        output_tokens: usage
            .candidates_token_count
            .unwrap_or(0)
            .saturating_add(usage.thoughts_token_count.unwrap_or(0)),
    });

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

    Ok(ModelCompletion {
        text,
        thinking,
        tool_calls,
        usage,
    })
}

async fn extract_api_error(response: reqwest::Response) -> String {
    let status = response.status();
    let body = response.text().await.unwrap_or_default();

    if let Ok(parsed) = serde_json::from_str::<GoogleErrorEnvelope>(&body) {
        let code = parsed.error.code.unwrap_or(status.as_u16());
        let status_name = parsed
            .error
            .status
            .unwrap_or_else(|| status.to_string().to_uppercase());
        let message = parsed
            .error
            .message
            .unwrap_or_else(|| "unknown google api error".to_string());
        return format!("google api error {code} {status_name}: {message}");
    }

    if body.is_empty() {
        format!("google api request failed ({status})")
    } else {
        format!("google api request failed ({status}): {body}")
    }
}

fn clean_gemini_schema(schema: Value) -> Value {
    let mut root = schema;
    let defs = match &mut root {
        Value::Object(map) => map
            .remove("$defs")
            .and_then(|value| match value {
                Value::Object(defs) => Some(defs),
                _ => None,
            })
            .unwrap_or_default(),
        _ => Map::new(),
    };

    let resolved = resolve_schema_refs(root, &defs);
    clean_schema_node(resolved, None)
}

fn resolve_schema_refs(value: Value, defs: &Map<String, Value>) -> Value {
    match value {
        Value::Object(map) => {
            if let Some(reference) = map.get("$ref").and_then(Value::as_str) {
                let ref_name = reference.rsplit('/').next().unwrap_or("");
                if let Some(definition) = defs.get(ref_name) {
                    let mut resolved = definition.clone();
                    if let Value::Object(ref mut resolved_map) = resolved {
                        for (key, value) in map {
                            if key != "$ref" {
                                resolved_map.insert(key, value);
                            }
                        }
                    }
                    return resolve_schema_refs(resolved, defs);
                }
            }

            let mut out = Map::new();
            for (key, value) in map {
                out.insert(key, resolve_schema_refs(value, defs));
            }
            Value::Object(out)
        }
        Value::Array(values) => Value::Array(
            values
                .into_iter()
                .map(|value| resolve_schema_refs(value, defs))
                .collect(),
        ),
        other => other,
    }
}

fn clean_schema_node(value: Value, parent_key: Option<&str>) -> Value {
    match value {
        Value::Object(map) => {
            let mut cleaned = Map::new();

            for (key, value) in map {
                let is_metadata_title = key == "title" && parent_key != Some("properties");
                if key == "additionalProperties" || key == "default" || is_metadata_title {
                    continue;
                }

                cleaned.insert(key.clone(), clean_schema_node(value, Some(&key)));
            }

            let type_name = cleaned
                .get("type")
                .and_then(Value::as_str)
                .map(|t| t.to_ascii_lowercase());
            if type_name.as_deref() == Some("object") {
                let needs_placeholder = cleaned
                    .get("properties")
                    .and_then(Value::as_object)
                    .map(|properties| properties.is_empty())
                    .unwrap_or(false);

                if needs_placeholder {
                    cleaned.insert(
                        "properties".to_string(),
                        json!({"_placeholder": {"type": "string"}}),
                    );
                }
            }

            Value::Object(cleaned)
        }
        Value::Array(values) => Value::Array(
            values
                .into_iter()
                .map(|value| clean_schema_node(value, parent_key))
                .collect(),
        ),
        other => other,
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
                    "query": {"type": "string", "default": "x"}
                },
                "required": ["query"],
                "additionalProperties": false,
                "title": "LookupTool"
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

        let mut config = GoogleModelConfig::new("key", "gemini-2.5-flash");
        config.temperature = Some(0.2);
        config.thinking_budget_tokens = Some(256);

        let request = build_request(
            &messages,
            &[tool_definition()],
            ModelToolChoice::Tool("lookup".to_string()),
            &config,
        );
        let value = serde_json::to_value(request).expect("serializes");

        assert_eq!(
            value["systemInstruction"]["parts"][0]["text"],
            "You are helpful"
        );
        assert_eq!(value["contents"][0]["role"], "user");
        assert_eq!(
            value["contents"][1]["parts"][1]["functionCall"]["name"],
            "lookup"
        );
        assert_eq!(
            value["contents"][2]["parts"][0]["functionResponse"]["response"]["result"],
            "ok"
        );
        assert_eq!(value["toolConfig"]["functionCallingConfig"]["mode"], "ANY");
        assert_eq!(
            value["toolConfig"]["functionCallingConfig"]["allowedFunctionNames"][0],
            "lookup"
        );
        assert_eq!(
            value["generationConfig"]["thinkingConfig"]["thinkingBudget"],
            256
        );
        assert!(
            value["tools"][0]["functionDeclarations"][0]["parameters"]
                .get("additionalProperties")
                .is_none()
        );
    }

    #[test]
    fn normalize_response_extracts_text_thinking_tool_calls_and_usage() {
        let response = GenerateContentResponse {
            candidates: vec![GoogleCandidate {
                content: Some(GoogleContent {
                    role: "model".to_string(),
                    parts: vec![
                        GooglePart {
                            text: Some("answer".to_string()),
                            thought: None,
                            function_call: None,
                            function_response: None,
                        },
                        GooglePart {
                            text: Some("reasoning".to_string()),
                            thought: Some(true),
                            function_call: None,
                            function_response: None,
                        },
                        GooglePart {
                            text: None,
                            thought: None,
                            function_call: Some(GoogleFunctionCall {
                                id: Some("call_x".to_string()),
                                name: Some("lookup".to_string()),
                                args: Some(json!({"q": "rust"})),
                            }),
                            function_response: None,
                        },
                    ],
                }),
            }],
            usage_metadata: Some(GoogleUsageMetadata {
                prompt_token_count: Some(11),
                candidates_token_count: Some(7),
                thoughts_token_count: Some(3),
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
    fn normalize_response_requires_candidates() {
        let err = normalize_response(GenerateContentResponse {
            candidates: Vec::new(),
            usage_metadata: None,
        })
        .expect_err("should fail");

        match err {
            ProviderError::Response(message) => {
                assert!(message.contains("missing candidates"));
            }
            other => panic!("unexpected error: {other}"),
        }
    }

    #[test]
    fn clean_gemini_schema_resolves_refs_and_handles_empty_objects() {
        let schema = json!({
            "$defs": {
                "Inner": {
                    "type": "object",
                    "properties": {},
                    "additionalProperties": false
                }
            },
            "type": "object",
            "properties": {
                "inner": {
                    "$ref": "#/$defs/Inner"
                }
            },
            "additionalProperties": false
        });

        let cleaned = clean_gemini_schema(schema);
        assert!(cleaned.get("$defs").is_none());
        assert!(cleaned.get("additionalProperties").is_none());
        assert_eq!(
            cleaned["properties"]["inner"]["properties"]["_placeholder"]["type"],
            "string"
        );
    }
}
