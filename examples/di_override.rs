use std::collections::VecDeque;
use std::error::Error;
use std::sync::Mutex;

use agent_sdk_rs::{
    Agent, ChatModel, ModelCompletion, ModelMessage, ModelToolCall, ModelToolChoice,
    ModelToolDefinition, ProviderError, ToolOutcome, ToolSpec,
};
use async_trait::async_trait;
use serde_json::json;

#[derive(Default)]
struct ScriptedModel {
    responses: Mutex<VecDeque<Result<ModelCompletion, ProviderError>>>,
}

impl ScriptedModel {
    fn new(responses: Vec<Result<ModelCompletion, ProviderError>>) -> Self {
        Self {
            responses: Mutex::new(VecDeque::from(responses)),
        }
    }
}

#[async_trait]
impl ChatModel for ScriptedModel {
    async fn invoke(
        &self,
        _messages: &[ModelMessage],
        _tools: &[ModelToolDefinition],
        _tool_choice: ModelToolChoice,
    ) -> Result<ModelCompletion, ProviderError> {
        let mut guard = self.responses.lock().expect("lock poisoned");
        guard.pop_front().unwrap_or_else(|| {
            Err(ProviderError::Response(
                "scripted model exhausted responses".to_string(),
            ))
        })
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    let read_dep_tool = ToolSpec::new("read_dep", "read injected value")
        .with_schema(json!({
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": false
        }))?
        .with_handler(|_args, deps| {
            let value = deps.get::<u32>().map(|v| *v).unwrap_or_default();
            async move { Ok(ToolOutcome::Text(value.to_string())) }
        });

    let done_tool = ToolSpec::new("done", "finish")
        .with_schema(json!({
            "type": "object",
            "properties": {
                "message": {"type": "string"}
            },
            "required": ["message"],
            "additionalProperties": false
        }))?
        .with_handler(|args, _deps| async move {
            let message = args
                .get("message")
                .and_then(|v| v.as_str())
                .unwrap_or("done");
            Ok(ToolOutcome::Done(message.to_string()))
        });

    let model = ScriptedModel::new(vec![
        Ok(ModelCompletion {
            text: None,
            thinking: None,
            tool_calls: vec![ModelToolCall {
                id: "call_1".to_string(),
                name: "read_dep".to_string(),
                arguments: json!({}),
            }],
        }),
        Ok(ModelCompletion {
            text: None,
            thinking: None,
            tool_calls: vec![ModelToolCall {
                id: "call_2".to_string(),
                name: "done".to_string(),
                arguments: json!({"message": "dependency override applied"}),
            }],
        }),
    ]);

    let mut agent = Agent::builder()
        .model(model)
        .tool(read_dep_tool)
        .tool(done_tool)
        .dependency(1_u32)
        .dependency_override(9_u32)
        .build()?;

    let response = agent.query("use dependency").await?;
    println!("final: {response}");

    Ok(())
}
