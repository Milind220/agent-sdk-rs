use std::collections::VecDeque;
use std::error::Error;
use std::sync::Mutex;

use async_trait::async_trait;
use futures_util::StreamExt;
use rust_sdk::{
    Agent, AgentEvent, ChatModel, ModelCompletion, ModelMessage, ModelToolCall, ModelToolChoice,
    ModelToolDefinition, ProviderError, ToolError, ToolOutcome, ToolSpec,
};
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

fn add_tool() -> ToolSpec {
    ToolSpec::new("add", "add two numbers")
        .with_schema(json!({
            "type": "object",
            "properties": {
                "a": {"type": "integer"},
                "b": {"type": "integer"}
            },
            "required": ["a", "b"],
            "additionalProperties": false
        }))
        .expect("valid schema")
        .with_handler(|args, _deps| async move {
            let a = args
                .get("a")
                .and_then(|v| v.as_i64())
                .ok_or_else(|| ToolError::Execution("a missing".to_string()))?;
            let b = args
                .get("b")
                .and_then(|v| v.as_i64())
                .ok_or_else(|| ToolError::Execution("b missing".to_string()))?;
            Ok(ToolOutcome::Text((a + b).to_string()))
        })
}

fn done_tool() -> ToolSpec {
    ToolSpec::new("done", "complete and return")
        .with_schema(json!({
            "type": "object",
            "properties": {
                "message": {"type": "string"}
            },
            "required": ["message"],
            "additionalProperties": false
        }))
        .expect("valid schema")
        .with_handler(|args, _deps| async move {
            let message = args
                .get("message")
                .and_then(|v| v.as_str())
                .ok_or_else(|| ToolError::Execution("message missing".to_string()))?;
            Ok(ToolOutcome::Done(message.to_string()))
        })
}

fn build_agent(responses: Vec<Result<ModelCompletion, ProviderError>>) -> Agent {
    Agent::builder()
        .model(ScriptedModel::new(responses))
        .tool(add_tool())
        .tool(done_tool())
        .build()
        .expect("agent builds")
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    let mut agent = build_agent(vec![
        Ok(ModelCompletion {
            text: Some("Working on it".to_string()),
            thinking: Some("Need arithmetic".to_string()),
            tool_calls: vec![ModelToolCall {
                id: "call_1".to_string(),
                name: "add".to_string(),
                arguments: json!({"a": 2, "b": 3}),
            }],
        }),
        Ok(ModelCompletion {
            text: None,
            thinking: None,
            tool_calls: vec![ModelToolCall {
                id: "call_2".to_string(),
                name: "done".to_string(),
                arguments: json!({"message": "2 + 3 = 5"}),
            }],
        }),
    ]);

    let final_response = agent.query("What is 2 + 3?").await?;
    println!("query final: {final_response}");

    let mut streaming_agent = build_agent(vec![
        Ok(ModelCompletion {
            text: Some("Streaming run".to_string()),
            thinking: Some("Will call add and done".to_string()),
            tool_calls: vec![ModelToolCall {
                id: "call_3".to_string(),
                name: "add".to_string(),
                arguments: json!({"a": 10, "b": 7}),
            }],
        }),
        Ok(ModelCompletion {
            text: None,
            thinking: None,
            tool_calls: vec![ModelToolCall {
                id: "call_4".to_string(),
                name: "done".to_string(),
                arguments: json!({"message": "10 + 7 = 17"}),
            }],
        }),
    ]);

    let stream = streaming_agent.query_stream("What is 10 + 7?");
    futures_util::pin_mut!(stream);
    while let Some(event) = stream.next().await {
        match event? {
            AgentEvent::Thinking { content } => println!("thinking: {content}"),
            AgentEvent::Text { content } => println!("text: {content}"),
            AgentEvent::ToolCall {
                tool,
                args_json,
                tool_call_id,
            } => println!("tool call [{tool_call_id}] {tool}: {args_json}"),
            AgentEvent::ToolResult {
                tool,
                result_text,
                tool_call_id,
                is_error,
            } => println!("tool result [{tool_call_id}] {tool}: {result_text} (error={is_error})"),
            AgentEvent::FinalResponse { content } => println!("stream final: {content}"),
        }
    }

    Ok(())
}
