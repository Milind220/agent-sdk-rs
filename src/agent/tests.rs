use std::collections::VecDeque;
use std::sync::Mutex;

use async_trait::async_trait;
use futures_util::StreamExt;
use serde_json::json;

use super::*;
use crate::error::ProviderError;
use crate::tools::{ToolOutcome, ToolSpec};

#[derive(Default)]
struct MockModel {
    responses: Mutex<VecDeque<Result<ModelCompletion, ProviderError>>>,
}

impl MockModel {
    fn with_responses(responses: Vec<Result<ModelCompletion, ProviderError>>) -> Self {
        Self {
            responses: Mutex::new(VecDeque::from(responses)),
        }
    }
}

#[async_trait]
impl ChatModel for MockModel {
    async fn invoke(
        &self,
        _messages: &[ModelMessage],
        _tools: &[ModelToolDefinition],
        _tool_choice: ModelToolChoice,
    ) -> Result<ModelCompletion, ProviderError> {
        let mut guard = self.responses.lock().expect("lock poisoned");
        guard.pop_front().unwrap_or_else(|| {
            Err(ProviderError::Response(
                "no more mock model responses".to_string(),
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
    ToolSpec::new("done", "complete task")
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

fn fail_tool() -> ToolSpec {
    ToolSpec::new("fail", "always fail")
        .with_schema(json!({
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": false
        }))
        .expect("valid schema")
        .with_handler(|_args, _deps| async move { Err(ToolError::Execution("boom".to_string())) })
}

fn tool_call(id: &str, name: &str, arguments: serde_json::Value) -> ModelToolCall {
    ModelToolCall {
        id: id.to_string(),
        name: name.to_string(),
        arguments,
    }
}

#[tokio::test]
async fn query_returns_no_tool_response() {
    let model = MockModel::with_responses(vec![Ok(ModelCompletion {
        text: Some("hello".to_string()),
        thinking: None,
        tool_calls: vec![],
    })]);

    let mut agent = Agent::builder().model(model).build().expect("agent builds");
    let response = agent.query("hi").await.expect("query succeeds");

    assert_eq!(response, "hello");
}

#[tokio::test]
async fn tool_call_then_final_response_flow() {
    let model = MockModel::with_responses(vec![
        Ok(ModelCompletion {
            text: None,
            thinking: None,
            tool_calls: vec![tool_call("call_1", "add", json!({"a": 2, "b": 3}))],
        }),
        Ok(ModelCompletion {
            text: Some("all done".to_string()),
            thinking: None,
            tool_calls: vec![],
        }),
    ]);

    let mut agent = Agent::builder()
        .model(model)
        .tool(add_tool())
        .build()
        .expect("agent builds");

    let events = agent
        .query_stream("add")
        .collect::<Vec<_>>()
        .await
        .into_iter()
        .collect::<Result<Vec<_>, _>>()
        .expect("events ok");

    assert_eq!(events.len(), 4);
    assert!(matches!(events[0], AgentEvent::ToolCall { .. }));
    assert!(matches!(
        events[1],
        AgentEvent::ToolResult {
            is_error: false,
            ..
        }
    ));
    assert_eq!(
        events[2],
        AgentEvent::Text {
            content: "all done".to_string()
        }
    );
    assert_eq!(
        events[3],
        AgentEvent::FinalResponse {
            content: "all done".to_string()
        }
    );
}

#[tokio::test]
async fn done_tool_stops_immediately() {
    let model = MockModel::with_responses(vec![Ok(ModelCompletion {
        text: None,
        thinking: None,
        tool_calls: vec![tool_call("call_2", "done", json!({"message": "finished"}))],
    })]);

    let mut agent = Agent::builder()
        .model(model)
        .tool(done_tool())
        .build()
        .expect("agent builds");

    let response = agent.query("wrap").await.expect("query succeeds");
    assert_eq!(response, "finished");
}

#[tokio::test]
async fn require_done_mode_keeps_looping_until_max_iterations() {
    let model = MockModel::with_responses(vec![
        Ok(ModelCompletion {
            text: Some("not done".to_string()),
            thinking: None,
            tool_calls: vec![],
        }),
        Ok(ModelCompletion {
            text: Some("still not done".to_string()),
            thinking: None,
            tool_calls: vec![],
        }),
    ]);

    let mut agent = Agent::builder()
        .model(model)
        .require_done_tool(true)
        .max_iterations(2)
        .build()
        .expect("agent builds");

    let err = agent.query("continue").await.expect_err("must fail");
    assert!(matches!(err, AgentError::MaxIterationsReached { .. }));
}

#[tokio::test]
async fn max_iterations_error_when_tool_loop_never_finishes() {
    let model = MockModel::with_responses(vec![Ok(ModelCompletion {
        text: None,
        thinking: None,
        tool_calls: vec![tool_call("call_3", "add", json!({"a": 1, "b": 1}))],
    })]);

    let mut agent = Agent::builder()
        .model(model)
        .tool(add_tool())
        .max_iterations(1)
        .build()
        .expect("agent builds");

    let err = agent.query("loop").await.expect_err("must fail");
    assert!(matches!(err, AgentError::MaxIterationsReached { .. }));
}

#[tokio::test]
async fn tool_error_emits_error_result_and_still_finishes() {
    let model = MockModel::with_responses(vec![
        Ok(ModelCompletion {
            text: None,
            thinking: None,
            tool_calls: vec![tool_call("call_4", "fail", json!({}))],
        }),
        Ok(ModelCompletion {
            text: Some("fallback".to_string()),
            thinking: None,
            tool_calls: vec![],
        }),
    ]);

    let mut agent = Agent::builder()
        .model(model)
        .tool(fail_tool())
        .build()
        .expect("agent builds");

    let events = agent
        .query_stream("try")
        .collect::<Vec<_>>()
        .await
        .into_iter()
        .collect::<Result<Vec<_>, _>>()
        .expect("events ok");

    assert!(
        events
            .iter()
            .any(|event| { matches!(event, AgentEvent::ToolResult { is_error: true, .. }) })
    );

    assert_eq!(
        events.last(),
        Some(&AgentEvent::FinalResponse {
            content: "fallback".to_string()
        })
    );
}

#[tokio::test]
async fn dependency_override_is_used_for_tool_execution() {
    let model = MockModel::with_responses(vec![
        Ok(ModelCompletion {
            text: None,
            thinking: None,
            tool_calls: vec![tool_call("call_5", "read_dep", json!({}))],
        }),
        Ok(ModelCompletion {
            text: Some("done".to_string()),
            thinking: None,
            tool_calls: vec![],
        }),
    ]);

    let dep_tool = ToolSpec::new("read_dep", "read number")
        .with_schema(json!({
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": false
        }))
        .expect("valid schema")
        .with_handler(|_args, deps| {
            let value = deps
                .get::<u32>()
                .ok_or(ToolError::MissingDependency("u32"))
                .map(|v| *v)
                .unwrap_or(0);
            async move { Ok(ToolOutcome::Text(value.to_string())) }
        });

    let mut agent = Agent::builder()
        .model(model)
        .tool(dep_tool)
        .dependency(1_u32)
        .dependency_override(9_u32)
        .build()
        .expect("agent builds");

    let events = agent
        .query_stream("dep")
        .collect::<Vec<_>>()
        .await
        .into_iter()
        .collect::<Result<Vec<_>, _>>()
        .expect("events ok");

    assert!(events.iter().any(|event| {
        matches!(
            event,
            AgentEvent::ToolResult {
                result_text,
                is_error: false,
                ..
            } if result_text == "9"
        )
    }));
}
