use std::collections::VecDeque;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use futures_util::StreamExt;
use serde_json::json;

use super::*;
use crate::error::ProviderError;
use crate::tools::{ToolOutcome, ToolSpec};

#[derive(Default)]
struct MockModel {
    responses: Arc<Mutex<VecDeque<Result<ModelCompletion, ProviderError>>>>,
    invocations: Arc<AtomicUsize>,
    seen_tool_choices: Arc<Mutex<Vec<ModelToolChoice>>>,
    seen_message_batches: Arc<Mutex<Vec<Vec<ModelMessage>>>>,
}

impl MockModel {
    fn with_responses(responses: Vec<Result<ModelCompletion, ProviderError>>) -> Self {
        Self {
            responses: Arc::new(Mutex::new(VecDeque::from(responses))),
            invocations: Arc::new(AtomicUsize::new(0)),
            seen_tool_choices: Arc::new(Mutex::new(Vec::new())),
            seen_message_batches: Arc::new(Mutex::new(Vec::new())),
        }
    }
}

#[async_trait]
impl ChatModel for MockModel {
    async fn invoke(
        &self,
        messages: &[ModelMessage],
        _tools: &[ModelToolDefinition],
        tool_choice: ModelToolChoice,
    ) -> Result<ModelCompletion, ProviderError> {
        self.invocations.fetch_add(1, Ordering::SeqCst);
        self.seen_tool_choices
            .lock()
            .expect("tool choices lock")
            .push(tool_choice);
        self.seen_message_batches
            .lock()
            .expect("message batches lock")
            .push(messages.to_vec());

        let mut guard = self.responses.lock().expect("responses lock poisoned");
        guard.pop_front().unwrap_or_else(|| {
            Err(ProviderError::Response(
                "no more mock model responses".to_string(),
            ))
        })
    }
}

fn completion(text: Option<&str>, tool_calls: Vec<ModelToolCall>) -> ModelCompletion {
    ModelCompletion {
        text: text.map(ToString::to_string),
        thinking: None,
        tool_calls,
        usage: None,
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
    let model = MockModel::with_responses(vec![Ok(completion(Some("hello"), vec![]))]);

    let mut agent = Agent::builder().model(model).build().expect("agent builds");
    let response = agent.query("hi").await.expect("query succeeds");

    assert_eq!(response, "hello");
}

#[tokio::test]
async fn query_stream_emits_message_and_step_events() {
    let model = MockModel::with_responses(vec![
        Ok(completion(
            Some("working"),
            vec![tool_call("call_1", "add", json!({"a": 2, "b": 3}))],
        )),
        Ok(completion(Some("all done"), vec![])),
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

    assert!(events.iter().any(|e| matches!(
        e,
        AgentEvent::MessageStart {
            role: AgentRole::User,
            ..
        }
    )));
    assert!(
        events
            .iter()
            .any(|e| matches!(e, AgentEvent::StepStart { step_id, .. } if step_id == "call_1"))
    );
    assert!(events.iter().any(|e| matches!(
        e,
        AgentEvent::StepComplete {
            status: StepStatus::Completed,
            ..
        }
    )));

    assert_eq!(
        events.last(),
        Some(&AgentEvent::FinalResponse {
            content: "all done".to_string()
        })
    );
}

#[tokio::test]
async fn done_tool_stops_immediately() {
    let model = MockModel::with_responses(vec![Ok(completion(
        None,
        vec![tool_call("call_2", "done", json!({"message": "finished"}))],
    ))]);

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
        Ok(completion(Some("not done"), vec![])),
        Ok(completion(Some("still not done"), vec![])),
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
    let model = MockModel::with_responses(vec![Ok(completion(
        None,
        vec![tool_call("call_3", "add", json!({"a": 1, "b": 1}))],
    ))]);

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
async fn tool_error_emits_error_result_and_step_error() {
    let model = MockModel::with_responses(vec![
        Ok(completion(
            None,
            vec![tool_call("call_4", "fail", json!({}))],
        )),
        Ok(completion(Some("fallback"), vec![])),
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
            .any(|event| matches!(event, AgentEvent::ToolResult { is_error: true, .. }))
    );

    assert!(events.iter().any(|event| matches!(
        event,
        AgentEvent::StepComplete {
            status: StepStatus::Error,
            ..
        }
    )));

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
        Ok(completion(
            None,
            vec![tool_call("call_5", "read_dep", json!({}))],
        )),
        Ok(completion(Some("done"), vec![])),
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

#[tokio::test]
async fn hidden_user_prompt_is_emitted_once() {
    let model = MockModel::with_responses(vec![
        Ok(completion(Some("not complete"), vec![])),
        Ok(completion(Some("final"), vec![])),
    ]);

    let mut agent = Agent::builder()
        .model(model)
        .hidden_user_message_prompt("You still have incomplete todos")
        .build()
        .expect("agent builds");

    let events = agent
        .query_stream("start")
        .collect::<Vec<_>>()
        .await
        .into_iter()
        .collect::<Result<Vec<_>, _>>()
        .expect("events ok");

    let hidden_count = events
        .iter()
        .filter(|e| matches!(e, AgentEvent::HiddenUserMessage { .. }))
        .count();
    assert_eq!(hidden_count, 1);

    assert_eq!(
        events.last(),
        Some(&AgentEvent::FinalResponse {
            content: "final".to_string()
        })
    );
}

#[tokio::test]
async fn retries_request_errors_then_succeeds() {
    let model = MockModel::with_responses(vec![
        Err(ProviderError::Request("timeout".to_string())),
        Ok(completion(Some("ok"), vec![])),
    ]);

    let invocations = model.invocations.clone();

    let mut agent = Agent::builder()
        .model(model)
        .llm_retry_config(2, 0, 0)
        .build()
        .expect("agent builds");

    let response = agent.query("retry").await.expect("query succeeds");
    assert_eq!(response, "ok");
    assert_eq!(invocations.load(Ordering::SeqCst), 2);
}

#[tokio::test]
async fn load_history_and_tool_choice_are_applied() {
    let model = MockModel::with_responses(vec![Ok(completion(Some("done"), vec![]))]);
    let seen_tool_choices = model.seen_tool_choices.clone();
    let seen_batches = model.seen_message_batches.clone();

    let mut agent = Agent::builder()
        .model(model)
        .tool(add_tool())
        .tool_choice(AgentToolChoice::Required)
        .build()
        .expect("agent builds");

    agent.load_history(vec![
        ModelMessage::System("sys".to_string()),
        ModelMessage::User("old".to_string()),
    ]);

    let response = agent.query("new").await.expect("query succeeds");
    assert_eq!(response, "done");

    assert_eq!(
        seen_tool_choices.lock().expect("lock").first(),
        Some(&ModelToolChoice::Required)
    );

    let first_batch = seen_batches.lock().expect("lock").first().cloned().unwrap();
    assert!(matches!(first_batch[0], ModelMessage::System(_)));
    assert!(matches!(first_batch[1], ModelMessage::User(_)));
    assert!(matches!(first_batch[2], ModelMessage::User(_)));
}
