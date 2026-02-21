use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

use async_stream::try_stream;
use futures_util::{Stream, StreamExt};
use tokio::time::{Duration, sleep};

use crate::error::{AgentError, ProviderError, ToolError};
use crate::llm::{
    ChatModel, ModelCompletion, ModelMessage, ModelToolCall, ModelToolChoice, ModelToolDefinition,
};
use crate::tools::{DependencyMap, ToolOutcome, ToolSpec};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AgentToolChoice {
    Auto,
    Required,
    None,
    Tool(String),
}

impl Default for AgentToolChoice {
    fn default() -> Self {
        Self::Auto
    }
}

#[derive(Debug, Clone)]
pub struct AgentConfig {
    pub require_done_tool: bool,
    pub max_iterations: u32,
    pub system_prompt: Option<String>,
    pub tool_choice: AgentToolChoice,
    pub llm_max_retries: u32,
    pub llm_retry_base_delay_ms: u64,
    pub llm_retry_max_delay_ms: u64,
    pub hidden_user_message_prompt: Option<String>,
}

impl Default for AgentConfig {
    fn default() -> Self {
        Self {
            require_done_tool: false,
            max_iterations: 24,
            system_prompt: None,
            tool_choice: AgentToolChoice::Auto,
            llm_max_retries: 5,
            llm_retry_base_delay_ms: 1_000,
            llm_retry_max_delay_ms: 60_000,
            hidden_user_message_prompt: None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AgentRole {
    User,
    Assistant,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StepStatus {
    Completed,
    Error,
}

#[derive(Debug, Clone, PartialEq)]
pub enum AgentEvent {
    MessageStart {
        message_id: String,
        role: AgentRole,
    },
    MessageComplete {
        message_id: String,
        content: String,
    },
    HiddenUserMessage {
        content: String,
    },
    StepStart {
        step_id: String,
        title: String,
        step_number: u32,
    },
    StepComplete {
        step_id: String,
        status: StepStatus,
        duration_ms: u128,
    },
    Thinking {
        content: String,
    },
    Text {
        content: String,
    },
    ToolCall {
        tool: String,
        args_json: serde_json::Value,
        tool_call_id: String,
    },
    ToolResult {
        tool: String,
        result_text: String,
        tool_call_id: String,
        is_error: bool,
    },
    FinalResponse {
        content: String,
    },
}

pub struct AgentBuilder {
    model: Option<Arc<dyn ChatModel>>,
    tools: Vec<ToolSpec>,
    config: AgentConfig,
    dependencies: DependencyMap,
    dependency_overrides: DependencyMap,
}

impl Default for AgentBuilder {
    fn default() -> Self {
        Self {
            model: None,
            tools: Vec::new(),
            config: AgentConfig::default(),
            dependencies: DependencyMap::new(),
            dependency_overrides: DependencyMap::new(),
        }
    }
}

impl AgentBuilder {
    pub fn model<M>(mut self, model: M) -> Self
    where
        M: ChatModel + 'static,
    {
        self.model = Some(Arc::new(model));
        self
    }

    pub fn tool(mut self, tool: ToolSpec) -> Self {
        self.tools.push(tool);
        self
    }

    pub fn tools(mut self, tools: Vec<ToolSpec>) -> Self {
        self.tools.extend(tools);
        self
    }

    pub fn config(mut self, config: AgentConfig) -> Self {
        self.config = config;
        self
    }

    pub fn system_prompt(mut self, system_prompt: impl Into<String>) -> Self {
        self.config.system_prompt = Some(system_prompt.into());
        self
    }

    pub fn require_done_tool(mut self, require_done_tool: bool) -> Self {
        self.config.require_done_tool = require_done_tool;
        self
    }

    pub fn max_iterations(mut self, max_iterations: u32) -> Self {
        self.config.max_iterations = max_iterations;
        self
    }

    pub fn tool_choice(mut self, tool_choice: AgentToolChoice) -> Self {
        self.config.tool_choice = tool_choice;
        self
    }

    pub fn llm_retry_config(
        mut self,
        max_retries: u32,
        base_delay_ms: u64,
        max_delay_ms: u64,
    ) -> Self {
        self.config.llm_max_retries = max_retries;
        self.config.llm_retry_base_delay_ms = base_delay_ms;
        self.config.llm_retry_max_delay_ms = max_delay_ms;
        self
    }

    pub fn hidden_user_message_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.config.hidden_user_message_prompt = Some(prompt.into());
        self
    }

    pub fn dependency<T>(self, value: T) -> Self
    where
        T: Send + Sync + 'static,
    {
        self.dependencies.insert(value);
        self
    }

    pub fn dependency_named<T>(self, key: impl Into<String>, value: T) -> Self
    where
        T: Send + Sync + 'static,
    {
        self.dependencies.insert_named(key, value);
        self
    }

    pub fn dependency_override<T>(self, value: T) -> Self
    where
        T: Send + Sync + 'static,
    {
        self.dependency_overrides.insert(value);
        self
    }

    pub fn dependency_override_named<T>(self, key: impl Into<String>, value: T) -> Self
    where
        T: Send + Sync + 'static,
    {
        self.dependency_overrides.insert_named(key, value);
        self
    }

    pub fn build(self) -> Result<Agent, AgentError> {
        let Some(model) = self.model else {
            return Err(AgentError::Config(
                "agent model must be configured via AgentBuilder::model(...)".to_string(),
            ));
        };

        let mut tool_map = HashMap::new();
        for tool in &self.tools {
            if tool_map
                .insert(tool.name().to_string(), tool.clone())
                .is_some()
            {
                return Err(AgentError::Config(format!(
                    "duplicate tool registered: {}",
                    tool.name()
                )));
            }
        }

        Ok(Agent {
            model,
            tools: self.tools,
            tool_map,
            config: self.config,
            dependencies: self.dependencies,
            dependency_overrides: self.dependency_overrides,
            history: Vec::new(),
            next_message_id: 0,
        })
    }
}

pub struct Agent {
    model: Arc<dyn ChatModel>,
    tools: Vec<ToolSpec>,
    tool_map: HashMap<String, ToolSpec>,
    config: AgentConfig,
    dependencies: DependencyMap,
    dependency_overrides: DependencyMap,
    history: Vec<ModelMessage>,
    next_message_id: u64,
}

impl Agent {
    pub fn builder() -> AgentBuilder {
        AgentBuilder::default()
    }

    pub fn clear_history(&mut self) {
        self.history.clear();
        self.next_message_id = 0;
    }

    pub fn load_history(&mut self, messages: Vec<ModelMessage>) {
        self.next_message_id = messages.len() as u64;
        self.history = messages;
    }

    pub fn messages_len(&self) -> usize {
        self.history.len()
    }

    pub fn messages(&self) -> &[ModelMessage] {
        &self.history
    }

    pub async fn query(&mut self, user_message: impl Into<String>) -> Result<String, AgentError> {
        let stream = self.query_stream(user_message);
        futures_util::pin_mut!(stream);

        let mut final_response: Option<String> = None;

        while let Some(event) = stream.next().await {
            match event? {
                AgentEvent::FinalResponse { content } => final_response = Some(content),
                AgentEvent::MessageStart { .. }
                | AgentEvent::MessageComplete { .. }
                | AgentEvent::HiddenUserMessage { .. }
                | AgentEvent::StepStart { .. }
                | AgentEvent::StepComplete { .. }
                | AgentEvent::Thinking { .. }
                | AgentEvent::Text { .. }
                | AgentEvent::ToolCall { .. }
                | AgentEvent::ToolResult { .. } => {}
            }
        }

        final_response.ok_or(AgentError::MissingFinalResponse)
    }

    pub fn query_stream(
        &mut self,
        user_message: impl Into<String>,
    ) -> impl Stream<Item = Result<AgentEvent, AgentError>> + '_ {
        let user_message = user_message.into();

        try_stream! {
            if self.history.is_empty() {
                if let Some(system_prompt) = &self.config.system_prompt {
                    self.history.push(ModelMessage::System(system_prompt.clone()));
                }
            }

            let user_message_id = self.next_message_id(AgentRole::User);
            yield AgentEvent::MessageStart {
                message_id: user_message_id.clone(),
                role: AgentRole::User,
            };
            self.history.push(ModelMessage::User(user_message.clone()));
            yield AgentEvent::MessageComplete {
                message_id: user_message_id,
                content: user_message,
            };

            let tool_definitions = self
                .tools
                .iter()
                .map(|tool| ModelToolDefinition {
                    name: tool.name().to_string(),
                    description: tool.description().to_string(),
                    parameters: tool.json_schema().clone(),
                })
                .collect::<Vec<_>>();

            let tool_choice = self.resolve_tool_choice(!tool_definitions.is_empty());
            let mut hidden_prompt_injected = false;

            for _ in 0..self.config.max_iterations {
                let completion = self
                    .invoke_with_retry(&tool_definitions, tool_choice.clone())
                    .await?;

                let assistant_message_id = self.next_message_id(AgentRole::Assistant);
                yield AgentEvent::MessageStart {
                    message_id: assistant_message_id.clone(),
                    role: AgentRole::Assistant,
                };

                if let Some(thinking) = completion.thinking.clone() {
                    yield AgentEvent::Thinking { content: thinking };
                }

                self.append_assistant_message(&completion);

                if let Some(text) = completion.text.clone() {
                    if !text.is_empty() {
                        yield AgentEvent::Text {
                            content: text.clone(),
                        };
                    }
                }

                let assistant_content = completion.text.clone().unwrap_or_default();
                yield AgentEvent::MessageComplete {
                    message_id: assistant_message_id,
                    content: assistant_content.clone(),
                };

                if completion.tool_calls.is_empty() {
                    if !self.config.require_done_tool {
                        if !hidden_prompt_injected {
                            if let Some(hidden_prompt) = self.config.hidden_user_message_prompt.clone() {
                                hidden_prompt_injected = true;
                                self.history.push(ModelMessage::User(hidden_prompt.clone()));
                                yield AgentEvent::HiddenUserMessage {
                                    content: hidden_prompt,
                                };
                                continue;
                            }
                        }

                        yield AgentEvent::FinalResponse {
                            content: completion.text.unwrap_or_default(),
                        };
                        return;
                    }
                    continue;
                }

                let mut step_number = 0_u32;
                for tool_call in completion.tool_calls {
                    step_number += 1;
                    yield AgentEvent::StepStart {
                        step_id: tool_call.id.clone(),
                        title: tool_call.name.clone(),
                        step_number,
                    };

                    yield AgentEvent::ToolCall {
                        tool: tool_call.name.clone(),
                        args_json: tool_call.arguments.clone(),
                        tool_call_id: tool_call.id.clone(),
                    };

                    let step_start = Instant::now();
                    let execution = self.execute_tool_call(&tool_call).await;
                    self.history.push(ModelMessage::ToolResult {
                        tool_call_id: tool_call.id.clone(),
                        tool_name: tool_call.name.clone(),
                        content: execution.result_text.clone(),
                        is_error: execution.is_error,
                    });

                    yield AgentEvent::ToolResult {
                        tool: tool_call.name.clone(),
                        result_text: execution.result_text.clone(),
                        tool_call_id: tool_call.id.clone(),
                        is_error: execution.is_error,
                    };

                    yield AgentEvent::StepComplete {
                        step_id: tool_call.id.clone(),
                        status: if execution.is_error {
                            StepStatus::Error
                        } else {
                            StepStatus::Completed
                        },
                        duration_ms: step_start.elapsed().as_millis(),
                    };

                    if let Some(done_message) = execution.done_message {
                        yield AgentEvent::FinalResponse {
                            content: done_message,
                        };
                        return;
                    }
                }
            }

            Err::<(), AgentError>(AgentError::MaxIterationsReached {
                max_iterations: self.config.max_iterations,
            })?;
        }
    }

    fn next_message_id(&mut self, role: AgentRole) -> String {
        self.next_message_id += 1;
        let role_label = match role {
            AgentRole::User => "user",
            AgentRole::Assistant => "assistant",
        };
        format!("msg_{}_{}", self.next_message_id, role_label)
    }

    fn resolve_tool_choice(&self, has_tools: bool) -> ModelToolChoice {
        if !has_tools {
            return ModelToolChoice::None;
        }

        match &self.config.tool_choice {
            AgentToolChoice::Auto => ModelToolChoice::Auto,
            AgentToolChoice::Required => ModelToolChoice::Required,
            AgentToolChoice::None => ModelToolChoice::None,
            AgentToolChoice::Tool(name) => ModelToolChoice::Tool(name.clone()),
        }
    }

    async fn invoke_with_retry(
        &self,
        tool_definitions: &[ModelToolDefinition],
        tool_choice: ModelToolChoice,
    ) -> Result<ModelCompletion, AgentError> {
        let max_retries = self.config.llm_max_retries.max(1);
        for attempt in 0..max_retries {
            match self
                .model
                .invoke(&self.history, tool_definitions, tool_choice.clone())
                .await
            {
                Ok(completion) => return Ok(completion),
                Err(err) => {
                    let should_retry =
                        is_retryable_provider_error(&err) && (attempt + 1) < max_retries;
                    if !should_retry {
                        return Err(AgentError::Provider(err));
                    }

                    let delay_ms = retry_delay_ms(
                        attempt,
                        self.config.llm_retry_base_delay_ms,
                        self.config.llm_retry_max_delay_ms,
                    );
                    sleep(Duration::from_millis(delay_ms)).await;
                }
            }
        }

        Err(AgentError::Config(
            "retry loop failed unexpectedly".to_string(),
        ))
    }

    fn append_assistant_message(&mut self, completion: &ModelCompletion) {
        self.history.push(ModelMessage::Assistant {
            content: completion.text.clone(),
            tool_calls: completion.tool_calls.clone(),
        });
    }

    async fn execute_tool_call(&self, tool_call: &ModelToolCall) -> ToolExecutionResult {
        let Some(tool) = self.tool_map.get(&tool_call.name) else {
            return ToolExecutionResult {
                result_text: format!("Unknown tool '{}'.", tool_call.name),
                is_error: true,
                done_message: None,
            };
        };

        let runtime_dependencies = self.dependencies.merged_with(&self.dependency_overrides);

        match tool
            .execute(tool_call.arguments.clone(), &runtime_dependencies)
            .await
        {
            Ok(ToolOutcome::Text(text)) => ToolExecutionResult {
                result_text: text,
                is_error: false,
                done_message: None,
            },
            Ok(ToolOutcome::Done(message)) => ToolExecutionResult {
                result_text: format!("Task completed: {message}"),
                is_error: false,
                done_message: Some(message),
            },
            Err(err) => ToolExecutionResult {
                result_text: format_tool_error(err),
                is_error: true,
                done_message: None,
            },
        }
    }
}

fn is_retryable_provider_error(err: &ProviderError) -> bool {
    match err {
        ProviderError::Request(_) => true,
        ProviderError::Response(_) => false,
    }
}

fn retry_delay_ms(attempt: u32, base_delay_ms: u64, max_delay_ms: u64) -> u64 {
    let mut delay = base_delay_ms;
    for _ in 0..attempt {
        delay = delay.saturating_mul(2);
    }
    delay.min(max_delay_ms)
}

fn format_tool_error(err: ToolError) -> String {
    err.to_string()
}

struct ToolExecutionResult {
    result_text: String,
    is_error: bool,
    done_message: Option<String>,
}

pub async fn query(
    agent: &mut Agent,
    user_message: impl Into<String>,
) -> Result<String, AgentError> {
    agent.query(user_message).await
}

pub fn query_stream(
    agent: &mut Agent,
    user_message: impl Into<String>,
) -> impl Stream<Item = Result<AgentEvent, AgentError>> + '_ {
    agent.query_stream(user_message)
}

#[cfg(test)]
mod tests;
