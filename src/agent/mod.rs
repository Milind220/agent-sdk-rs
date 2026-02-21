use std::collections::HashMap;
use std::sync::Arc;

use async_stream::try_stream;
use futures_util::{Stream, StreamExt};

use crate::error::{AgentError, ToolError};
use crate::llm::{
    ChatModel, ModelCompletion, ModelMessage, ModelToolCall, ModelToolChoice, ModelToolDefinition,
};
use crate::tools::{DependencyMap, ToolOutcome, ToolSpec};

#[derive(Debug, Clone)]
pub struct AgentConfig {
    pub require_done_tool: bool,
    pub max_iterations: u32,
    pub system_prompt: Option<String>,
}

impl Default for AgentConfig {
    fn default() -> Self {
        Self {
            require_done_tool: false,
            max_iterations: 24,
            system_prompt: None,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum AgentEvent {
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
}

impl Agent {
    pub fn builder() -> AgentBuilder {
        AgentBuilder::default()
    }

    pub fn clear_history(&mut self) {
        self.history.clear();
    }

    pub fn messages_len(&self) -> usize {
        self.history.len()
    }

    pub async fn query(&mut self, user_message: impl Into<String>) -> Result<String, AgentError> {
        let stream = self.query_stream(user_message);
        futures_util::pin_mut!(stream);

        let mut final_response: Option<String> = None;

        while let Some(event) = stream.next().await {
            match event? {
                AgentEvent::FinalResponse { content } => final_response = Some(content),
                AgentEvent::Thinking { .. }
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
                    self.history
                        .push(ModelMessage::System(system_prompt.clone()));
                }
            }

            self.history.push(ModelMessage::User(user_message));

            let tool_definitions = self
                .tools
                .iter()
                .map(|tool| ModelToolDefinition {
                    name: tool.name().to_string(),
                    description: tool.description().to_string(),
                    parameters: tool.json_schema().clone(),
                })
                .collect::<Vec<_>>();

            let tool_choice = if tool_definitions.is_empty() {
                ModelToolChoice::None
            } else {
                ModelToolChoice::Auto
            };

            for _ in 0..self.config.max_iterations {
                let completion = self
                    .model
                    .invoke(&self.history, &tool_definitions, tool_choice.clone())
                    .await?;

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

                if completion.tool_calls.is_empty() {
                    if !self.config.require_done_tool {
                        let final_content = completion.text.unwrap_or_default();
                        yield AgentEvent::FinalResponse {
                            content: final_content,
                        };
                        return;
                    }
                    continue;
                }

                for tool_call in completion.tool_calls {
                    yield AgentEvent::ToolCall {
                        tool: tool_call.name.clone(),
                        args_json: tool_call.arguments.clone(),
                        tool_call_id: tool_call.id.clone(),
                    };

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
