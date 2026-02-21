use thiserror::Error;

#[derive(Debug, Error)]
pub enum SchemaError {
    #[error("tool schema must be a JSON object")]
    SchemaNotObject,
    #[error("tool schema must declare type=object")]
    RootTypeMustBeObject,
    #[error("required must be an array of strings")]
    InvalidRequired,
}

#[derive(Debug, Error)]
pub enum ToolError {
    #[error("tool not found: {0}")]
    NotFound(String),
    #[error("invalid tool arguments for {tool}: {message}")]
    InvalidArguments { tool: String, message: String },
    #[error("dependency missing: {0}")]
    MissingDependency(&'static str),
    #[error("tool execution failed: {0}")]
    Execution(String),
    #[error(transparent)]
    Schema(#[from] SchemaError),
}

#[derive(Debug, Error)]
pub enum ProviderError {
    #[error("provider request failed: {0}")]
    Request(String),
    #[error("provider response invalid: {0}")]
    Response(String),
}

#[derive(Debug, Error)]
pub enum AgentError {
    #[error(transparent)]
    Tool(#[from] ToolError),
    #[error(transparent)]
    Provider(#[from] ProviderError),
    #[error("max iterations reached ({max_iterations})")]
    MaxIterationsReached { max_iterations: u32 },
    #[error("agent stream ended without final response")]
    MissingFinalResponse,
    #[error("agent configuration error: {0}")]
    Config(String),
}
