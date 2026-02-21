use std::env;
use std::error::Error;
use std::fs;

use agent_sdk_rs::tools::claude_code::{SandboxContext, all_tools};
use agent_sdk_rs::{Agent, AgentEvent, AnthropicModel};
use futures_util::StreamExt;

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    let prompt = env::args().skip(1).collect::<Vec<_>>().join(" ");
    let prompt = if prompt.trim().is_empty() {
        "List all Rust files in this sandbox and summarize what they do".to_string()
    } else {
        prompt
    };

    let model_name =
        env::var("ANTHROPIC_MODEL").unwrap_or_else(|_| "claude-sonnet-4-5".to_string());
    let model = AnthropicModel::from_env(model_name)?;

    let sandbox_root = env::var("CLAUDE_CODE_SANDBOX").ok();
    let ctx = SandboxContext::create(sandbox_root)?;

    seed_workspace(&ctx)?;

    println!("sandbox: {}", ctx.root_dir().display());
    println!("session: {}", ctx.session_id());

    let mut agent = Agent::builder()
        .model(model)
        .tools(all_tools())
        .require_done_tool(true)
        .system_prompt(format!(
            "You are a coding assistant. Work only inside this sandbox: {}. Always call the done tool when complete.",
            ctx.working_dir().display()
        ))
        .dependency(ctx)
        .max_iterations(64)
        .build()?;

    let stream = agent.query_stream(prompt);
    futures_util::pin_mut!(stream);

    while let Some(event) = stream.next().await {
        match event? {
            AgentEvent::MessageStart { message_id, role } => {
                println!("message-start [{message_id}] {role:?}")
            }
            AgentEvent::MessageComplete {
                message_id,
                content,
            } => {
                if !content.trim().is_empty() {
                    println!(
                        "message-complete [{message_id}] {}",
                        truncate(&content, 180)
                    );
                }
            }
            AgentEvent::HiddenUserMessage { content } => {
                println!("hidden-user: {}", truncate(&content, 160));
            }
            AgentEvent::StepStart {
                step_id,
                title,
                step_number,
            } => {
                println!("step-start #{step_number} [{step_id}] {title}");
            }
            AgentEvent::ToolCall {
                tool,
                args_json,
                tool_call_id,
            } => {
                println!(
                    "tool-call [{tool_call_id}] {tool}: {}",
                    truncate(&args_json.to_string(), 160)
                );
            }
            AgentEvent::ToolResult {
                tool,
                result_text,
                tool_call_id,
                is_error,
            } => {
                println!(
                    "tool-result [{tool_call_id}] {tool} (error={is_error}): {}",
                    truncate(&result_text, 240)
                );
            }
            AgentEvent::StepComplete {
                step_id,
                status,
                duration_ms,
            } => {
                println!("step-complete [{step_id}] {status:?} ({duration_ms} ms)");
            }
            AgentEvent::Thinking { content } => {
                println!("thinking: {}", truncate(&content, 160));
            }
            AgentEvent::Text { content } => {
                println!("assistant: {}", truncate(&content, 200));
            }
            AgentEvent::FinalResponse { content } => {
                println!("\nfinal:\n{content}");
            }
        }
    }

    Ok(())
}

fn seed_workspace(ctx: &SandboxContext) -> Result<(), Box<dyn Error>> {
    fs::create_dir_all(ctx.root_dir().join("src"))?;
    fs::write(
        ctx.root_dir().join("src").join("main.rs"),
        "fn main() { println!(\"hello from sandbox\"); }\n",
    )?;
    fs::write(
        ctx.root_dir().join("src").join("lib.rs"),
        "pub fn add(a: i64, b: i64) -> i64 { a + b }\n",
    )?;
    Ok(())
}

fn truncate(text: &str, max: usize) -> String {
    if text.len() <= max {
        text.to_string()
    } else {
        format!("{}...", &text[..max])
    }
}
