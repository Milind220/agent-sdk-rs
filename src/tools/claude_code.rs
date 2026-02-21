use std::fs;
use std::path::{Component, Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

use glob::glob;
use regex::Regex;
use serde::{Deserialize, Serialize};
use serde_json::json;
use tokio::process::Command;
use tokio::time::{Duration, timeout};
use walkdir::WalkDir;

use crate::error::ToolError;
use crate::tools::{DependencyMap, ToolOutcome, ToolSpec};

#[derive(Debug, Clone)]
pub struct SandboxContext {
    root_dir: PathBuf,
    working_dir: PathBuf,
    session_id: String,
    todos: Arc<Mutex<Vec<TodoItem>>>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct TodoItem {
    pub content: String,
    pub status: String,
    #[serde(default)]
    pub active_form: Option<String>,
}

impl SandboxContext {
    pub fn create(root_dir: Option<impl Into<PathBuf>>) -> Result<Self, std::io::Error> {
        let session_id = short_session_id();
        let root = if let Some(path) = root_dir {
            path.into()
        } else {
            PathBuf::from(format!("./tmp/sandbox/{session_id}"))
        };

        fs::create_dir_all(&root)?;
        let root_dir = root.canonicalize()?;

        Ok(Self {
            working_dir: root_dir.clone(),
            root_dir,
            session_id,
            todos: Arc::new(Mutex::new(Vec::new())),
        })
    }

    pub fn root_dir(&self) -> &Path {
        &self.root_dir
    }

    pub fn working_dir(&self) -> &Path {
        &self.working_dir
    }

    pub fn session_id(&self) -> &str {
        &self.session_id
    }

    pub fn resolve_path(&self, path: impl AsRef<Path>) -> Result<PathBuf, String> {
        let candidate = path.as_ref();
        let unresolved = if candidate.is_absolute() {
            candidate.to_path_buf()
        } else {
            self.working_dir.join(candidate)
        };
        let resolved = normalize_absolute_path(&unresolved);

        if !resolved.starts_with(&self.root_dir) {
            return Err(format!(
                "Path escapes sandbox: {} -> {}",
                candidate.display(),
                resolved.display()
            ));
        }

        Ok(resolved)
    }

    fn read_todos(&self) -> Vec<TodoItem> {
        self.todos.lock().expect("todo lock poisoned").clone()
    }

    fn write_todos(&self, todos: Vec<TodoItem>) {
        *self.todos.lock().expect("todo lock poisoned") = todos;
    }
}

fn normalize_absolute_path(path: &Path) -> PathBuf {
    let mut normalized = PathBuf::new();
    for component in path.components() {
        match component {
            Component::Prefix(prefix) => normalized.push(prefix.as_os_str()),
            Component::RootDir => normalized.push(Path::new("/")),
            Component::CurDir => {}
            Component::ParentDir => {
                let _ = normalized.pop();
            }
            Component::Normal(part) => normalized.push(part),
        }
    }
    normalized
}

pub fn all_tools() -> Vec<ToolSpec> {
    vec![
        bash_tool(),
        read_tool(),
        write_tool(),
        edit_tool(),
        glob_search_tool(),
        grep_tool(),
        todo_read_tool(),
        todo_write_tool(),
        done_tool(),
    ]
}

pub fn bash_tool() -> ToolSpec {
    ToolSpec::new("bash", "Execute a shell command and return output")
        .with_schema(json!({
            "type": "object",
            "properties": {
                "command": {"type": "string"},
                "timeout": {"type": "integer"}
            },
            "required": ["command"],
            "additionalProperties": false
        }))
        .expect("valid schema")
        .with_handler(|args, deps| {
            let command = args
                .get("command")
                .and_then(|v| v.as_str())
                .unwrap_or_default()
                .to_string();
            let timeout_secs = args.get("timeout").and_then(|v| v.as_u64()).unwrap_or(30);
            let ctx = get_ctx(deps);

            async move {
                let ctx = match ctx {
                    Ok(ctx) => ctx,
                    Err(err) => return Ok(ToolOutcome::Text(format!("Error: {err}"))),
                };

                let output = timeout(
                    Duration::from_secs(timeout_secs),
                    Command::new("sh")
                        .arg("-lc")
                        .arg(command)
                        .current_dir(ctx.working_dir())
                        .output(),
                )
                .await;

                match output {
                    Ok(Ok(out)) => {
                        let stdout = String::from_utf8_lossy(&out.stdout);
                        let stderr = String::from_utf8_lossy(&out.stderr);
                        let mut rendered = format!("{}{}", stdout, stderr).trim().to_string();
                        if rendered.is_empty() {
                            rendered = "(no output)".to_string();
                        }
                        Ok(ToolOutcome::Text(rendered))
                    }
                    Ok(Err(err)) => Ok(ToolOutcome::Text(format!("Error: {err}"))),
                    Err(_) => Ok(ToolOutcome::Text(format!(
                        "Command timed out after {timeout_secs}s"
                    ))),
                }
            }
        })
}

pub fn read_tool() -> ToolSpec {
    ToolSpec::new("read", "Read contents of a file")
        .with_schema(json!({
            "type": "object",
            "properties": {
                "file_path": {"type": "string"}
            },
            "required": ["file_path"],
            "additionalProperties": false
        }))
        .expect("valid schema")
        .with_handler(|args, deps| {
            let file_path = args
                .get("file_path")
                .and_then(|v| v.as_str())
                .unwrap_or_default()
                .to_string();
            let ctx = get_ctx(deps);

            async move {
                let ctx = match ctx {
                    Ok(ctx) => ctx,
                    Err(err) => return Ok(ToolOutcome::Text(format!("Error: {err}"))),
                };

                let path = match ctx.resolve_path(&file_path) {
                    Ok(path) => path,
                    Err(err) => return Ok(ToolOutcome::Text(format!("Security error: {err}"))),
                };

                if !path.exists() {
                    return Ok(ToolOutcome::Text(format!("File not found: {file_path}")));
                }
                if path.is_dir() {
                    return Ok(ToolOutcome::Text(format!(
                        "Path is a directory: {file_path}"
                    )));
                }

                match fs::read_to_string(path) {
                    Ok(content) => {
                        let numbered = content
                            .lines()
                            .enumerate()
                            .map(|(idx, line)| format!("{:4}  {}", idx + 1, line))
                            .collect::<Vec<_>>()
                            .join("\n");
                        Ok(ToolOutcome::Text(numbered))
                    }
                    Err(err) => Ok(ToolOutcome::Text(format!("Error reading file: {err}"))),
                }
            }
        })
}

pub fn write_tool() -> ToolSpec {
    ToolSpec::new("write", "Write content to a file")
        .with_schema(json!({
            "type": "object",
            "properties": {
                "file_path": {"type": "string"},
                "content": {"type": "string"}
            },
            "required": ["file_path", "content"],
            "additionalProperties": false
        }))
        .expect("valid schema")
        .with_handler(|args, deps| {
            let file_path = args
                .get("file_path")
                .and_then(|v| v.as_str())
                .unwrap_or_default()
                .to_string();
            let content = args
                .get("content")
                .and_then(|v| v.as_str())
                .unwrap_or_default()
                .to_string();
            let ctx = get_ctx(deps);

            async move {
                let ctx = match ctx {
                    Ok(ctx) => ctx,
                    Err(err) => return Ok(ToolOutcome::Text(format!("Error: {err}"))),
                };

                let path = match ctx.resolve_path(&file_path) {
                    Ok(path) => path,
                    Err(err) => return Ok(ToolOutcome::Text(format!("Security error: {err}"))),
                };

                if let Some(parent) = path.parent() {
                    if let Err(err) = fs::create_dir_all(parent) {
                        return Ok(ToolOutcome::Text(format!("Error writing file: {err}")));
                    }
                }

                match fs::write(path, content.as_bytes()) {
                    Ok(_) => Ok(ToolOutcome::Text(format!(
                        "Wrote {} bytes to {file_path}",
                        content.len()
                    ))),
                    Err(err) => Ok(ToolOutcome::Text(format!("Error writing file: {err}"))),
                }
            }
        })
}

pub fn edit_tool() -> ToolSpec {
    ToolSpec::new("edit", "Replace text in a file")
        .with_schema(json!({
            "type": "object",
            "properties": {
                "file_path": {"type": "string"},
                "old_string": {"type": "string"},
                "new_string": {"type": "string"}
            },
            "required": ["file_path", "old_string", "new_string"],
            "additionalProperties": false
        }))
        .expect("valid schema")
        .with_handler(|args, deps| {
            let file_path = args
                .get("file_path")
                .and_then(|v| v.as_str())
                .unwrap_or_default()
                .to_string();
            let old_string = args
                .get("old_string")
                .and_then(|v| v.as_str())
                .unwrap_or_default()
                .to_string();
            let new_string = args
                .get("new_string")
                .and_then(|v| v.as_str())
                .unwrap_or_default()
                .to_string();
            let ctx = get_ctx(deps);

            async move {
                let ctx = match ctx {
                    Ok(ctx) => ctx,
                    Err(err) => return Ok(ToolOutcome::Text(format!("Error: {err}"))),
                };

                let path = match ctx.resolve_path(&file_path) {
                    Ok(path) => path,
                    Err(err) => return Ok(ToolOutcome::Text(format!("Security error: {err}"))),
                };

                if !path.exists() {
                    return Ok(ToolOutcome::Text(format!("File not found: {file_path}")));
                }

                let content = match fs::read_to_string(&path) {
                    Ok(content) => content,
                    Err(err) => return Ok(ToolOutcome::Text(format!("Error editing file: {err}"))),
                };

                if !content.contains(&old_string) {
                    return Ok(ToolOutcome::Text(format!(
                        "String not found in {file_path}"
                    )));
                }

                let count = content.matches(&old_string).count();
                let updated = content.replace(&old_string, &new_string);
                match fs::write(&path, updated.as_bytes()) {
                    Ok(_) => Ok(ToolOutcome::Text(format!(
                        "Replaced {count} occurrence(s) in {file_path}"
                    ))),
                    Err(err) => Ok(ToolOutcome::Text(format!("Error editing file: {err}"))),
                }
            }
        })
}

pub fn glob_search_tool() -> ToolSpec {
    ToolSpec::new("glob_search", "Find files matching a glob pattern")
        .with_schema(json!({
            "type": "object",
            "properties": {
                "pattern": {"type": "string"},
                "path": {"type": "string"}
            },
            "required": ["pattern"],
            "additionalProperties": false
        }))
        .expect("valid schema")
        .with_handler(|args, deps| {
            let pattern = args
                .get("pattern")
                .and_then(|v| v.as_str())
                .unwrap_or_default()
                .to_string();
            let path = args
                .get("path")
                .and_then(|v| v.as_str())
                .map(ToString::to_string);
            let ctx = get_ctx(deps);

            async move {
                let ctx = match ctx {
                    Ok(ctx) => ctx,
                    Err(err) => return Ok(ToolOutcome::Text(format!("Error: {err}"))),
                };

                let search_dir = match path {
                    Some(p) => match ctx.resolve_path(p) {
                        Ok(path) => path,
                        Err(err) => {
                            return Ok(ToolOutcome::Text(format!("Security error: {err}")));
                        }
                    },
                    None => ctx.working_dir().to_path_buf(),
                };

                let query = search_dir.join(&pattern).display().to_string();
                let entries = match glob(&query) {
                    Ok(entries) => entries,
                    Err(err) => {
                        return Ok(ToolOutcome::Text(format!("Invalid glob pattern: {err}")));
                    }
                };

                let mut files = Vec::new();
                for entry in entries.flatten() {
                    if !entry.is_file() {
                        continue;
                    }

                    let shown = entry
                        .strip_prefix(ctx.root_dir())
                        .unwrap_or(&entry)
                        .display()
                        .to_string();
                    files.push(shown);

                    if files.len() >= 50 {
                        break;
                    }
                }

                if files.is_empty() {
                    Ok(ToolOutcome::Text(format!(
                        "No files match pattern: {pattern}"
                    )))
                } else {
                    Ok(ToolOutcome::Text(format!(
                        "Found {} file(s):\n{}",
                        files.len(),
                        files.join("\n")
                    )))
                }
            }
        })
}

pub fn grep_tool() -> ToolSpec {
    ToolSpec::new("grep", "Search file contents with regex")
        .with_schema(json!({
            "type": "object",
            "properties": {
                "pattern": {"type": "string"},
                "path": {"type": "string"}
            },
            "required": ["pattern"],
            "additionalProperties": false
        }))
        .expect("valid schema")
        .with_handler(|args, deps| {
            let pattern = args
                .get("pattern")
                .and_then(|v| v.as_str())
                .unwrap_or_default()
                .to_string();
            let path = args
                .get("path")
                .and_then(|v| v.as_str())
                .map(ToString::to_string);
            let ctx = get_ctx(deps);

            async move {
                let ctx = match ctx {
                    Ok(ctx) => ctx,
                    Err(err) => return Ok(ToolOutcome::Text(format!("Error: {err}"))),
                };

                let search_dir = match path {
                    Some(p) => match ctx.resolve_path(p) {
                        Ok(path) => path,
                        Err(err) => {
                            return Ok(ToolOutcome::Text(format!("Security error: {err}")));
                        }
                    },
                    None => ctx.working_dir().to_path_buf(),
                };

                let regex = match Regex::new(&pattern) {
                    Ok(regex) => regex,
                    Err(err) => return Ok(ToolOutcome::Text(format!("Invalid regex: {err}"))),
                };

                let mut results = Vec::new();
                for entry in WalkDir::new(&search_dir).into_iter().flatten() {
                    if !entry.file_type().is_file() {
                        continue;
                    }

                    let content = match fs::read_to_string(entry.path()) {
                        Ok(content) => content,
                        Err(_) => continue,
                    };

                    for (index, line) in content.lines().enumerate() {
                        if regex.is_match(line) {
                            let rel = entry
                                .path()
                                .strip_prefix(ctx.root_dir())
                                .unwrap_or(entry.path())
                                .display();
                            let line_preview = if line.len() > 100 {
                                format!("{}...", &line[..100])
                            } else {
                                line.to_string()
                            };
                            results.push(format!("{rel}:{}: {line_preview}", index + 1));
                            if results.len() >= 50 {
                                results.push("... (truncated)".to_string());
                                return Ok(ToolOutcome::Text(results.join("\n")));
                            }
                        }
                    }
                }

                if results.is_empty() {
                    Ok(ToolOutcome::Text(format!("No matches for: {pattern}")))
                } else {
                    Ok(ToolOutcome::Text(results.join("\n")))
                }
            }
        })
}

pub fn todo_read_tool() -> ToolSpec {
    ToolSpec::new("todo_read", "Read current todo list")
        .with_schema(json!({
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": false
        }))
        .expect("valid schema")
        .with_handler(|_args, deps| {
            let ctx = get_ctx(deps);
            async move {
                let ctx = match ctx {
                    Ok(ctx) => ctx,
                    Err(err) => return Ok(ToolOutcome::Text(format!("Error: {err}"))),
                };

                let todos = ctx.read_todos();
                if todos.is_empty() {
                    return Ok(ToolOutcome::Text("Todo list is empty".to_string()));
                }

                let lines = todos
                    .iter()
                    .enumerate()
                    .map(|(idx, item)| {
                        let marker = match item.status.as_str() {
                            "pending" => "[ ]",
                            "in_progress" => "[>]",
                            "completed" => "[x]",
                            _ => "[?]",
                        };
                        format!("{}. {} {}", idx + 1, marker, item.content)
                    })
                    .collect::<Vec<_>>()
                    .join("\n");

                Ok(ToolOutcome::Text(lines))
            }
        })
}

pub fn todo_write_tool() -> ToolSpec {
    ToolSpec::new("todo_write", "Update the todo list")
        .with_schema(json!({
            "type": "object",
            "properties": {
                "todos": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "content": {"type": "string"},
                            "status": {"type": "string"},
                            "active_form": {"type": "string"}
                        },
                        "required": ["content", "status"],
                        "additionalProperties": true
                    }
                }
            },
            "required": ["todos"],
            "additionalProperties": false
        }))
        .expect("valid schema")
        .with_handler(|args, deps| {
            let todos_value = args.get("todos").cloned().unwrap_or(json!([]));
            let ctx = get_ctx(deps);

            async move {
                let ctx = match ctx {
                    Ok(ctx) => ctx,
                    Err(err) => return Ok(ToolOutcome::Text(format!("Error: {err}"))),
                };

                let mut todos: Vec<TodoItem> = match serde_json::from_value(todos_value) {
                    Ok(items) => items,
                    Err(err) => {
                        return Ok(ToolOutcome::Text(format!(
                            "Invalid todos payload: {err}"
                        )));
                    }
                };

                for item in &mut todos {
                    if item.status != "pending"
                        && item.status != "in_progress"
                        && item.status != "completed"
                    {
                        item.status = "pending".to_string();
                    }
                }

                let pending = todos.iter().filter(|t| t.status == "pending").count();
                let in_progress = todos.iter().filter(|t| t.status == "in_progress").count();
                let completed = todos.iter().filter(|t| t.status == "completed").count();

                ctx.write_todos(todos);

                Ok(ToolOutcome::Text(format!(
                    "Updated todos: {pending} pending, {in_progress} in progress, {completed} completed"
                )))
            }
        })
}

pub fn done_tool() -> ToolSpec {
    ToolSpec::new("done", "Signal that the task is complete")
        .with_schema(json!({
            "type": "object",
            "properties": {
                "message": {"type": "string"}
            },
            "required": ["message"],
            "additionalProperties": false
        }))
        .expect("valid schema")
        .with_handler(|args, _deps| {
            let message = args
                .get("message")
                .and_then(|v| v.as_str())
                .unwrap_or("task complete")
                .to_string();
            async move { Ok(ToolOutcome::Done(message)) }
        })
}

fn get_ctx(deps: &DependencyMap) -> Result<Arc<SandboxContext>, ToolError> {
    deps.get::<SandboxContext>()
        .ok_or(ToolError::MissingDependency("SandboxContext"))
}

fn short_session_id() -> String {
    let millis = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis())
        .unwrap_or_default();
    format!("{:x}", millis)
}

#[cfg(test)]
mod tests {
    use std::fs;

    use serde_json::json;

    use super::*;

    fn test_context() -> SandboxContext {
        let root = std::env::temp_dir().join(format!("agent_sdk_rs_tools_{}", short_session_id()));
        SandboxContext::create(Some(root)).expect("sandbox create")
    }

    fn deps_with_ctx(ctx: SandboxContext) -> DependencyMap {
        let deps = DependencyMap::new();
        deps.insert(ctx);
        deps
    }

    #[tokio::test]
    async fn path_resolution_blocks_escape() {
        let ctx = test_context();
        let result = ctx.resolve_path("../../etc/passwd");
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn read_write_edit_roundtrip() {
        let ctx = test_context();
        let deps = deps_with_ctx(ctx.clone());

        let write = write_tool();
        let read = read_tool();
        let edit = edit_tool();

        let wrote = write
            .execute(
                json!({"file_path": "hello.txt", "content": "hello world"}),
                &deps,
            )
            .await
            .expect("write ok");
        assert!(matches!(wrote, ToolOutcome::Text(_)));

        let edited = edit
            .execute(
                json!({
                    "file_path": "hello.txt",
                    "old_string": "world",
                    "new_string": "rust"
                }),
                &deps,
            )
            .await
            .expect("edit ok");
        assert!(matches!(edited, ToolOutcome::Text(ref text) if text.contains("Replaced")));

        let read_out = read
            .execute(json!({"file_path": "hello.txt"}), &deps)
            .await
            .expect("read ok");

        assert!(matches!(read_out, ToolOutcome::Text(ref text) if text.contains("hello rust")));

        let _ = fs::remove_dir_all(ctx.root_dir());
    }

    #[tokio::test]
    async fn todo_and_search_tools_work() {
        let ctx = test_context();
        let deps = deps_with_ctx(ctx.clone());

        let root_file = ctx.root_dir().join("src").join("main.rs");
        fs::create_dir_all(root_file.parent().expect("parent")).expect("mkdirs");
        fs::write(&root_file, "fn main() { println!(\"hello\"); }\n").expect("write sample");

        let todo_write = todo_write_tool();
        let todo_read = todo_read_tool();
        let glob_search = glob_search_tool();
        let grep = grep_tool();
        let done = done_tool();

        let todo_result = todo_write
            .execute(
                json!({
                    "todos": [
                        {"content": "Ship SDK", "status": "in_progress"},
                        {"content": "Add docs", "status": "pending"}
                    ]
                }),
                &deps,
            )
            .await
            .expect("todo write ok");
        assert!(matches!(todo_result, ToolOutcome::Text(ref t) if t.contains("Updated todos")));

        let todo_read_result = todo_read
            .execute(json!({}), &deps)
            .await
            .expect("todo read ok");
        assert!(matches!(todo_read_result, ToolOutcome::Text(ref t) if t.contains("Ship SDK")));

        let glob_result = glob_search
            .execute(json!({"pattern": "**/*.rs"}), &deps)
            .await
            .expect("glob ok");
        assert!(matches!(glob_result, ToolOutcome::Text(ref t) if t.contains("main.rs")));

        let grep_result = grep
            .execute(json!({"pattern": "println"}), &deps)
            .await
            .expect("grep ok");
        assert!(matches!(grep_result, ToolOutcome::Text(ref t) if t.contains("println")));

        let done_result = done
            .execute(json!({"message": "all done"}), &deps)
            .await
            .expect("done ok");
        assert_eq!(done_result, ToolOutcome::Done("all done".to_string()));

        let _ = fs::remove_dir_all(ctx.root_dir());
    }
}
