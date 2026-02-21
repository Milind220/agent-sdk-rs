pub mod claude_code;

use std::any::{Any, TypeId};
use std::collections::HashMap;
use std::future::Future;
use std::sync::{Arc, RwLock};

use futures_util::future::BoxFuture;
use serde_json::Value;

use crate::error::{SchemaError, ToolError};

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ToolOutcome {
    Text(String),
    Done(String),
}

type DynDependency = Arc<dyn Any + Send + Sync>;
type ToolHandler = dyn Fn(Value, &DependencyMap) -> BoxFuture<'static, Result<ToolOutcome, ToolError>>
    + Send
    + Sync;

#[derive(Clone, Default, Debug)]
pub struct DependencyMap {
    typed: Arc<RwLock<HashMap<TypeId, DynDependency>>>,
    named: Arc<RwLock<HashMap<String, DynDependency>>>,
}

impl DependencyMap {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn insert<T>(&self, value: T)
    where
        T: Send + Sync + 'static,
    {
        let mut typed = self
            .typed
            .write()
            .expect("dependency typed map lock poisoned");
        typed.insert(TypeId::of::<T>(), Arc::new(value));
    }

    pub fn get<T>(&self) -> Option<Arc<T>>
    where
        T: Send + Sync + 'static,
    {
        let typed = self.typed.read().ok()?;
        let value = typed.get(&TypeId::of::<T>())?.clone();
        Arc::downcast::<T>(value).ok()
    }

    pub fn insert_named<T>(&self, key: impl Into<String>, value: T)
    where
        T: Send + Sync + 'static,
    {
        let mut named = self
            .named
            .write()
            .expect("dependency named map lock poisoned");
        named.insert(key.into(), Arc::new(value));
    }

    pub fn get_named<T>(&self, key: &str) -> Option<Arc<T>>
    where
        T: Send + Sync + 'static,
    {
        let named = self.named.read().ok()?;
        let value = named.get(key)?.clone();
        Arc::downcast::<T>(value).ok()
    }

    pub fn merged_with(&self, overrides: &DependencyMap) -> DependencyMap {
        let merged = DependencyMap::new();

        {
            let mut dst_typed = merged
                .typed
                .write()
                .expect("dependency typed map lock poisoned");
            if let Ok(src_typed) = self.typed.read() {
                for (key, value) in &*src_typed {
                    dst_typed.insert(*key, value.clone());
                }
            }
            if let Ok(src_typed_override) = overrides.typed.read() {
                for (key, value) in &*src_typed_override {
                    dst_typed.insert(*key, value.clone());
                }
            }
        }

        {
            let mut dst_named = merged
                .named
                .write()
                .expect("dependency named map lock poisoned");
            if let Ok(src_named) = self.named.read() {
                for (key, value) in &*src_named {
                    dst_named.insert(key.clone(), value.clone());
                }
            }
            if let Ok(src_named_override) = overrides.named.read() {
                for (key, value) in &*src_named_override {
                    dst_named.insert(key.clone(), value.clone());
                }
            }
        }

        merged
    }
}

#[derive(Clone)]
pub struct ToolSpec {
    name: String,
    description: String,
    json_schema: Value,
    handler: Arc<ToolHandler>,
}

impl std::fmt::Debug for ToolSpec {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ToolSpec")
            .field("name", &self.name)
            .field("description", &self.description)
            .field("json_schema", &self.json_schema)
            .finish()
    }
}

impl ToolSpec {
    pub fn new(name: impl Into<String>, description: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            json_schema: serde_json::json!({
                "type": "object",
                "properties": {},
                "required": [],
                "additionalProperties": true,
            }),
            handler: Arc::new(|_args, _deps| {
                Box::pin(async {
                    Err(ToolError::Execution(
                        "tool handler not configured".to_string(),
                    ))
                })
            }),
        }
    }

    pub fn with_schema(mut self, schema: Value) -> Result<Self, SchemaError> {
        validate_schema(&schema)?;
        self.json_schema = schema;
        Ok(self)
    }

    pub fn with_handler<F, Fut>(mut self, handler: F) -> Self
    where
        F: Fn(Value, &DependencyMap) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Result<ToolOutcome, ToolError>> + Send + 'static,
    {
        self.handler = Arc::new(move |args, deps| Box::pin(handler(args, deps)));
        self
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn description(&self) -> &str {
        &self.description
    }

    pub fn json_schema(&self) -> &Value {
        &self.json_schema
    }

    pub async fn execute(
        &self,
        args: Value,
        dependencies: &DependencyMap,
    ) -> Result<ToolOutcome, ToolError> {
        validate_arguments(self.name(), &self.json_schema, &args)?;
        (self.handler)(args, dependencies).await
    }
}

fn validate_schema(schema: &Value) -> Result<(), SchemaError> {
    let schema_obj = schema.as_object().ok_or(SchemaError::SchemaNotObject)?;

    let root_type = schema_obj
        .get("type")
        .and_then(Value::as_str)
        .ok_or(SchemaError::RootTypeMustBeObject)?;

    if root_type != "object" {
        return Err(SchemaError::RootTypeMustBeObject);
    }

    if let Some(required) = schema_obj.get("required") {
        let required_arr = required.as_array().ok_or(SchemaError::InvalidRequired)?;
        for item in required_arr {
            if !item.is_string() {
                return Err(SchemaError::InvalidRequired);
            }
        }
    }

    Ok(())
}

fn validate_arguments(tool_name: &str, schema: &Value, args: &Value) -> Result<(), ToolError> {
    let args_obj = args
        .as_object()
        .ok_or_else(|| ToolError::InvalidArguments {
            tool: tool_name.to_string(),
            message: "arguments must be a JSON object".to_string(),
        })?;

    let schema_obj = schema
        .as_object()
        .ok_or_else(|| ToolError::InvalidArguments {
            tool: tool_name.to_string(),
            message: "tool schema must be a JSON object".to_string(),
        })?;

    if let Some(required) = schema_obj.get("required").and_then(Value::as_array) {
        for field in required {
            let Some(field_name) = field.as_str() else {
                continue;
            };
            if !args_obj.contains_key(field_name) {
                return Err(ToolError::InvalidArguments {
                    tool: tool_name.to_string(),
                    message: format!("missing required field: {field_name}"),
                });
            }
        }
    }

    let properties = schema_obj
        .get("properties")
        .and_then(Value::as_object)
        .cloned()
        .unwrap_or_default();

    if schema_obj
        .get("additionalProperties")
        .and_then(Value::as_bool)
        == Some(false)
    {
        for key in args_obj.keys() {
            if !properties.contains_key(key) {
                return Err(ToolError::InvalidArguments {
                    tool: tool_name.to_string(),
                    message: format!("unknown field: {key}"),
                });
            }
        }
    }

    for (key, value) in args_obj {
        if let Some(field_schema) = properties.get(key) {
            if let Some(type_name) = field_schema.get("type").and_then(Value::as_str) {
                if !value_matches_type(value, type_name) {
                    return Err(ToolError::InvalidArguments {
                        tool: tool_name.to_string(),
                        message: format!("field '{key}' must be of type {type_name}"),
                    });
                }
            }
        }
    }

    Ok(())
}

fn value_matches_type(value: &Value, type_name: &str) -> bool {
    match type_name {
        "string" => value.is_string(),
        "integer" => value.as_i64().is_some() || value.as_u64().is_some(),
        "number" => value.as_f64().is_some(),
        "boolean" => value.is_boolean(),
        "object" => value.is_object(),
        "array" => value.is_array(),
        "null" => value.is_null(),
        _ => true,
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    #[test]
    fn schema_validation_rejects_non_object_root() {
        let result = ToolSpec::new("bad", "bad").with_schema(json!({"type": "string"}));
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn dependency_overrides_win() {
        let base = DependencyMap::new();
        base.insert::<u32>(1);

        let overrides = DependencyMap::new();
        overrides.insert::<u32>(9);

        let merged = base.merged_with(&overrides);
        assert_eq!(merged.get::<u32>().as_deref(), Some(&9));

        let tool = ToolSpec::new("read", "read dep")
            .with_schema(json!({
                "type": "object",
                "properties": {},
                "required": [],
                "additionalProperties": false
            }))
            .expect("schema should be valid")
            .with_handler(|_args, deps| {
                let value = deps
                    .get::<u32>()
                    .ok_or(ToolError::MissingDependency("u32"))
                    .map(|v| *v)
                    .unwrap_or(0);
                async move { Ok(ToolOutcome::Text(value.to_string())) }
            });

        let outcome = tool
            .execute(json!({}), &merged)
            .await
            .expect("tool executes");
        assert_eq!(outcome, ToolOutcome::Text("9".to_string()));
    }

    #[tokio::test]
    async fn argument_validation_reports_missing_required() {
        let tool = ToolSpec::new("req", "required")
            .with_schema(json!({
                "type": "object",
                "properties": {"value": {"type": "string"}},
                "required": ["value"],
                "additionalProperties": false
            }))
            .expect("schema valid")
            .with_handler(|_args, _deps| async move { Ok(ToolOutcome::Text("ok".into())) });

        let err = tool
            .execute(json!({}), &DependencyMap::new())
            .await
            .expect_err("should fail");

        let message = err.to_string();
        assert!(message.contains("missing required field"));
    }
}
