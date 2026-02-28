# Introduction

`agent-sdk-rs` is a minimal Rust SDK for tool-using LLM agents.

Design goals:

- explicit agent loop (`query`, `query_stream`)
- provider swap without loop rewrite (`ChatModel` trait)
- JSON-schema tools + dependency injection
- explicit completion support (`ToolOutcome::Done`)
- hard safety bounds (`max_iterations`)

Current provider adapters:

- Anthropic (`AnthropicModel`)
- Google Gemini (`GoogleModel`)
- xAI Grok (`GrokModel`)

Core modules:

- `agent`: run loop, events, builder
- `llm`: provider interface + adapters
- `tools`: tool specs, argument validation, DI map
- `error`: runtime + provider + schema errors

Evidence in repo:

- stop semantics + loop guards: [`src/agent/tests.rs`](https://github.com/Milind220/agent-sdk-rs/blob/main/src/agent/tests.rs)
- tool schema checks: [`src/tools/mod.rs`](https://github.com/Milind220/agent-sdk-rs/blob/main/src/tools/mod.rs)
- provider adapters: [`src/llm/`](https://github.com/Milind220/agent-sdk-rs/tree/main/src/llm)
