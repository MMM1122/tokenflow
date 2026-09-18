# Local gateway outcomes and metrics

The gateway is an M2 local preview. Start it on loopback using the README command. Configuration comes from the process environment at startup; no environment file is loaded automatically.

`/v1/optimize` does not call a provider. `/v1/generate` and `/v1/metrics` require `Authorization: Bearer <TOKENFLOW_API_KEY>`. Generation requires a configured provider. `/health` reports whether a provider object is configured, not whether remote credentials or model access have been verified.

## Handled route outcomes

- `200`: `success`. Optimization returns locally estimated input counts; generation also returns validated provider usage.
- `422`: `budget_exceeded`. Protected context cannot fit under the configured policy. No provider request was attempted.
- `500`: `optimizer_error`. Optimization failed before a provider attempt. The response excludes the internal exception message.
- `503`: `provider_unavailable`. The generation route has no configured provider. No provider request was attempted.
- `502`: `provider_error`. The provider invocation raised an exception or returned an invalid result. Provider receipt and billing may be unknown; the gateway does not retry automatically.

Each completed route operation receives a server-generated UUID. Handled route responses include it in `X-Request-ID`; successful responses also include `request_id` in the JSON body. Match that value against the authenticated metrics endpoint when investigating a failure. The identifier is for correlation, not idempotency: submitting a request again can incur another charge.

Authentication failures (`401`, or `503` when the gateway key is missing), request schema failures (`422`), health requests, and metrics reads do not enter the route handlers and are not included in these counters or request IDs. An in-flight request or a process termination before recording can also be absent. This is route instrumentation, not a complete HTTP access log or billing ledger.

## Counter and event semantics

`GET /v1/metrics` returns:

- `operations_recorded`: completed operations recorded since this process started.
- `outcome_counts`: counts such as `optimize.success`, `generate.budget_exceeded`, and `generate.provider_error`.
- `provider_attempts`: invocations of the provider interface, including failures. It does not prove that the provider received or billed a request.
- `event_capacity`, `events_evicted`, and `events`: the most recent 1,000 events, with the number removed from the bounded history.

Counters survive history eviction. A snapshot reads the counters and events under the same lock, and callers receive copies. Everything is in memory, resets on restart, and is independent per worker. Use one worker when inspecting a coherent local history.

Each event contains its operation, outcome, ID, elapsed route time, provider-attempt flag, and provider usage fields. When optimization succeeded, it also contains the local optimization metrics even if the subsequent provider call failed. These local token estimates are not billing usage or proof of cost savings.

`billing_usage` has three values:

- `not_called`: no provider invocation; provider token fields are null.
- `reported`: a successful validated response supplied input, cached-input, and output usage.
- `unknown`: the provider invocation failed or its result was invalid; provider usage and duration remain null. This does not mean zero tokens or zero cost.

`total_latency_ms` measures execution inside the route through result handling. It excludes request parsing, authentication, thread-pool waiting, response serialization, and network transfer. It is not client-observed end-to-end latency. `provider_latency_ms` is available only for successful validated responses.

## Data and deployment boundaries

Metrics contain no request text, document content, generated answers, API keys, or raw exception messages. API optimization/generation responses still intentionally contain application context and answers. These metrics do not collect monetary prices or establish billed savings.

Provider results reject negative usage, cached input above total input, nonfinite latency, and empty text/model identifiers. Invalid results are failures even when a custom provider constructed a model while bypassing validation.

Production deployment, rate limiting, tenant isolation, admission control, persistent metrics, streaming body limits, and cancellation guarantees remain outside this local preview. Continue with real quality and cost evaluation before expanding infrastructure.
