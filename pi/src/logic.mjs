import { createReadStream } from "node:fs";
import { createInterface } from "node:readline";

export function sessionIdFrom(ctx) {
  try {
    const fromManager = ctx?.sessionManager?.getSessionId?.();
    if (typeof fromManager === "string" && fromManager && !isGenericSession(fromManager)) return fromManager;
  } catch {
    // A missing session manager is not a session id.
  }
  const file = ctx?.sessionManager?.getSessionFile?.();
  if (typeof file === "string" && file.length > 0) {
    const base = file.split("/").pop() || "";
    const id = base.replace(/\.jsonl?$/, "");
    if (id && !isGenericSession(id)) return id;
  }
  if (typeof ctx?.sessionId === "string" && ctx.sessionId && !isGenericSession(ctx.sessionId)) return ctx.sessionId;
  return "";
}

function isGenericSession(value) {
  return value === "pi-session" || value === "session";
}

export function textOf(message) {
  if (!message) return "";
  if (typeof message.content === "string") return message.content;
  if (Array.isArray(message.content)) {
    return message.content
      .map((part) => {
        if (typeof part === "string") return part;
        if (part?.type === "image" || part?.type === "image_url") return "[image]";
        return part?.text || "";
      })
      .filter(Boolean)
      .join("\n");
  }
  return typeof message.text === "string" ? message.text : "";
}

/** Pi image content parts ({type: "image", data, mimeType}) become "[image]", so the trace never stores the bytes. */
function withoutImages(value) {
  if (Array.isArray(value)) return value.map(withoutImages);
  if (value && typeof value === "object") {
    if (value.type === "image") return "[image]";
    const out = {};
    for (const [key, item] of Object.entries(value)) out[key] = withoutImages(item);
    return out;
  }
  return value;
}

/** A tool result as the model read it: its content parts, not Pi's details or structuredContent. */
function modelSaw(result) {
  if (Array.isArray(result?.content)) return textOf(result);
  return withoutImages(result ?? null);
}

/** A failed request has no content. Its error is what the model call produced. */
function outputOf(message) {
  return textOf(message) || (typeof message?.errorMessage === "string" ? message.errorMessage : "");
}

export function modelName(value) {
  if (typeof value === "string") return value;
  if (value && typeof value === "object") return value.id || value.name || "";
  return "";
}

export function modelOf(message, fallback) {
  return modelName(message?.model) || modelName(message?.modelId) || modelName(fallback) || "";
}

export function providerOf(message) {
  if (typeof message?.provider === "string") return message.provider;
  const model = message?.model;
  if (model && typeof model === "object" && typeof model.provider === "string") return model.provider;
  return "";
}

export function usageOf(usage) {
  if (!usage || typeof usage !== "object") return undefined;
  const cost = usage.cost && typeof usage.cost === "object" ? usage.cost.total : usage.total_cost_usd;
  const out = {};
  const input = usage.input_tokens ?? usage.input;
  const output = usage.output_tokens ?? usage.output;
  const cacheRead = usage.cache_read_input_tokens ?? usage.cacheRead;
  const cacheWrite = usage.cache_creation_input_tokens ?? usage.cacheWrite ?? usage.cache_write_input_tokens;
  if (input != null) out.input_tokens = input;
  if (output != null) out.output_tokens = output;
  if (cacheRead != null) out.cache_read_input_tokens = cacheRead;
  if (cacheWrite != null) out.cache_creation_input_tokens = cacheWrite;
  if (cost != null) out.total_cost_usd = cost;
  return Object.keys(out).length ? out : undefined;
}

function withTimes(record, event) {
  if (event?.start_ts != null) record.start_ts = event.start_ts;
  if (event?.end_ts != null) record.end_ts = event.end_ts;
  record.ts = event?.end_ts ?? Date.now();
  return record;
}

export function toolCallsOf(message) {
  const content = message?.content;
  if (!Array.isArray(content)) return [];
  const calls = [];
  for (const part of content) {
    if (!part || typeof part !== "object") continue;
    const type = part.type;
    if (type !== "toolCall" && type !== "tool_call" && type !== "tool_use") continue;
    calls.push({
      id: String(part.id || part.toolCallId || ""),
      name: String(part.name || part.toolName || "tool"),
      arguments: part.arguments ?? part.input ?? {},
    });
  }
  return calls;
}

function asMessage(row) {
  if (!row || typeof row !== "object") return null;
  if (row.message && typeof row.message === "object" && row.message.role) return row.message;
  if (row.role) return row;
  return null;
}

/** A child session file can be large. Stream it line by line so Pi's event loop keeps running between chunks. */
async function readSessionMessages(file) {
  if (typeof file !== "string" || !file) return [];
  const messages = [];
  try {
    for await (const line of createInterface({ input: createReadStream(file, "utf8"), crlfDelay: Infinity })) {
      if (!line.trim()) continue;
      try {
        const message = asMessage(JSON.parse(line));
        if (message) messages.push(message);
      } catch {
        // skip
      }
    }
  } catch {
    return [];
  }
  return messages;
}

async function childMessages(result) {
  if (Array.isArray(result?.messages) && result.messages.length) return result.messages.map(asMessage).filter(Boolean);
  const details = result?.details;
  if (Array.isArray(details?.messages) && details.messages.length) return details.messages.map(asMessage).filter(Boolean);
  const file = result?.sessionFile || details?.sessionFile;
  return readSessionMessages(file);
}

function stampMessage(message, fallbackStart) {
  const start = Number(message?.timestamp);
  const duration = Number(message?.durationMs);
  const startTs = Number.isFinite(start) ? start : fallbackStart;
  let endTs = startTs;
  if (startTs != null && Number.isFinite(duration) && duration > 0) endTs = startTs + duration;
  return { start_ts: startTs, end_ts: endTs == null || endTs <= startTs ? startTs + 1 : endTs };
}

// Pi stamps a tool result when it arrives, so its timestamp is the tool's end.
function stampToolResult(message, fallbackStart) {
  const end = Number(message?.timestamp);
  const duration = Number(message?.durationMs);
  if (!Number.isFinite(end)) return stampMessage(message, fallbackStart);
  const start = Number.isFinite(duration) && duration > 0 ? end - duration : end - 1;
  return { start_ts: start, end_ts: end };
}

function agentRecords(messages, child, info) {
  const base = {
    v: 1,
    harness: "pi",
    session_id: info.sessionId,
    cwd: info.cwd,
    turn_id: info.turnId,
    environment: "production",
  };
  const { agentId, agentType } = child;
  // Record the prompt the child received. pi-subagents and the official example both prefix it with "Task: ".
  const task = textOf(messages.find((message) => message.role === "user")) || child.task;
  const assistants = messages.filter((message) => message.role === "assistant");
  const answer = assistants.length ? outputOf(assistants[assistants.length - 1]) : "";
  const first = stampMessage(assistants[0] || messages[0], child.start_ts);
  const last = stampMessage(assistants[assistants.length - 1] || messages[messages.length - 1], child.end_ts || first.end_ts);
  const records = [];
  for (const message of assistants) {
    const times = stampMessage(message, first.start_ts);
    const record = {
      ...base,
      event: "model",
      agent_id: agentId,
      model: modelOf(message, child.model || info.model),
      provider: providerOf(message) || info.provider || "",
      output: outputOf(message),
      prompt: task,
      ...times,
      ts: times.end_ts,
    };
    const usage = usageOf(message.usage);
    if (usage) record.usage = usage;
    const calls = toolCallsOf(message);
    if (calls.length) record.tool_calls = calls;
    records.push(record);
  }
  const argsById = new Map(assistants.flatMap((message) => toolCallsOf(message).map((call) => [call.id, call.arguments])));
  for (const message of messages) {
    if (message.role !== "toolResult" && message.role !== "tool") continue;
    const times = stampToolResult(message, first.start_ts);
    const callId = String(message.toolCallId || message.tool_call_id || "");
    records.push({
      ...base,
      event: "tool",
      agent_id: agentId,
      tool_name: String(message.toolName || message.name || "tool"),
      tool_call_id: callId,
      tool_input: withoutImages(argsById.get(callId) ?? null),
      tool_output: textOf(message) || message.content || "",
      is_error: Boolean(message.isError),
      ...times,
      ts: times.end_ts,
    });
  }
  // No subagent_start: the hook binds a start to the session's open turn, and a late child belongs to the turn that launched it.
  records.push({
    ...base,
    event: "subagent_end",
    agent_id: agentId,
    agent_type: agentType,
    prompt: task,
    output: answer,
    start_ts: first.start_ts,
    end_ts: last.end_ts,
    ts: last.end_ts,
  });
  return records;
}

export async function childRecords(event, info) {
  if (event?.toolName !== "subagent") return [];
  const result = event?.result && typeof event.result === "object" ? event.result : {};
  const details = result.details && typeof result.details === "object" ? result.details : result;
  const results = Array.isArray(details.results) ? details.results : Array.isArray(result.results) ? result.results : [];
  const records = [];
  for (const [index, item] of results.entries()) {
    const messages = await childMessages(item);
    if (!messages.length) continue;
    const redacted = item.task === "[prompt redacted]";
    records.push(...agentRecords(
      messages,
      {
        agentId: String(item.agentId || item.runId || item.sessionId || `${event.toolCallId || "sub"}:${index}`),
        agentType: String(item.agent || event?.args?.agent || "subagent"),
        task: String((!redacted && item.task) || event?.args?.task || ""),
        model: item.model,
        start_ts: event?.start_ts,
        end_ts: event?.end_ts,
      },
      info,
    ));
  }
  return records;
}

/** The run id of a background launch, or "" when the subagent ran in the foreground. */
export function backgroundRunId(event) {
  const details = event?.result?.details;
  if (event?.toolName !== "subagent" || !details || (Array.isArray(details.results) && details.results.length)) return "";
  return typeof details.asyncId === "string" ? details.asyncId : "";
}

/** The run a pi-subagents async-complete bus event reports, as {runId, children: [{runId, agent, sessionFile}]}. */
export function completedRun(data) {
  const runId = typeof data?.runId === "string" ? data.runId : typeof data?.id === "string" ? data.id : "";
  if (!runId) return null;
  const results = Array.isArray(data?.results) ? data.results.filter((item) => typeof item?.sessionFile === "string") : [];
  const children = results.map((item, index) => ({
    runId: results.length > 1 ? `${runId}:${index}` : runId,
    agent: typeof item.agent === "string" ? item.agent : "",
    sessionFile: item.sessionFile,
  }));
  return { runId, children };
}

/** Runs a pi-subagents completion notice reports, as {runId, sessionFile, agent}. */
export function noticeRuns(message) {
  if (message?.customType !== "subagent-notify") return [];
  const runs = message?.details?.runs;
  if (!Array.isArray(runs)) return [];
  return runs
    .filter((run) => run && typeof run.runId === "string")
    .map((run) => ({
      runId: run.runId,
      agent: typeof run.agent === "string" ? run.agent : "",
      sessionFile: run.sessionFile || (run.sessionLabel === "Session file" ? run.sessionValue : ""),
    }));
}

/**
 * The notice as the model received it. pi-subagents wraps it in a marker on context_with_system, which may run
 * after this extension's handlers, so read it from the provider payload: the last string that holds the notice.
 */
export function noticeAsSent(payload, notice) {
  let found = "";
  const visit = (value) => {
    if (typeof value === "string") {
      if (value.includes(notice)) found = value;
    } else if (Array.isArray(value)) value.forEach(visit);
    else if (value && typeof value === "object") Object.values(value).forEach(visit);
  };
  visit(payload);
  return found || notice;
}

/** A finished background child, sent to the turn that launched it. */
export async function lateChildRecords(run, launch, info) {
  const messages = await readSessionMessages(run.sessionFile);
  if (!messages.length) return [];
  return agentRecords(
    messages,
    {
      agentId: run.runId,
      agentType: run.agent || launch.agent || "subagent",
      task: launch.task,
      start_ts: launch.start_ts,
    },
    { ...info, turnId: launch.turnId },
  );
}

export function recordsFor(kind, event, info) {
  const base = {
    v: 1,
    harness: "pi",
    session_id: info.sessionId,
    cwd: info.cwd,
    turn_id: info.turnId,
    environment: "production",
  };
  if (kind === "before_agent_start") {
    if (event?.messageType && event.messageType !== "user") return [];
    const prompt = event?.prompt || textOf(event?.message);
    if (!prompt) return [];
    return [withTimes({ ...base, event: "turn_start", prompt }, event)];
  }
  if (kind === "message_end") {
    const message = event?.message || {};
    if (message.role !== "assistant") return [];
    const record = {
      ...base,
      event: "model",
      model: modelOf(message, info.model),
      provider: providerOf(message) || (typeof info.provider === "string" ? info.provider : ""),
      output: outputOf(message),
    };
    // A request answering a background notice, not the turn's prompt.
    if (event?.prompt) record.prompt = event.prompt;
    const usage = usageOf(message.usage);
    if (usage) record.usage = usage;
    const calls = toolCallsOf(message).map((call) => ({ ...call, arguments: withoutImages(call.arguments) }));
    if (calls.length) record.tool_calls = calls;
    return [withTimes(record, event)];
  }
  if (kind === "tool_execution_end") {
    return [
      withTimes(
        {
          ...base,
          event: "tool",
          tool_name: event?.toolName || "tool",
          tool_call_id: event?.toolCallId || "",
          tool_input: withoutImages(event?.args ?? null),
          tool_output: modelSaw(event?.result),
          is_error: Boolean(event?.isError),
          duration_ms: event?.durationMs ?? null,
        },
        event,
      ),
    ];
  }
  if (kind === "agent_end") {
    return [withTimes({ ...base, event: "turn_end", output: info.finalOutput || "" }, event)];
  }
  return [];
}
