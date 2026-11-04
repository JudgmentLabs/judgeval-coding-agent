import { spawn, spawnSync } from "node:child_process";
import { existsSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { backgroundRunId, childRecords, completedRun, lateChildRecords, modelName, noticeAsSent, noticeRuns, recordsFor, sessionIdFrom, textOf } from "./logic.mjs";

function bin(name: string): string {
  const here = dirname(fileURLToPath(import.meta.url));
  const vendored = join(here, "..", "vendor", "judgment-core", "bin", name);
  const repo = join(here, "..", "..", "core", "bin", name);
  return existsSync(vendored) ? vendored : repo;
}

const HOOK_TIMEOUT_MS = 5000;
const DRAIN_MS = 3000;

function tracingOff(): boolean {
  return ["0", "false", "no", "off"].includes(String(process.env.JUDGMENT_TRACING || "").toLowerCase());
}

function hook(record: unknown): Promise<void> {
  const payload = JSON.stringify(record);
  return new Promise((resolve) => {
    let child;
    try {
      child = spawn(bin("judgment-hook"), ["--raw"], { stdio: ["pipe", "ignore", "ignore"] });
    } catch {
      resolve();
      return;
    }
    const timer = setTimeout(() => child.kill("SIGKILL"), HOOK_TIMEOUT_MS);
    timer.unref();
    const done = () => {
      clearTimeout(timer);
      resolve();
    };
    child.on("error", done);
    child.on("close", done);
    // The hook exits without reading stdin when tracing is off. EPIPE then must not crash Pi.
    child.stdin.on("error", () => {});
    child.stdin.end(payload);
  });
}

// One hook at a time keeps the spool in event order. Nothing Pi awaits waits on this chain.
let writes: Promise<void> = Promise.resolve();

function drain(): Promise<unknown> {
  return Promise.race([writes, new Promise((resolve) => setTimeout(resolve, DRAIN_MS).unref())]);
}

function send(records: unknown[] | (() => Promise<unknown[]>)): void {
  writes = writes
    .then(async () => {
      for (const record of typeof records === "function" ? await records() : records) await hook(record);
    })
    .catch(() => {});
}

type PiApi = {
  events?: { on: (name: string, handler: (data: unknown) => void) => void };
  registerCommand: (name: string, options: { description: string; handler: (args: string, ctx: SessionCtx) => Promise<void> }) => void;
  on: (name: string, handler: (event: Record<string, unknown>, ctx: SessionCtx) => Promise<void>) => void;
};

type SessionCtx = {
  cwd?: string;
  model?: string | { id?: string; name?: string; provider?: string };
  ui: { notify: (message: string, level: string) => void };
  sessionManager?: { getSessionFile?: () => string };
  sessionId?: string;
};

export default function judgmentTracing(pi: PiApi): void {
  if (process.env.JUDGMENT_PARENT) return;
  let turnId = "";
  let lastTurnId = "";
  let finalOutput = "";
  let notice = "";
  // before_agent_start already recorded the prompt Pi is about to deliver as a user message.
  let promptPending = false;
  const subagentCalls = new Set<string>();
  const argsByCall = new Map<string, unknown>();
  const toolStarts = new Map<string, number>();
  const requestStarts: number[] = [];

  const info = (ctx: SessionCtx) => ({
    sessionId: sessionIdFrom(ctx),
    cwd: ctx?.cwd || process.cwd(),
    turnId,
    finalOutput,
    model: modelName(ctx?.model),
    provider: typeof ctx?.model === "object" && ctx.model ? String((ctx.model as { provider?: string }).provider || "") : "",
  });

  const background = new Map<string, { turnId: string; agent: string; task: string; start_ts?: number; info: ReturnType<typeof info> }>();

  pi.registerCommand("judgment", {
    description: "Turn Judgment tracing on or off for this session",
    handler: async (args, ctx) => {
      const action = String(args || "status").trim().split(/\s+/)[0] || "status";
      if (action !== "on" && action !== "off" && action !== "status") {
        ctx.ui.notify("Use /judgment on, /judgment off, or /judgment status.", "info");
        return;
      }
      // Records already queued belong to turns traced before the switch.
      await drain();
      const result = spawnSync(
        bin("judgment-session"),
        [action, "--cwd", ctx?.cwd || process.cwd(), "--session", sessionIdFrom(ctx)],
        { encoding: "utf8" },
      );
      ctx.ui.notify((result.stdout || "").trim(), "info");
    },
  });

  if (tracingOff()) return;

  const startTurn = (event: Record<string, unknown>, ctx: SessionCtx) => {
    turnId = `${sessionIdFrom(ctx)}:${Date.now()}`;
    finalOutput = "";
    send(recordsFor("before_agent_start", event, info(ctx)));
  };

  // Pi exits when shutdown handlers return, and a record still queued would be lost.
  pi.on("session_shutdown", async () => {
    await drain();
  });

  pi.on("before_agent_start", async (event, ctx) => {
    const messageType = String(event?.messageType || "");
    const custom = String(event?.customType || event?.message?.customType || "");
    // A background completion is a custom notice. Keep it on the launch.
    if (lastTurnId && (custom === "subagent-notify" || messageType === "custom")) {
      const launch = noticeRuns(event?.message).map((run) => background.get(run.runId)).find(Boolean);
      turnId = launch?.turnId || lastTurnId;
      return;
    }
    const text = String(event?.prompt || "");
    const images = Array.isArray(event?.images) ? event.images.map(() => "[image]") : [];
    promptPending = !messageType || messageType === "user";
    startTurn({ ...event, prompt: [text.trim() ? text : "", ...images].filter(Boolean).join("\n") }, ctx);
  });

  pi.on("before_provider_request", async (event) => {
    requestStarts.push(Date.now());
    if (notice) notice = noticeAsSent(event?.payload, notice);
  });

  pi.on("message_start", async (event) => {
    const message = event?.message as { role?: string } | undefined;
    if (message?.role && message.role !== "assistant") return;
    // message_start is the first token. Use it only when the provider hook did not fire.
    if (requestStarts.length === 0) requestStarts.push(Date.now());
  });

  pi.on("message_end", async (event, ctx) => {
    const message = event?.message as { role?: string } | undefined;
    if (message?.role === "custom") {
      for (const run of noticeRuns(message)) background.delete(run.runId);
      if ((message as { customType?: string }).customType === "subagent-notify") notice = textOf(message);
      return;
    }
    if (message?.role === "user") {
      if (promptPending) {
        promptPending = false;
        return;
      }
      // A follow-up or steer typed while Pi ran. Pi delivers it without before_agent_start, and it is a new request.
      if (turnId) send(recordsFor("agent_end", {}, info(ctx)));
      lastTurnId = turnId || lastTurnId;
      startTurn({ messageType: "user", prompt: textOf(message) }, ctx);
      return;
    }
    if (message?.role === "assistant") finalOutput = textOf(message);
    if (!turnId) turnId = lastTurnId || `${sessionIdFrom(ctx)}:0`;
    const started = requestStarts.length ? requestStarts.shift() : undefined;
    const end = Date.now();
    const duration = Number(event?.durationMs);
    let startTs = started;
    let endTs = end;
    if (startTs != null && Number.isFinite(duration) && duration > 0) endTs = startTs + duration;
    else if (startTs == null && Number.isFinite(duration) && duration > 0) startTs = end - duration;
    const prompt = message?.role === "assistant" ? notice : "";
    if (prompt) notice = "";
    send(recordsFor("message_end", { ...event, prompt, start_ts: startTs, end_ts: endTs }, info(ctx)));
  });

  pi.on("tool_execution_start", async (event, ctx) => {
    const id = String(event.toolCallId || "");
    if (!id) return;
    argsByCall.set(id, event.args);
    toolStarts.set(id, Date.now());
    if (event.toolName === "subagent") {
      const sid = sessionIdFrom(ctx);
      if (sid) process.env.JUDGMENT_PARENT = `${sid}:${turnId}:${id}`;
      subagentCalls.add(id);
    }
  });

  pi.on("tool_call", async (event) => {
    const id = String(event.toolCallId || "");
    if (id && event.input != null) argsByCall.set(id, event.input);
  });

  pi.on("tool_execution_end", async (event, ctx) => {
    if (!turnId) turnId = lastTurnId || `${sessionIdFrom(ctx)}:0`;
    const id = String(event.toolCallId || "");
    const args = event.args ?? (id ? argsByCall.get(id) : undefined);
    const started = id ? toolStarts.get(id) : undefined;
    if (id) {
      argsByCall.delete(id);
      toolStarts.delete(id);
    }
    const end = Date.now();
    const duration = Number(event?.durationMs);
    let startTs = started;
    let endTs = end;
    if (startTs != null && Number.isFinite(duration) && duration > 0) endTs = startTs + duration;
    else if (startTs == null && Number.isFinite(duration) && duration > 0) startTs = end - duration;
    const current = info(ctx);
    const timed = { ...event, args, start_ts: startTs, end_ts: endTs };
    send(recordsFor("tool_execution_end", timed, current));
    if (event.toolName === "subagent") {
      send(() => childRecords(timed, current));
      const runId = backgroundRunId(event);
      if (runId) {
        const params = (args && typeof args === "object" ? args : {}) as { agent?: string; task?: string };
        background.set(runId, { turnId, agent: String(params.agent || ""), task: String(params.task || ""), start_ts: endTs, info: current });
      }
      // Parallel calls share the process env. Keep the marker until the last running call ends.
      subagentCalls.delete(id);
      if (subagentCalls.size === 0) delete process.env.JUDGMENT_PARENT;
    }
  });

  // The completion notice can wait for the next turn's tool to finish. The bus event fires when the child does,
  // so the launch turn is resent then. The notice still routes the reply to the launch turn.
  pi.events?.on("subagent:async-complete", (data) => {
    const run = completedRun(data);
    const launch = run && background.get(run.runId);
    if (!launch) return;
    for (const child of run.children) send(() => lateChildRecords(child, launch, launch.info));
  });

  // agent_end is not final: Pi may retry a failed request or run queued messages after it.
  pi.on("agent_settled", async (_event, ctx) => {
    if (!turnId) return;
    send(recordsFor("agent_end", _event, info(ctx)));
    promptPending = false;
    lastTurnId = turnId;
    turnId = "";
  });
}
