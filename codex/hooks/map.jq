include "prepare";

def trimmed:
  if type == "string" then gsub("^\\s+|\\s+$"; "") else "" end;

def base:
  {
    v: 1,
    harness: "codex",
    provider: "openai",
    session_id: (.session_id // ""),
    cwd: (.cwd // ""),
    turn_id: (.turn_id // .session_id // ""),
    model: (.model // ""),
    transcript_path: (.transcript_path // "")
  };

def tagged($id):
  if $id != "" then . + {agent_id: $id} else . end;

def command_name($prompt):
  ($prompt | gsub("^\\s+"; "")) as $p
  | if $p | test("^\\$judgment(\\s+(on|off|status))?$") then (($p | split(" "))[1] // "status")
    elif $p | test("^/judgment:(on|off|status)$") then ($p | split(":")[1])
    elif $p | test("^/judgment(\\s+(on|off|status))?$") then (($p | split(" "))[1] // "status")
    else "" end;

def spawn_response:
  if ((.tool_name // "") | test("(^|/)spawn_agent$|collaborationspawn_agent$")) then
    # v2 returns the response as a JSON string with only the task name.
    (.tool_response | (if type == "string" then (try fromjson catch {}) else . end)
      | if type == "object" then . else {} end)
  else {} end;

(.agent_id // "") as $agent
| (.prompt | trimmed) as $prompt
| (command_name($prompt)) as $command
| if $event == "UserPromptSubmit" and $command != "" and $agent == "" then
    {kind: "command", name: $command, cwd: (.cwd // ""), session_id: (.session_id // "")}
  elif $event == "SessionStart" then
    base + {event: "session_model"}
    | select(.model != "")
  elif $event == "UserPromptSubmit" and $agent != "" then
    # The child prompt is the task. It is not a new user turn.
    base + {event: "subagent_prompt", prompt: (.prompt // ""), agent_id: $agent, agent_type: (.agent_type // "")}
  elif $event == "UserPromptSubmit" then
    base + {event: "turn_start", prompt: (.prompt // "")}
    # The Codex app runs its own background requests, such as next-message suggestions, as threads in the project.
    + (if $prompt | test("^This is an app-generated request|^# Overview\\s+Generate \\d+ to \\d+ hyperpersonalized suggestions") then {hidden: true} else {} end)
  elif $event == "PreModelRequest" then
    base + {event: "model_start"} | tagged($agent)
  elif $event == "PostModelRequest" then
    base + {
      event: "model",
      usage: (if .token_usage == null then null else (.token_usage + {input_includes_cache: true}) end),
      output: (if .status == "completed" then "" else (.error // "") end)
    } | tagged($agent)
  elif $event == "PreToolUse" then
    base + {
      event: "tool_start",
      tool_name: (.tool_name // "tool"),
      tool_call_id: (.tool_use_id // ""),
      tool_input: .tool_input
    } | tagged($agent)
  elif $event == "PostToolUse" then
    (base + {
      event: "tool",
      tool_name: (.tool_name // "tool"),
      tool_call_id: (.tool_use_id // ""),
      tool_input: .tool_input,
      tool_output: .tool_response
    } | tagged($agent))
    + (spawn_response
       | if (.agent_id // .agentId // "") != "" then {bind_agent: (.agent_id // .agentId)}
         elif (.task_name // "") != "" then {bind_task: .task_name}
         else {} end)
  elif $event == "SubagentStart" then
    base + {
      event: "subagent_start",
      agent_id: (.agent_id // ""),
      agent_type: (.agent_type // .agent_nickname // "subagent"),
      agent_transcript_path: (.agent_transcript_path // "")
    }
  elif $event == "SubagentStop" then
    base + {
      event: "subagent_end",
      agent_id: (.agent_id // ""),
      agent_type: (.agent_type // .agent_nickname // "subagent"),
      output: (.last_assistant_message // ""),
      agent_transcript_path: (.agent_transcript_path // .transcript_path // "")
    }
  elif $event == "Stop" then
    base + {event: "turn_end", output: (.last_assistant_message // "")}
  elif $event == "SessionEnd" then
    base + {event: "session_end"}
  else
    empty
  end
| judgment_prepare($cfg[0]; $mode; $root; $model)
