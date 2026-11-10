include "prepare";

def trimmed:
  if type == "string" then jp_trim else "" end;

def base:
  {
    v: 1,
    harness: "claude-code",
    provider: "anthropic",
    session_id: (.session_id // ""),
    cwd: (.cwd // ""),
    turn_id: (.prompt_id // .session_id // ""),
    transcript_path: (.transcript_path // "")
  };

def with_agent($id):
  if $id != "" then . + {agent_id: $id} else . end;

def agent_text:
  if (.tool_response.content | type) == "array" then
    [.tool_response.content[]? | .text // empty | if type == "string" then . else tojson end] | jp_join("\n")
  elif (.tool_response.content | type) == "string" then
    .tool_response.content
  else
    ""
  end;

def command_name($prompt):
  ($prompt | jp_ltrim) as $p
  | if $p | jp_test("^/judgment:(on|off|status)$") then ($p | split(":")[1])
    elif $p | jp_test("^/judgment-tracing:(on|off|status)$") then ($p | split(":")[1])
    elif $p | jp_test("^/judgment(\\s+(on|off|status))?$") then (($p | split(" "))[1] // "status")
    else "" end;

# Cursor also runs installed Claude Code hooks. Its own plugin traces that session.
select((has("conversation_id") or has("workspace_roots")) | not)
| (.agent_id // "") as $agent
| (.prompt | trimmed) as $prompt
| (command_name($prompt)) as $command
| if $event == "UserPromptSubmit" and $command != "" then
    {
      kind: "command",
      name: $command,
      cwd: (.cwd // ""),
      session_id: (.session_id // "")
    }
  elif $event == "SessionStart" or $event == "PostModelSwitch" then
    base + {event: "session_model", model: (.model // .to_model // "")}
    | select(.model != "")
  elif $event == "UserPromptSubmit" then
    base + {event: "turn_start", prompt: (.prompt // "")}
  elif $event == "Stop" or $event == "StopFailure" then
    base + {event: "turn_end", output: (.last_assistant_message // "")}
  elif $event == "PostToolUse" and .tool_name == "Agent" then
    (base + {
      event: "subagent_end",
      agent_id: (.tool_response.agentId // .agent_id // .tool_use_id // "agent"),
      agent_type: (.tool_input.subagent_type // "Agent"),
      prompt: (.tool_input.prompt // ""),
      output: agent_text
    }),
    (if (agent_text != "") or ((.tool_response.usage.input_tokens // .tool_response.usage.output_tokens) != null) then
      base + {
        event: "model",
        agent_id: (.tool_response.agentId // .agent_id // .tool_use_id // "agent"),
        model: (.tool_response.resolvedModel // ""),
        prompt: (.tool_input.prompt // ""),
        output: agent_text,
        usage: (.tool_response.usage // null),
        duration_ms: (.tool_response.totalDurationMs // .duration_ms // null)
      }
    else empty end)
  elif $event == "PostToolUse" or $event == "PostToolUseFailure" then
    (base + {
      event: "tool",
      tool_name: (.tool_name // "tool"),
      tool_call_id: (.tool_use_id // ""),
      tool_input: .tool_input,
      tool_output: (if $event == "PostToolUseFailure" then (.error // "") else .tool_response end),
      is_error: ($event == "PostToolUseFailure"),
      duration_ms: (.duration_ms // null)
    } | with_agent($agent))
  elif $event == "SubagentStart" then
    base + {
      event: "subagent_start",
      agent_id: (.agent_id // ""),
      agent_type: (.agent_type // ""),
      prompt: (.prompt // ""),
      agent_transcript_path: (.agent_transcript_path // "")
    }
  elif $event == "SubagentStop" then
    base + {
      event: "subagent_end",
      agent_id: (.agent_id // ""),
      agent_type: (.agent_type // ""),
      output: (.last_assistant_message // ""),
      agent_transcript_path: (.agent_transcript_path // "")
    }
  elif $event == "SessionEnd" then
    base + {event: "session_end"}
  else
    empty
  end
| judgment_prepare($cfg[0]; $mode; $root; $model)
