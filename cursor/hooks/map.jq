include "prepare";

def trimmed:
  if type == "string" then gsub("^\\s+|\\s+$"; "") else "" end;

def strip_copy($gid):
  if $gid | test("-[0-9]+-[A-Za-z0-9]{4}$") then
    strip_copy($gid | sub("-[0-9]+-[A-Za-z0-9]{4}$"; ""))
  else $gid end;

def turn_key:
  # A thought is delivered twice. The copy's generation id ends in the routed
  # model, or in -<n>-<4 chars>. Both copies belong to one turn.
  (.generation_id // "") as $gid
  | (.model // "") as $model
  | (if $gid == "" then "" else strip_copy($gid) end) as $base
  | (if $base == "" then ""
     elif $model != "" and ($base | endswith($model)) then
       ($base[0:($base | length) - ($model | length)] | gsub("[\\-_:]+$"; ""))
     elif ($base | test(":(auto-smart|grok-[^:]+)$")) then
       ($base | sub(":(auto-smart|grok-[^:]+)$"; ""))
     else $base end) as $key
  | if $key == "" then (.conversation_id // .session_id // "") else $key end;

def usage_from_stop:
  if .usage != null then
    (.usage + {input_includes_cache: true})
  elif .input_tokens != null or .output_tokens != null or .cache_read_tokens != null or .cache_read_input_tokens != null then
    {
      input_tokens: (.input_tokens // 0),
      output_tokens: (.output_tokens // 0),
      cache_read_input_tokens: (.cache_read_input_tokens // .cache_read_tokens // 0),
      cache_creation_input_tokens: (.cache_creation_input_tokens // .cache_write_tokens // 0),
      input_includes_cache: true
    }
  else null end;

def parsed_output:
  if (.tool_output | type) == "string" then
    (try (.tool_output | fromjson) catch {})
  elif .tool_output == null then {}
  else .tool_output end;

def base:
  {
    v: 1,
    harness: "cursor",
    session_id: (.parent_conversation_id // .conversation_id // .session_id // ""),
    cwd: (.cwd // .workspace_roots[0] // ""),
    turn_id: turn_key,
    model: (.model // ""),
    transcript_path: (.transcript_path // "")
  };

def sub_id:
  # subagentStop does not include subagent_id. Both events include type and task.
  # The task is text the user may not want stored, so the id carries a checksum of it instead.
  (.subagent_type // "subagent") + ":"
  + ((.task // .description // "") | explode | reduce .[] as $c (7; (. * 31 + $c) % 4294967291) | tostring);

(.prompt | trimmed) as $prompt
| if $event == "beforeSubmitPrompt" and ($prompt | test("^/judgment(\\s+(on|off|status))?$")) then
    {
      kind: "command",
      name: (($prompt | split(" "))[1] // "status"),
      cwd: (.cwd // .workspace_roots[0] // ""),
      session_id: (.conversation_id // .session_id // "")
    }
  elif $event == "sessionStart" then
    base + {event: "session_model"}
    | select(.model != "")
  elif $event == "beforeSubmitPrompt" then
    base + {event: "turn_start", prompt: (.prompt // "")}
  elif $event == "afterAgentResponse" then
    base + {event: "model", output: (.text // "")}
  elif $event == "afterAgentThought" then
    base + {event: "thought", output: (.text // ""), duration_ms: (.duration // .duration_ms // null)}
  elif $event == "preToolUse" then
    base + {
      event: "tool_start",
      tool_name: (.tool_name // "tool"),
      tool_call_id: (.tool_use_id // ""),
      tool_input: .tool_input,
      task_pending: (.tool_name == "Task"),
      prompt: (if .tool_name == "Task" then (.tool_input.prompt // "") else "" end),
      agent_type: (.tool_input.subagent_type // "")
    }
  elif $event == "postToolUse" and .tool_name == "Task" then
    (parsed_output) as $out
    | (base + {
        event: "tool",
        tool_name: "Task",
        tool_call_id: (.tool_use_id // ""),
        tool_input: .tool_input,
        tool_output: .tool_output,
        duration_ms: (.duration // $out.durationMs // null),
        bind_agent: ($out.agentId // ""),
        agent_type: (.tool_input.subagent_type // ""),
        background: ($out.isBackground == true)
      }),
      (base + {
        event: "subagent_end",
        agent_id: ($out.agentId // ""),
        agent_type: (.tool_input.subagent_type // "subagent"),
        prompt: (.tool_input.prompt // ""),
        output: "",
        duration_ms: ($out.durationMs // .duration // null),
        background: ($out.isBackground == true)
      })
  elif $event == "postToolUse" or $event == "postToolUseFailure" then
    # Shell reports {output, exitCode}. Keep the output text. Nothing shows the model the exit code.
    (parsed_output) as $out
    | base + {
      event: "tool",
      tool_name: (.tool_name // "tool"),
      tool_call_id: (.tool_use_id // ""),
      tool_input: .tool_input,
      tool_output: (
        if $event == "postToolUseFailure" then (.error_message // "")
        elif ($out | type) == "object" and ($out.output | type) == "string" then $out.output
        else (.tool_output // "") end
      ),
      is_error: ($event == "postToolUseFailure"),
      duration_ms: (.duration // .duration_ms // null)
    }
  elif $event == "subagentStart" then
    base + {
      event: "subagent_start",
      agent_id: sub_id,
      agent_type: (.subagent_type // ""),
      model: (.subagent_model // .model // ""),
      prompt: (.task // "")
    }
  elif $event == "subagentStop" then
    base + {
      event: "subagent_end",
      agent_id: sub_id,
      agent_type: (.subagent_type // ""),
      output: (.summary // ""),
      duration_ms: (.duration_ms // null)
    }
  elif $event == "stop" then
    base + {event: "turn_end", output: (.text // .last_assistant_message // ""), usage: usage_from_stop}
  elif $event == "sessionEnd" then
    # Headless cursor-agent -p never fires beforeSubmitPrompt, afterAgentResponse, or stop.
    # sessionEnd marks the turns already spooled. It does not open a new empty turn.
    base + {event: "session_end"}
  else
    empty
  end
| judgment_prepare($cfg[0]; $mode; $root; $model)
