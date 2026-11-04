# Shared by adapter maps. One jq applies the map and this filter.
# Commands pass through so a session toggle still runs when tracing is off.

def judgment_prepare($cfg; $mode; $root; $model):
  if .kind == "command" then
    .
  else
    # An empty or cut-off opt-in file is no opt-in. It must never fall back to full capture.
    (if ($cfg | type) == "object" then $cfg else {enabled: false} end) as $c
    | (if $c.sessionDefault == "off" then "off" else "on" end) as $default
    | (if $mode == "on" or $mode == "off" then $mode else $default end) as $active
    | if $c.enabled == false or $active != "on" or (.session_id // "") == "" then
        empty
      else
        (if $c.capture == "metadata" then "metadata" else "full" end) as $capture
        | (if (($c.environment | type) == "string") and (($c.environment | gsub("^\\s+|\\s+$"; "")) != "") then $c.environment else "production" end) as $environment
        # The flusher enforces the same policy again after it reads transcripts. The stamp tells it to.
        | if $capture == "metadata" then
            .prompt = "" | .output = "" | .capture = "metadata"
            | del(.tool_input, .tool_output, .tool_results, .received)
            | if (.tool_calls | type) == "array" then
                .tool_calls |= map(select(type == "object") | {id: (.id // .tool_call_id), name: (.name // .tool_name)})
              else . end
          else . end
        | .ts = (now * 1000 | floor)
        | .environment = $environment
        | .project_root = $root
        | if ((.model // "") == "") and $model != "" then .model = $model else . end
      end
  end;
