# Shared by adapter maps. One jq applies the map and this filter.
# Commands pass through so a session toggle still runs when tracing is off.
#
# An event goes map -> capture policy -> redact -> inbox. The rules are data in redact-rules.json,
# which judgment_core/redact.py also reads; its functions mirror the steps below, step for step.
#
# jq compiles this on every hook run, on the agent's critical path, and each closure costs compile
# time: so `. + {k: v}` instead of `.k = v`, `if c then . else empty end` instead of select(c),
# and the rules live in the data. tests/bench_hook.py measures the hook.

import "redact-rules" as $redact_rules;

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
        # The capture table. judgment_core/optin.py resolve_capture is the same table.
        (if $c.capture == "metadata" or $c.capture == "full" or $c.capture == "sanitized" then $c.capture
         elif $c.redact == false then "full"
         else "sanitized" end) as $capture
        | $redact_rules::redact_rules[0] as $r
        | (reduce $r.keep_fields[] as $k ({}; . + {($k): true})) as $keep
        # Only the structural fields. tool_calls keep their id and name.
        | def strip: . as $ev
            | reduce keys_unsorted[] as $k ({}; if $keep[$k] then . + {($k): $ev[$k]} else . end)
            | if ($ev.tool_calls | type) == "array" then
                . + {tool_calls: [$ev.tool_calls[] | if type == "object" then {id: (.id // .tool_call_id), name: (.name // .tool_name)} else empty end]}
              else . end;
        # Rules whose needs text t holds. A rule without needs always applies.
          def gate(t): if .needs == null or (t as $t | any(.needs[]; . as $n | $t | contains($n))) then . else empty end;
        # One string: env values, then each rule's matches, each replaced as a literal, longest first.
        # Literals need no match offsets, which jq 1.6 counts in bytes.
          def red(envs; rules):
            reduce envs as $e (.; split($e.value) | join("[REDACTED:env:" + $e.name + "]"))
            | reduce rules as $p (.;
                reduce ([match($p.re; "g") | [.string, ([.captures[] | if .name == "p" then .string else empty end][0] // "")]] | sort_by(-(.[0] | length)))[] as $m
                  (.; split($m[0]) | join($m[1] + "[REDACTED:" + $p.id + "]")));
        (if (($c.environment | type) == "string") and (($c.environment | gsub("^\\s+|\\s+$"; "")) != "") then $c.environment else "production" end) as $environment
        # The flusher enforces the same policy again after it reads transcripts. The stamp tells it to.
        | if $capture == "metadata" then
            strip + {prompt: "", output: ""}
          elif $capture == "sanitized" then
            . as $raw
            | try (
                # redactPatterns: regexes. redactPaths: globs, where * is any run of characters and ? one.
                [($c.redactPatterns // [])[]? | if type == "string" and . != "" then {id: "custom", re: .} else empty end] as $user
                | ($r.patterns + $user) as $rules
                | [($c.redactPaths // [])[]? | if type == "string" and . != "" then . else empty end
                   | {id: "custom_path", re: ("(?<![^\\n/])" + (reduce (explode[] | [.] | implode) as $ch (""; . + ({"*": "[^\\n]*", "?": "[^\\n]"}[$ch] // (if ".\\+^$()[]{}|" | contains($ch) then "\\" + $ch else $ch end)))) + "(?![^\\n])")}] as $paths
                # Env values of secret-named variables, longest first. The rule's regex runs only on lines
                # whose name holds one of its needs, found with C builtins on the names' JSON.
                | ($ENV | keys_unsorted | tojson) as $names
                | ([$r.env.needs[] | . as $w | if $names | contains($w) then . else empty end] as $words
                   | [$ENV | keys_unsorted[] | . as $k | if any($words[]; . as $w | $k | contains($w)) then $k + "=" + $ENV[$k] + "\n" else empty end] | join("")
                   | [match($r.env.re; "g") | {name: .captures[0].string, value: .captures[1].string}]
                   | sort_by([-(.value | length), .name])) as $envs
                # Sensitive: tool input that names a sensitive path, or dumps the env, blanks the event.
                # Path rules see one word per line, from path values and commands split at shell punctuation.
                | $r.sensitive as $s
                | [(.tool_input, (.tool_calls[]? | if type == "object" then .arguments, .input else empty end)) | if . == null then empty else . end
                   | if type == "string" then (try fromjson catch {command: .}) else . end
                   | if type == "string" then {command: .} else . end
                   | tojson | match($s.texts_re; "g").captures | (.[0].string // (.[1].string | match($s.string_re; "g").captures[0].string))
                   | "\"" + . + "\"" | fromjson] as $texts
                | ([$texts[] | match($s.token_re; "g").string]
                   + (if (.prompt | type) == "string" and (.prompt | contains("@")) then [.prompt | match($s.mention_re; "g").string[1:]] else [] end)) as $tokens
                | ($tokens | join("\n")) as $blob
                | ($texts | join("\n")) as $cmds
                | [(($s.paths + $paths)[] | gate($blob) | .re as $re | if $blob | test($re) then {rule: .id, trigger: [$tokens[] | if test($re) then . else empty end][0]} else empty end),
                   ($s.commands[] | gate($cmds) | if (.re as $re | $cmds | test($re)) then {rule: .id, trigger: ("cmd:" + .id)} else empty end)][0] as $hit
                | if $hit then
                    strip + {sensitive: true, sensitive_rule: $hit.rule, sensitive_trigger: ($hit.trigger | (($ENV.HOME // "") + "/") as $home
                      | if $home != "/" and startswith($home) then "~/" + .[($home | length):] else . end
                      | red($envs[]; $rules[]))}
                  else . end
                # Every text field. .redactions lists {rule, field, count} of the placeholders each holds.
                | (reduce keys_unsorted[] as $k ([]; if $keep[$k] then . else . + [$k] end)) as $fields
                | ([.[$fields[]] | .. | if type == "string" then . else empty end] | join("\n")) as $text
                | [$envs[] | if .value as $v | $text | contains($v) then . else empty end] as $e
                | [$rules[] | gate($text) | if (.re as $re | $text | test($re)) then . else empty end] as $p
                | if $e == [] and $p == [] then .
                  else
                    def walk_text: if type == "string" then red($e[]; $p[])
                      elif type == "array" then [.[] | walk_text]
                      elif type == "object" then . as $o | reduce keys_unsorted[] as $k ({}; . + {($k): ($o[$k] | walk_text)})
                      else . end;
                    reduce $fields[] as $f (.; . + {($f): (.[$f] | walk_text)})
                    | . as $ev
                    | [$fields[] as $f
                       | reduce ($ev[$f] | .. | if type == "string" then match("\\[REDACTED:([A-Za-z0-9_:.-]+)\\]"; "g").captures[0].string else empty end) as $rule ({}; . + {($rule): (.[$rule] + 1)})
                       | keys[] as $rule | {rule: $rule, field: $f, count: .[$rule]}] as $audit
                    | if $audit == [] then . else . + {redactions: $audit} end
                  end)
              # A rule that fails, such as an invalid user regex, drops the event's text. Never the raw text.
              catch ($raw | strip + {redaction_error: true})
          else . end
        | . + {capture: $capture, ts: (if (.ts | type) == "number" then .ts else (now * 1000 | floor) end), environment: $environment, project_root: $root}
        | if ((.model // "") == "") and $model != "" then . + {model: $model} else . end
      end
  end;
