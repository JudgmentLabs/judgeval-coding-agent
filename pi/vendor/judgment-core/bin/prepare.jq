# Shared by adapter maps. One jq applies the map and this filter.
# Commands pass through so a session toggle still runs when tracing is off.
#
# An event goes map -> capture policy -> sensitive check -> redact -> inbox. The rules are data in
# redact-rules.json, which judgment_core/redact.py also reads; its functions mirror the steps below.
# The audit of what was redacted is counted later, by the sender, from the placeholders.
#
# Redaction is two plain steps. pairs: run the env values, then each rule, over one blob of every
# string and key in the event's text fields, and list each (found text, placeholder) in order. Then
# apply those pairs, as literals, to the fields' JSON. Literals need no match offsets, which jq 1.6
# counts in bytes, and one pass over the JSON reaches every string and key at any depth.
#
# jq compiles this on every hook run, on the agent's critical path, at about 1.5 microseconds per
# instruction: so no recursive walker, and the C builtin _match_impl instead of the larger test,
# match, and join definitions. tests/bench_hook.py measures the hook.
#
# jq's global match counts each match's offset from the start of its string, so many matches in one
# long string cost the square of its length: words are split at literals, and rules run over windows.

import "redact-rules" as $redact_rules;

# Cheap forms of gsub, sub, and test, which the maps also use. Same regex engine, same \s.
def jp_trim: _match_impl("\\S(?:[\\s\\S]*\\S)?"; null; false)[0].string // "";
def jp_ltrim: _match_impl("\\S[\\s\\S]*"; null; false)[0].string // "";
# The text before a suffix matching re at the end, or null.
def jp_before($re): _match_impl("^([\\s\\S]*?)" + $re + "$"; null; false)[0].captures[0].string;
# join for strings, linear: jq's join, and a reduce of +, copy the whole string at every step.
def jp_join($x): .[:1] + (.[1:] | map($x + .)) | add // "";
# Any of the literals is in the input. No literals (null) means any text.
def jp_holds($needs): . as $t | $needs == null or any($needs[]; . as $n | $t | contains($n));
def jp_test($re): _match_impl($re; null; true);
# Each (found text, placeholder) replaced, in order, as literals.
def jp_apply($pairs): reduce $pairs[] as $x (.; split($x[0]) | jp_join($x[1]));
# Split at each of the literals: the pieces, one per line.
def jp_lines($at): reduce $at[] as $c (.; split($c) | jp_join("\n"));
# The windows a rule runs over (limits in redact-rules.json). A whole rule runs over all the text, but
# fails on text that holds more than `most` of the literals it needs.
def jp_windows($p; $l):
  if $p.whole then
    if ([$p.needs[] as $n | split($n) | length - 1] | add) > ($p.most // 1e9) then error("redaction limit") else . end
  elif length <= $l.window then .
  else . as $t | range(0; (length - 1) / $l.step + 1 | floor) * $l.step | $t[.:. + $l.window] end;
# Each distinct match of rule $p, longest first, with the placeholder it becomes. The text the
# rule's group p holds stays. A window is matched after the finds before it are replaced in it, so a
# secret that repeats is matched once, not once per copy.
def jp_found($p; $l):
  reduce jp_windows($p; $l) as $w ([];
    . as $found
    | . + [$w | jp_apply($found) | _match_impl($p.re; "g"; false)[] | select(.length > 0)
           | [.string, (.captures[0] | if .name == "p" then .string else "" end) + "[REDACTED:" + $p.id + "]"]]
    | unique | sort_by(-(.[0] | length))
    | if length > $l.pairs then error("redaction limit") else . end);

def judgment_prepare($cfg; $mode; $root; $model):
  if .kind == "command" then
    .
  else
    # An empty or cut-off opt-in file is no opt-in. It must never fall back to full capture.
    (if ($cfg | type) == "object" then $cfg else {enabled: false} end) as $c
    | (if $mode == "on" or $mode == "off" then $mode elif $c.sessionDefault == "off" then "off" else "on" end) as $active
    | if $c.enabled == false or $active != "on" or (.session_id // "") == "" then
        empty
      else
        # The capture table. judgment_core/optin.py resolve_capture is the same table.
        (if $c.capture == "metadata" or $c.capture == "full" or $c.capture == "sanitized" then $c.capture
         elif $c.redact == false then "full"
         else "sanitized" end) as $capture
        | (if ($c.environment | type) == "string" and ($c.environment | jp_test("\\S")) then $c.environment else "production" end) as $environment
        # The flusher enforces the same policy again after it reads transcripts. The stamp tells it to.
        | if $capture == "full" then . else
            $redact_rules::redact_rules[0] as $r
            # A field's class says what survives losing the text, and what is redacted.
            | def class: $r.fields.classes[$r.fields.of[.] // "text"];
              # Only the fields whose class passes f. tool_calls keep their id and name.
              def strip(f): . as $ev
                | reduce (keys_unsorted[] | select(class | f)) as $k ({}; . + {($k): $ev[$k]})
                | if ($ev.tool_calls | type) == "array" then
                    . + {tool_calls: [$ev.tool_calls[] | objects | {id: (.id // .tool_call_id), name: (.name // .tool_name)}]}
                  else . end;
              # A rule's regex. A list joins its parts; {"def": name} is a shared fragment.
              def re: if type == "array" then map(if type == "object" then $r.defs[.def] else . end) | jp_join("") else . end;
            . as $raw
            | try (
                # Sensitive: tool input that names a sensitive path, or dumps the env, keeps only its structure.
                # Path rules see one word per line, from path and command values. Command rules see the values.
                $r.sensitive as $s
                | (if $capture == "metadata" then null else
                    [(.tool_input, (.tool_calls[]? | objects | .arguments, .input)) | values
                     | if type == "string" then (. as $t | try fromjson catch {command: $t}) else . end
                     | if type == "string" then {command: .} else . end
                     | .. | objects | . as $o | keys_unsorted[] | select($s.keys[.]) | $o[.] | (strings, (arrays | .. | strings))] as $texts
                    | ($texts | jp_join("\n")) as $cmds
                    | (($cmds | jp_lines($s.split)) + "\n"
                      + (if (.prompt | type) == "string" and (.prompt | contains("@")) then
                           if (.prompt | split("@") | length) > $r.limits.mentions + 1 then error("redaction limit") else . end
                           | .prompt | jp_lines($s.mention.split) | split("\n") | map(select(startswith("@")) | .[1:] | jp_lines($s.mention.end) | split("\n")[0] // "" | select(. != "")) | jp_join("\n")
                         else "" end)) as $words
                    | first(
                        (($s.paths + [($c.redactPaths // [])[]? | strings | select(. != "")
                           | {id: "custom_path", re: ("(?<![^\\n/])" + ([explode[] | [.] | implode | $s.glob[.] // .] | jp_join("")) + "(?![^\\n])")}])[]
                         | .re as $re | .needs as $n | select(($words | jp_holds($n)) and ($words | jp_test($re)))
                         | {rule: .id, trigger: ($words | _match_impl("(?<![^\\n])[^\\n]*?(?:" + $re + ")[^\\n]*"; null; false)[0].string)}),
                        ($s.commands[] | .needs as $n | .re as $re | select(($cmds | jp_holds($n)) and ($cmds | jp_test($re))) | {rule: .id, trigger: ("cmd:" + .id)}),
                        null)
                  end) as $hit
                | if $capture == "metadata" then strip(.strip) + {prompt: "", output: ""}
                  elif $hit then strip(.strip) + {sensitive: true, sensitive_rule: $hit.rule, sensitive_trigger: $hit.trigger}
                  else . end
                # Redact every field whose class says so: its strings, and the keys of objects in it.
                | . as $ev
                | (reduce (keys_unsorted[] | select(class.redact)) as $k ({}; . + {($k): $ev[$k]})) as $sub
                | ([$sub | .. | strings, (objects | keys_unsorted[])] | jp_join("\n")) as $text
                # Env values of secret-named variables that occur in the text, longest first: the whole value, and each line of it.
                | (([$ENV | keys_unsorted | jp_join("\n") | _match_impl($r.env.name_re; "g"; false)[].string as $k
                    | $ENV[$k] | (., (select(contains("\n")) | split("\n")[] | rtrimstr("\r")))
                    | . as $v | select(length >= 8 and ($text | contains($v)) and jp_test($r.env.value_re))
                    | {v: ., p: ("[REDACTED:env:" + $k + "]")}]
                   | sort_by(-(.v | length), .p))
                  # Then the rules, built-in first, in table order.
                  + [($r.patterns + [($c.redactPatterns // [])[]? | strings | select(. != "") | {id: "custom", re: .}])[]
                     | .needs as $n | select($text | jp_holds($n)) | . + {re: (.re | re)}]) as $steps
                | ([$r.fields_rule | .needs as $n | select($text | jp_holds($n)) | . + {re: (.json_re | re)}][0]) as $kv
                | if $steps == [] and $kv == null then .
                  else
                    (reduce $steps[] as $p ([$text, []];
                      .[0] as $blob
                      | (if $p.v then [[$p.v, $p.p] | select($blob | contains($p.v))] else $blob | jp_found($p; $r.limits) end) as $m
                      | if (.[1] | length) + ($m | length) > $r.limits.pairs then error("redaction limit") else . end
                      | [($blob | jp_apply($m)), .[1] + $m]))[1] as $pairs
                    | ($sub | tojson) as $json
                    | (if $kv then $json | jp_found($kv; $r.limits) else [] end) as $members
                    | if ($pairs | length) + ($members | length) > $r.limits.pairs then error("redaction limit") else . end
                    | . + ($json | jp_apply($members) | jp_apply([$pairs[] | map(tojson | .[1:-1])]) | fromjson)
                  end)
              # A rule that fails, such as an invalid user regex, drops the event's text, and the labels it could
              # not redact. Never the raw text.
              catch ($raw | strip(.strip and (.redact | not)) + {redaction_error: true})
          end
        | . + {capture: $capture, ts: (if (.ts | type) == "number" then .ts else (now * 1000 | floor) end), environment: $environment, project_root: $root}
        | if (.model // "") == "" and $model != "" then . + {model: $model} else . end
      end
  end;
