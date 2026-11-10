# Shared by adapter maps. One jq applies the map and this filter.
# Commands pass through so a session toggle still runs when tracing is off.
#
# An event goes map -> capture policy -> sensitive check -> redact -> inbox. The rules are data: redact-rules.json is
# the source, and scripts/gen_redact.py composes it into redact-compiled.json, which this loads. judgment_core/redact.py
# reads the same source, and its functions mirror the steps below (tests/test_redact.py checks they agree).
#
# Redaction: one blob of every string and key in the event's text fields gives the pairs (found text, replacement, rule):
# env values, then each rule, every one matched against the unchanged blob, one pair per found text, longest first.
# Members under secret keys are found in the fields' JSON. Each field then gets its members, then the pairs, replaced as
# literals in its JSON; every replacement is counted into records {rule, field, count}, carried on the event as
# `redactions`. Literals need no offsets in the replacing, and one pass over the JSON reaches every string and key at any
# depth. When two keys of one object would become the same placeholder (the field has fewer members after), the field
# is redone with every replacement numbered (#1, #2, ...), so no member is lost.
#
# jq compiles this on every hook run, on the agent's critical path, at about 1.5 microseconds per instruction: so no
# recursive walker, and only C builtins or tiny definitions (no any, first-of-many, test, gsub, to_entries, paths,
# limit). tests/bench_hook.py measures the hook.

import "redact-compiled" as $redact_rules;

def jp_trim: _match_impl("\\S(?:[\\s\\S]*\\S)?"; null; false)[0].string // "";
def jp_ltrim: _match_impl("\\S[\\s\\S]*"; null; false)[0].string // "";
def jp_before($re): _match_impl("^([\\s\\S]*?)" + $re + "$"; null; false)[0].captures[0].string;
def jp_join($x): .[:1] + (.[1:] | map($x + .)) | add // "";
def jp_test($re): _match_impl($re; null; true);
# Any of the literals is in the input. No literals (null) means any text.
def jp_holds($needs): . as $t | $needs == null or ([$needs[] as $n | $t | contains($n) | if . then 1 else empty end] != []);
# Split at each of the literals: the pieces, one per line.
def jp_lines($at): reduce $at[] as $c (.; split($c) | jp_join("\n"));
# Only the fields in $set.
def jp_pick($set): . as $e | reduce (keys_unsorted[] | if $set[.] then . else empty end) as $k ({}; . + {($k): $e[$k]});
# tool_calls as their ids, and their names when $names: a name is free-form text.
def jp_calls($names):
  if (.tool_calls | type) == "array" then {tool_calls: [.tool_calls[] | objects | {id: (.id // .tool_call_id), name: (.name // .tool_name)} | if $names then . else {id} end]} else {} end;
# [text, counts, n] -> each pair [found, replacement, rule] replaced in order as a literal, counted by rule. With $ord,
# each replacement is numbered #n+1, #n+2, ... in the order made: two redacted keys of one object cannot collide.
def jp_count($pairs; $ord):
  reduce $pairs[] as $x (.;
    (.[0] | split($x[0])) as $s
    | if $s[1:] == [] then .
      else [(if $ord then .[2] as $n | [$s[0], (range(1; $s | length) as $i | $x[1] + "#" + ($n + $i | tostring) + $s[$i])] | add
             else $s | jp_join($x[1]) end),
            (.[1] | setpath([$x[2]]; .[$x[2]] + ($s | length) - 1)), .[2] + ($s | length) - 1]
      end);
# A window's matches, leftmost first, each one's copies blanked to the same length before the next search: one search per
# distinct match, so a secret repeated thousands of times costs one search, not a global match whose offsets jq counts
# from the start of the string for every copy. An empty match (a user regex) falls back to the global match.
def jp_scan($re; $budget):
  if $budget < 0 then error("redaction limit") else
    _match_impl($re; null; false)[0] as $m
    | if $m == null then empty
      elif $m.length == 0 then _match_impl($re; "g"; false)[] | if .length > 0 then . else empty end
      else $m, (split($m.string) | jp_join("\u0001" * $m.length) | jp_scan($re; $budget - 1)) end
  end;
# Each distinct complete match of rule $p on the unchanged text, as [found, kept group + placeholder, rule], sorted.
# Windows: see limits in redact-rules.json. A match that reaches the end of a window that is not the last is left to the
# next window, which holds it from its start; one that starts before the next window does fails closed.
def jp_found($p; $l):
  . as $t
  | (if $p.whole or length <= $l.window then 1 else ((length - $l.window) / $l.step | ceil) + 1 end) as $k
  # A window is matched after every complete match found before it is blanked in it to the same length, so a secret
  # that repeats is matched once, not once per copy, and every offset stays where it was.
  | reduce range(0; $k) as $i ([];
      . as $found
      | (if $k == 1 then $t else reduce $found[] as $x ($t[$i * $l.step:$i * $l.step + $l.window]; split($x[0]) | jp_join("\u0001" * ($x[0] | length))) end) as $w
      # A short window: one global match, one regex compile, and few enough matches that their offsets cost little. A long
      # one: the scan, so a secret repeated thousands of times costs one search.
      | (. + [$w | if length <= $l.dense
                   then _match_impl($p.re; "g"; false)[] | if .length > 0 then . else empty end else jp_scan($p.re; $l.pairs) end
              | if $i + 1 == $k or .offset + .length < ($w | length) then
                  [.string, (.captures[0] | if .name == "p" then .string else null end) + "[REDACTED:" + $p.id + "]", $p.id]
                elif .offset < $l.step then error("redaction limit")
                else empty end]
         | unique)
      | if length > $l.pairs then error("redaction limit") else . end);
# Every rule whose needs and pre the text passes, over the unchanged text: one pair per found text, first in rule order,
# longest first. A whole rule fails when the text holds more than `most` of its first need.
def jp_pairs($rules; $l):
  . as $t
  | [$rules[] | . as $p
     | if ($t | jp_holds($p.needs)) and ($p.pre == null or ($t | jp_test($p.pre))) then
         if $p.most and ($t | split($p.needs[0]) | length) > $p.most + 1 then error("redaction limit") else $t | jp_found($p; $l)[] end
       else empty end]
  | unique_by(.[0]) | sort_by(-(.[0] | length));
# {rule, trigger} for the first sensitive path or env dump the tool input names, or null. Path rules see one word per line,
# from path and command values and the prompt's @mentions. Command rules see the values, one per line.
def jp_sensitive($c; $r):
  $r.sensitive as $s
  | ([(.tool_input, (.tool_calls[]? | if type == "object" then .arguments, .input else empty end))
     | if type == "string" then (. as $t | try fromjson catch $t) else . end
     | if type == "string" then {command: .} else . end
     | .. | if type == "object" then . as $o | keys_unsorted[] | if $s.keys[.] then $o[.] | .. | if type == "string" then . else empty end else empty end else empty end]
     | jp_join("\n")) as $cmds
  # Split, not a global match: jq counts each match's offset from the start of the string, so many matches cost its square.
  | (($cmds | jp_lines($s.split)) + "\n"
     + (if (.prompt | type) == "string" and (.prompt | contains("@")) then
          if (.prompt | split("@") | length) > $r.limits.mentions + 1 then error("redaction limit") else . end
          | [.prompt | jp_lines($s.mention.split) | split("\n")[] | if startswith("@") then .[1:] | jp_lines($s.mention.end) | split("\n")[0] // "" else "" end
             | if . == "" then empty else . end] | jp_join("\n")
        else "" end)) as $words
  | first(
      ($s.paths[], ($c.redactPaths // [] | .[]? | if type == "string" and . != "" then
          {id: "custom_path", re: ("(?<![^\\n/])" + ([explode[] | [.] | implode | $s.glob[.] // .] | jp_join("")) + "(?![^\\n])")} else empty end)
       | . as $p | if ($words | jp_holds($p.needs)) and ($words | jp_test($p.re)) then
           {rule: $p.id, trigger: ($words | _match_impl("(?<![^\\n])[^\\n]*?(?:" + $p.re + ")[^\\n]*"; null; false)[0].string)} else empty end),
      ($s.commands[] | . as $p | if ($cmds | jp_holds($p.needs)) and ($cmds | jp_test($p.re)) then {rule: $p.id, trigger: ("cmd:" + $p.id)} else empty end),
      null);
# Every field the class table does not keep plain, redacted, with its records.
def jp_scrub($c; $r):
  . as $ev
  | (reduce (keys_unsorted[] | if $r.fields.plain[.] then empty else . end) as $k ({}; . + {($k): $ev[$k]})) as $sub
  | ([$sub | .. | if type == "string" then . elif type == "object" then keys_unsorted[] else empty end] | jp_join("\n")) as $text
  | ([$ENV | keys_unsorted[] as $k | $ENV[$k] | (., (if contains("\n") then split("\n")[] | rtrimstr("\r") else empty end))
      | if length >= $r.env.min_length and (. as $v | $text | contains($v)) and jp_test("\\S") then [., $k] else empty end]
     | ([.[][1]] | unique | jp_join("\n") | [_match_impl($r.env.name_re; "g"; false)[] | {(.string): true}] | add) as $secret
     | [.[] | if $secret[.[1]] then [.[0], "[REDACTED:env:" + .[1] + "]", "env:" + .[1]] else empty end]
     | unique | sort_by(-(.[0] | length), .[1])) as $env
  | ($env + ($text | jp_pairs($r.patterns + [$c.redactPatterns // [] | .[]? | if type == "string" and . != "" then {id: "custom", re: .} else empty end]; $r.limits))
     | unique_by(.[0]) | sort_by(-(.[0] | length))) as $pairs
  | ([$sub[]] | tojson | jp_pairs($r.members; $r.limits)) as $members
  | if ($pairs | length) + ($members | length) > $r.limits.pairs then error("redaction limit")
    elif $pairs == [] and $members == [] then .
    else
      [$pairs[] | [(.[0] | tojson | .[1:-1]), (.[1] | tojson | .[1:-1]), .[2]]] as $jpairs
      | reduce ($sub | keys_unsorted[]) as $k ([., []];
          ($sub[$k] | [.. | objects | length] | add) as $size
          | ([($sub[$k] | tojson), {}, 0] | jp_count($members; false)) as $m
          | ($m | jp_count($jpairs; false) | [(.[0] | fromjson), .[1]]
             # Two keys redacted to one name would lose a member: number every replacement in this field instead.
             | if (.[0] | [.. | objects | length] | add) == $size then . else $m | jp_count($jpairs; true) | [(.[0] | fromjson), .[1]]
               | if (.[0] | [.. | objects | length] | add) == $size then . else error("redaction limit") end end) as $d
          | [(.[0] | setpath([$k]; $d[0])), .[1] + [$d[1] | keys_unsorted[] as $rule | {rule: $rule, field: $k, count: $d[1][$rule]}]])
      | .[0] + (if .[1] == [] then {} else {redactions: (.[1] | sort_by(.field, .rule))} end)
    end;
# The sanitized event with its records. When a rule fails (an invalid user regex, a limit), only what is never redacted,
# without labels or tool names: never the raw text.
def judgment_redact($c; $capture):
  $redact_rules::redact_rules[0] as $r
  | . as $raw
  | try (
      (if $capture == "metadata" then jp_pick($r.fields.strip) + jp_calls(true) + {prompt: "", output: ""}
       else jp_sensitive($c; $r) as $hit
         | if $hit then jp_pick($r.fields.strip) + jp_calls(true) + {sensitive: true, sensitive_rule: $hit.rule, sensitive_trigger: $hit.trigger} else . end
       end)
      | jp_scrub($c; $r))
    catch ($raw | jp_pick($r.fields.bare) + jp_calls(false) + {redaction_error: true});

def judgment_prepare($cfg; $mode; $root; $model):
  if .kind == "command" then
    .
  else
    # An empty or cut-off opt-in file is no opt-in. It must never fall back to full capture.
    (if ($cfg | type) == "object" then $cfg else {enabled: false} end) as $c
    | if $c.enabled == false or ({on: "on", off: "off"}[$mode] // {off: "off"}[$c.sessionDefault | strings] // "on") != "on" or (.session_id // "") == "" then
        empty
      else
        # The capture table. judgment_core/optin.py resolve_capture is the same table.
        ({metadata: "metadata", full: "full", sanitized: "sanitized"}[$c.capture | strings] // if $c.redact == false then "full" else "sanitized" end) as $capture
        | if $capture == "full" then . else judgment_redact($c; $capture) end
        | . + {capture: $capture, ts: (.ts | numbers // (now * 1000 | floor)),
               environment: ([$c.environment | strings | if jp_test("\\S") then . else empty end][0] // "production"), project_root: $root}
        | if (.model // "") == "" and $model != "" then . + {model: $model} else . end
      end
  end;
