# Shared by adapter maps. One jq applies the map and this filter.
# Commands pass through so a session toggle still runs when tracing is off.
#
# An event goes map -> capture policy -> sensitive check -> redact -> inbox. The rules are data: redact-rules.json is
# the source, and scripts/gen_redact.py composes it into redact-compiled.json, which this loads. judgment_core/redact.py
# reads the same source, and its functions mirror the steps below (tests/test_redact.py checks they agree).
#
# Redaction: one blob of every string and key in the event's text fields gives the pairs (found text, replacement, rule):
# env values, then each rule, every one a complete match on the unchanged blob, one pair per found text, longest first.
# Members under secret keys are found in the fields' JSON as "key":"value", made exact with the closing quote. Each field
# then gets its members, then the pairs, replaced as literals in its JSON, when every pair can only sit inside a string
# there (no quote; a character no number, true, false, null, or punctuation holds): so a replacement never touches a
# number or the syntax. Otherwise the pairs are replaced in each string value. Every replacement is counted into records
# {rule, field, count}, carried on the event as `redactions`. When two keys of one object would become the same
# placeholder (the field has fewer members after), the field is redone with every replacement numbered (#1, #2, ...), so
# no member is lost. A found text still there afterwards (it spanned two strings of the blob) fails the event closed.
# JSON nested in a string deeper than the rules read is decoded into layers (jp_layers); a find there is replaced as
# encoded where it sits. env_found records which env values the raw text held, before anything can fail.
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
# Every string and key, in order: the items of the blob rules run over, one per line.
def jp_items: [.. | if type == "string" then . elif type == "object" then keys_unsorted[] else empty end];
def jp_blob: jp_items | jp_join("\n");
# The text as it reads $n JSON string encodings down.
def jp_esc($n): if $n == 0 then . else tojson | .[1:-1] | jp_esc($n - 1) end;
# Each JSON object or array a string holds whole, or from the first { or [ of it or of one of its lines, decoded.
def jp_containers:
  (., (if contains("\n") then split("\n")[] else empty end)) as $piece
  | ("{", "[") as $o
  | ($piece | split($o)) as $parts
  | if ($parts | length) < 2 then empty
    else ($o + ($parts[1:] | jp_join($o))) | try fromjson catch null | if type == "object" or type == "array" then . else empty end end;
# For each depth below the items, every string and key of the JSON nested in the strings above it that hold an escaped
# backslash before a quote (JSON deeper than a quote, or one escaped quote, which the rules read). Deeper than $max fails.
def jp_layers($max):
  def go($d):
    [.[] | if contains("\\\"") then jp_containers | jp_items[] else empty end] as $next
    | if $next == [] then empty elif $d == $max then error("redaction limit") else $next, ($next | go($d + 1)) end;
  go(0);
# How many members the value's objects hold in all.
def jp_size: [.. | objects | length] | add;
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
# A window's matches, leftmost first, one search per distinct match: a secret repeated thousands of times costs one search,
# not a global match whose offsets jq counts from the start of the string for every copy. After each match, only its copies
# followed by the character that ended it are blanked (to the same length, that character kept): a copy followed by anything
# else may start a longer credential, which a later search then finds whole. A match that ends the window is the last one.
# An empty match (a user regex) falls back to the global match.
def jp_scan($re; $budget):
  if $budget < 0 then error("redaction limit") else
    _match_impl($re; null; false)[0] as $m
    | if $m == null then empty
      elif $m.length == 0 then _match_impl($re; "g"; false)[] | if .length > 0 then . else empty end
      else $m, (($m.offset + $m.length) as $e | .[$e:$e + 1] as $c
                | if $c == "" then empty else split($m.string + $c) | jp_join("\u0001" * $m.length + $c) | jp_scan($re; $budget - 1) end) end
  end;
# Each distinct complete match of rule $p on the unchanged text, as [found, kept group + placeholder, rule], sorted.
# Windows: see limits in redact-rules.json. Each window is matched as it is in the text, so every match is a complete span.
# A match that reaches the end of a window that is not the last is left to the next window, which holds it from its start;
# one that starts before the next window does fails closed.
def jp_found($p; $l):
  . as $t
  | (if $p.whole or length <= $l.window then 1 else ((length - $l.window) / $l.step | ceil) + 1 end) as $k
  | reduce range(0; $k) as $i ([];
      (if $k == 1 then $t else $t[$i * $l.step:$i * $l.step + $l.window] end) as $w
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
     | .. | if type == "object" then . as $o | keys_unsorted[] | if $s.keys[.] then $o[.] | .. | if type == "string" then . else empty end else empty end else empty end]) as $vals
  | ($vals | jp_join("\n")) as $cmds
  # Split, not a global match: jq counts each match's offset from the start of the string, so many matches cost its square.
  | (($cmds | jp_lines($s.split)) + "\n"
     + (if (.prompt | type) == "string" and (.prompt | contains("@")) then
          if (.prompt | split("@") | length) > $r.limits.mentions + 1 then error("redaction limit") else . end
          | [.prompt | jp_lines($s.mention.split) | split("\n")[] | if startswith("@") then .[1:] | jp_lines($s.mention.end) | split("\n")[0] // "" else "" end
             | if . == "" then empty else . end] | jp_join("\n")
        else "" end)) as $words
  | first(
      ($s.paths[], ($c.redactPaths // [] | .[]? | if type == "string" and . != "" then
          {id: "custom_path", re: ("(?<![^\\n/:])" + ([explode[] | [.] | implode | $s.glob[.] // .] | jp_join("")) + "(?![^\\n])")} else empty end)
       | . as $p | if ($words | jp_holds($p.needs)) and ($words | jp_test($p.re)) then
           {rule: $p.id, trigger: ($words | _match_impl("(?<![^\\n])[^\\n]*?(?:" + $p.re + ")[^\\n]*"; null; false)[0].string)} else empty end),
      # Each value is its own command line: a quote one leaves open never hides the next one's command.
      ($s.commands[] | . as $p | if [$vals[] | if jp_holds($p.needs) and jp_test($p.re) then 1 else empty end] != [] then {rule: $p.id, trigger: ("cmd:" + $p.id)} else empty end),
      null);
# [value, placeholder, rule] for each secret-named env value (and each line of one that spans lines) the text holds.
def jp_env($text; $r):
  [$ENV | keys_unsorted[] as $k | $ENV[$k] | (., (if contains("\n") then split("\n")[] | rtrimstr("\r") else empty end))
   | if length >= $r.env.min_length and (. as $v | $text | contains($v)) and jp_test("\\S") then [., $k] else empty end]
  | ([.[][1]] | unique | jp_join("\n") | [_match_impl($r.env.name_re; "g"; false)[] | {(.string): true}] | add) as $secret
  | [.[] | if $secret[.[1]] then [.[0], "[REDACTED:env:" + .[1] + "]", "env:" + .[1]] else empty end]
  | unique | sort_by(-(.[0] | length), .[1]);
# {name, field} for each secret env value a redacted field's text holds, whatever rule then replaces it and whether or
# not the rules fail: the sender reads it to know which env values the hook's text depended on (flush.carry_records).
def jp_env_found($r):
  . as $e
  | [keys_unsorted[] | if $r.fields.plain[.] then empty else . end | . as $k
     | jp_env($e[$k] | jp_blob; $r)[] | {name: .[2][4:], field: $k}]
  | unique | sort_by(.field, .name);
# Every field the class table does not keep plain, redacted, with its records.
def jp_scrub($c; $r):
  . as $ev
  | (reduce (keys_unsorted[] | if $r.fields.plain[.] then empty else . end) as $k ({}; . + {($k): $ev[$k]})) as $sub
  | ($sub | jp_items) as $items
  | ($items | jp_join("\n")) as $text
  | ($r.patterns + [$c.redactPatterns // [] | .[]? | if type == "string" and . != "" then {id: "custom", re: .} else empty end]) as $rules
  # The blob's pairs, then each layer of JSON nested in its strings: a find there is encoded as deep as its layer, so it is
  # replaced where it sits. One whose encoding is not in the blob (its JSON was written another way) fails closed.
  | (jp_env($text; $r) + ($text | jp_pairs($rules; $r.limits))
     + ([$items | jp_layers($r.limits.depth)] as $layers
        | [range(0; $layers | length) as $i | ($layers[$i] | jp_join("\n")) as $lt
           | (jp_env($lt; $r) + ($lt | jp_pairs($rules; $r.limits)))[]
           | [(.[0] | jp_esc($i + 1)), (.[1] | jp_esc($i + 1)), .[2]]
           | . as $x | if $text | contains($x[0]) then . else error("redaction limit") end])
     | unique_by(.[0]) | sort_by(-(.[0] | length))) as $pairs
  # A member's text "key":"value is made exact with its closing quote: it is replaced only where it is that whole member.
  | ([$sub[]] | tojson | [jp_pairs($r.members; $r.limits)[] | [.[0] + "\"", .[1] + "\"", .[2]]]) as $members
  | if ($pairs | length) + ($members | length) > $r.limits.pairs then error("redaction limit")
    elif $pairs == [] and $members == [] then .
    else
      [$pairs[] | [(.[0] | tojson | .[1:-1]), (.[1] | tojson | .[1:-1]), .[2]]] as $jpairs
      # The pairs are replaced in each field's JSON text only when every one of them can sit nowhere but inside a string:
      # no quote, and a character no number, true, false, null, or punctuation holds. Otherwise they are replaced in each
      # string value, one by one, and a key that holds one fails closed below.
      | ([$jpairs[] | .[0] | if contains("\"") or jp_test("^[-+.0-9eEtrufalsn{}\\[\\],:]*$") then 1 else empty end] == []) as $in_text
      | reduce ($sub | keys_unsorted[]) as $k ([., []];
          ($sub[$k] | jp_size) as $size
          | ([($sub[$k] | tojson), {}, 0] | jp_count($members; false)) as $m
          | (if $in_text then
               $m | jp_count($jpairs; false) | [(.[0] | fromjson), .[1]]
               # Two keys redacted to one name would lose a member: number every replacement in this field instead.
               | if (.[0] | jp_size) == $size then . else $m | jp_count($jpairs; true) | [(.[0] | fromjson), .[1]]
                 | if (.[0] | jp_size) == $size then . else error("redaction limit") end end
             else
               [($m[0] | fromjson), $m[1]]
               | reduce path(.[0] | .. | strings) as $p (.; ([getpath($p), .[1], 0] | jp_count($pairs; false)) as $y | setpath($p; $y[0]) | setpath([1]; $y[1]))
             end) as $d
          | [(.[0] | setpath([$k]; $d[0])), .[1] + [$d[1] | keys_unsorted[] as $rule | {rule: $rule, field: $k, count: $d[1][$rule]}]])
      | .[0] as $out | .[1] as $recs
      # A found text still there spanned two strings (the blob joins them with a newline), or sits in a key the string pass
      # does not reach: fail closed.
      | ($out | jp_pick($sub) | jp_blob) as $after
      | if [$pairs[] | .[0] as $f | $after | contains($f) | if . then 1 else empty end] != [] then error("redaction limit") else . end
      | $out + (if $recs == [] then {} else {redactions: ($recs | sort_by(.field, .rule))} end)
    end;
# The sanitized event with its records. When a rule fails (an invalid user regex, a limit), only what is never redacted,
# without labels or tool names: never the raw text.
def judgment_redact($c; $capture):
  $redact_rules::redact_rules[0] as $r
  # Recorded before anything can fail, so a failure that drops the text still says which env values it held.
  | jp_env_found($r) as $found
  | del(.env_found) + (if $found == [] then {} else {env_found: $found} end)
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
