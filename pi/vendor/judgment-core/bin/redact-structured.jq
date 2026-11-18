# Typed values under secret-named keys: numbers, arrays, and objects are masked whole, then the rest of the event is
# scrubbed again. Flat on purpose (see prepare.jq): the caller passes in the two steps that live in other modules.
#   include "prepare"; include "redact-replace"; include "redact-structured";
#   .[0] as $f | $f | js_resume(jr_replace; jp_rescrub(jr_replace)) | jr_stamp($f)
def js_holds($needs): . as $t | $needs == null or ([$needs[] as $n | $t | contains($n) | if . then 1 else empty end] != []);
def js_marker($limit):
  ([.. | arrays | if length == 1 then .[0] | numbers else empty end
    | if . >= 0 and . <= $limit and floor == . then {((. + 0) | tostring): true} else empty end] | add // {}) as $used
  | first(range(0; $limit + 1) | . as $n | if $used[($n | tostring)] then empty else $n end) // error("redaction limit");
def js_key($members; $key):
  ("{" + ($key | tojson) + ":\"x\"}") as $probe
  | [$members[] as $p | if ($probe | js_holds($p.needs) and _match_impl($p.re; null; true)) then 1 else empty end] != [];
def js_mask($members; $marker; $budget):
  if type == "object" then . as $o
    | reduce keys_unsorted[] as $k ([{}, 0];
        $o[$k] as $v
        | if (($v | type) as $t | $t == "number" or $t == "array" or $t == "object") and js_key($members; $k) then
            [.[0] + {($k): [$marker]}, .[1] + 1]
          else ($budget - .[1]) as $left | ($v | js_mask($members; $marker; $left)) as $d
            | [.[0] + {($k): $d[0]}, .[1] + $d[1]] end
        | if .[1] > $budget then error("redaction limit") else . end)
  elif type == "array" then
    reduce .[] as $v ([[], 0]; ($budget - .[1]) as $left | ($v | js_mask($members; $marker; $left)) as $d
      | [.[0] + [$d[0]], .[1] + $d[1]]
      | if .[1] > $budget then error("redaction limit") else . end)
  else [., 0] end;
def js_restore($marker):
  if type == "array" then
    if length == 1 and (.[0] | type) == "number" and .[0] == $marker then "[REDACTED:secret_field]"
    else map(js_restore($marker)) end
  elif type == "object" then with_entries(.value |= js_restore($marker))
  else . end;
def js_structured(rescrub):
  . as $frame | ($frame.sub | js_marker($frame.rules.limits.sender_pairs)) as $marker
  | (reduce ($frame.sub | keys_unsorted[]) as $k ([{}, [], 0];
       ($frame.budget - .[2]) as $left | ($frame.sub[$k] | js_mask($frame.policy.members; $marker; $left)) as $d
       | [.[0] + {($k): $d[0]},
          .[1] + (if $d[1] > 0 then [{rule: "secret_field", field: $k, count: $d[1]}] else [] end), .[2] + $d[1]])) as $masked
  | ($frame.rules | .limits.pairs -= $masked[2]) as $rules
  | {event: ($frame.event + $masked[0] | if $masked[2] > 0 then del(.redactions) else . end),
     policy: ($frame.policy + {structured: []}), rules: $rules}
  | rescrub
  | reduce ($frame.sub | keys_unsorted[]) as $k (.; .[$k] |= js_restore($marker))
  | if $masked[2] > 0 then
      .redactions = (($masked[1] + (.redactions // [])) | group_by([.field, .rule])
        | map({rule: .[0].rule, field: .[0].field, count: (map(.count) | add)}))
    else . end;
# A frame from prepare.jq, finished: its event with every find replaced, or its error form when a rule fails.
def js_resume(replace; rescrub):
  . as $frame | try (if .structured then js_structured(rescrub) else replace end) catch $frame.error;
