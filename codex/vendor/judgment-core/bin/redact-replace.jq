def jr_size: [.. | objects | length] | add;
def jr_pick($set): delpaths([keys_unsorted[] | if $set[.] then empty else [.] end]);
def jr_blob: [.. | if type == "string" then . elif type == "object" then keys_unsorted[] else empty end] | join("\n");
def jp_sorted_spans:
  if length < 2 then .[0] // [] else
    (length / 2 | floor) as $m
    | (.[:$m] | jp_sorted_spans) as $a | (.[$m:] | jp_sorted_spans) as $b
    | [foreach range(0; ($a | length) + ($b | length)) as $_ ([0, 0, null];
         if .[1] >= ($b | length) or (.[0] < ($a | length)
             and ($a[.[0]][0] < $b[.[1]][0] or ($a[.[0]][0] == $b[.[1]][0] and $a[.[0]][1] >= $b[.[1]][1]))) then
           [.[0] + 1, .[1], $a[.[0]]]
         else [.[0], .[1] + 1, $b[.[1]]] end; .[2])]
  end;
def jp_spans($pairs):
  . as $t
  | [$pairs[] as $x | ("[REDACTED:" + $x[2] + "]") as $marker
     | ($x[1] | split($marker)) as $kept
     | if $x[0] == "" or ($kept | length) != 2 or ($x[0] | startswith($kept[0]) and endswith($kept[1]) | not)
          or (($kept[0] | length) + ($kept[1] | length) > ($x[0] | length)) then error("redaction limit") else . end
     | ($x[0] | length) as $n
     | [foreach ($t | split($x[0]) | .[:-1][]) as $part (0; . + ($part | length) + $n;
         if $t[. - $n + 1:. + $n - 1] | contains($x[0]) then error("redaction limit")
         else [. - $n, ., . - $n + ($kept[0] | length), . - ($kept[1] | length), $x[2]] end)]]
  | jp_sorted_spans
  | reduce .[] as $s ([];
      if length > 0 and $s[0] < .[-1][1] then
        .[-1][1] = ([.[-1][1], $s[1]] | max) | .[-1][2] = ([.[-1][2], $s[2]] | min)
        | .[-1][3] = ([.[-1][3], $s[3]] | max)
      else . + [$s] end);
def jp_merged($pairs):
  . as $t | jp_spans($pairs)
  | map([$t[.[0]:.[1]], ($t[.[0]:.[2]] + "[REDACTED:" + .[4] + "]" + $t[.[3]:.[1]]), .[4]])
  | reduce .[] as $p ({};
      if has($p[0]) and .[$p[0]] != $p then error("redaction limit") else .[$p[0]] = $p end)
  | [.[]]
  | sort_by(-(.[0] | length), .[0]);
def jp_count($pairs; $ord):
  . as $in | .[0] as $t | ($t | jp_spans($pairs)) as $spans
  | reduce $spans[] as $s ([[], 0, $in[1], $in[2]];
      .[0] += [$t[.[1]:$s[2]] + "[REDACTED:" + $s[4] + "]"
                + (if $ord then "#" + (.[3] + 1 | tostring) else "" end) + $t[$s[3]:$s[1]]]
      | .[1] = $s[1] | .[2][$s[4]] += 1 | .[3] += 1)
  | [(.[0] + [$t[.[1]:]] | add // ""), .[2], .[3]];

def jr_replace:
  . as $frame | .event | $frame.sub as $sub | $frame.members as $members
  | ($sub | jr_blob | jp_merged($frame.pairs)) as $pairs
  | [$pairs[] | [(.[0] | tojson | .[1:-1]), (.[1] | tojson | .[1:-1]), .[2]]] as $jpairs
      | ([$sub[]] | tojson) as $canonical
      | if [$members[], $jpairs[] | .[0] as $f | if $canonical | contains($f) then empty else 1 end] != []
           or ($canonical | jp_merged($members + $jpairs) | length) > $frame.budget then error("redaction limit") else . end
      # The pairs are replaced in each field's JSON text only when every one of them can sit nowhere but inside a string:
      # no quote, and a character no number, true, false, null, or punctuation holds. Otherwise they are replaced in each
      # string value, one by one, and a key that holds one fails closed below.
      | ([$jpairs[] | .[0] | if contains("\"") or _match_impl("^[-+.0-9eEtrufalsn{}\\[\\],:]*$"; null; true) then 1 else empty end] == []) as $in_text
      | reduce ($sub | keys_unsorted[]) as $k ([., []];
          ($sub[$k] | jr_size) as $size
          | (if $in_text then
               [($sub[$k] | tojson), {}, 0] | jp_count($members + $jpairs; false) | [(.[0] | fromjson), .[1]]
               # Two keys redacted to one name would lose a member: number every replacement in this field instead.
               | if (.[0] | jr_size) == $size then . else [($sub[$k] | tojson), {}, 0] | jp_count($members + $jpairs; true) | [(.[0] | fromjson), .[1]]
                 | if (.[0] | jr_size) == $size then . else error("redaction limit") end end
             else
               [($sub[$k] | tojson), {}, 0] | jp_count($members; false) | [(.[0] | fromjson), .[1]]
               | reduce path(.[0] | .. | strings) as $p (.; ([getpath($p), .[1], 0] | jp_count($pairs; false)) as $y | setpath($p; $y[0]) | setpath([1]; $y[1]))
             end) as $d
          | [(.[0] | setpath([$k]; $d[0])), .[1] + [$d[1] | keys_unsorted[] as $rule | {rule: $rule, field: $k, count: $d[1][$rule]}]])
      | .[0] as $out | .[1] as $recs
      # A found text still there spanned two strings (the blob joins them with a newline), or sits in a key the string pass
      # does not reach: fail closed.
      | ($out | jr_pick($sub) | jr_blob) as $after
      | if [$pairs[] | .[0] as $f | $after | contains($f) | if . then 1 else empty end] != [] then error("redaction limit") else . end
      | $out + (if $recs == [] then {} else {redactions: ($recs | sort_by(.field, .rule))} end);
def jr_stamp($frame):
  . + $frame.stamp
  | if (.model // "") == "" and $frame.default_model != "" then . + {model: $frame.default_model} else . end;
def jr_resume:
  . as $frame | (try jr_replace catch $frame.error) | jr_stamp($frame);
