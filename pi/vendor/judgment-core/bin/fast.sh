# Shared by hook scripts. Source this file. Do not execute it.
# Helpers set JR instead of printing, so callers need no subshell. A fork costs
# 5-10 ms on a busy laptop, and hooks run on the agent's critical path.

judgment_tracing_off() {
  case ${JUDGMENT_TRACING:-} in
    0|false|FALSE|no|NO|off|OFF) return 0 ;;
  esac
  return 1
}

# judgment_field JSON KEY: JR is the first string value of KEY, or the call fails.
judgment_field() {
  JR=
  _rest=$1
  case $_rest in
    *"\"$2\":\""*) _rest=${_rest#*"\"$2\":\""} ;;
    *"\"$2\": \""*) _rest=${_rest#*"\"$2\": \""} ;;
    *) return 1 ;;
  esac
  JR=${_rest%%\"*}
}

judgment_json_escape() {
  _s=$1
  JR=
  while [ -n "$_s" ]; do
    _c=${_s%"${_s#?}"}
    _s=${_s#?}
    case $_c in
      \\) JR="${JR}\\\\" ;;
      \") JR="${JR}\\\"" ;;
      *) JR="${JR}${_c}" ;;
    esac
  done
}

# judgment_project_root DIR: JR is the nearest directory with .judgment/tracing.json.
judgment_project_root() {
  _dir=${1:-$PWD}
  case $_dir in
    /*) ;;
    *) _dir=$PWD/$_dir ;;
  esac
  while :; do
    if [ -f "$_dir/.judgment/tracing.json" ]; then
      JR=${_dir:-/}
      return 0
    fi
    [ -n "$_dir" ] && [ "$_dir" != / ] || return 1
    _dir=${_dir%/*}
  done
}
