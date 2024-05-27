#!/usr/bin/env bash

# shellcheck disable=SC2016,SC2086

set -Eeuo pipefail

script_name=$(basename "${BASH_SOURCE[0]}")
args=(DIR_PATH)

usage() {
    cat << EOF
Format jsonl dataset with global time split in multae format.

USAGE: ${script_name} [OPTIONS] ${args[*]}

OPTIONS:
  -o, --out-dir    Output directory. Default current directory.
  -h, --help       Print help.
EOF
    exit
}

msg() {
    echo >&2 -e "${1-}"
}

die() {
    local msg=$1
    local code=${2-1}
    msg "${script_name}: ${msg}"
    exit "${code}"
}

parse_params() {
    while :; do
        case "${1-}" in
        -h | --help) usage ;;
        -o | --out-dir)
            out_dir="${2-}"
            shift
            ;;
        -?*) die "Unknown option: $1" ;;
        *) break ;;
        esac
        shift
    done
    argv=("$@")
    paste <(tr " " "\n" <<< "${args[@]}") <(tr " " "\n" <<< "${argv[@]:-""}") | while read -r arg value; do
        if [ -z "${value}" ]; then
            die "Missing argument: ${arg}"
        fi
    done
    return 0
}

build_eval() {
  dst=$1
  src=$2
  mlr --ijsonl --ojsonl join -j user -f \
    <(mlr --ijsonl --ojson cat "${src}/train-with-fold-in.jsonl" | jq -c 'group_by(.user) | map({user: (.[0].user), source: [.[] | .item] })[]') \
    <(mlr --ijsonl --ojsonl put '$target = $item' "${src}/eval-grouped.jsonl" | mlr --ijsonl --ojsonl cut -f user,target) \
    | mlr --ijsonl --ojsonl cut -f user,source,target > "${dst}"
  msg "Built ${dst}"
}

build_test() {
  dst=$1
  src=$2
  mlr --ijsonl --ojsonl join -j user -f \
    <(mlr --ijsonl --ojson cat "${src}/full-train-with-fold-in.jsonl" | jq -c 'group_by(.user) | map({user: (.[0].user), source: [.[] | .item] })[]') \
    <(mlr --ijsonl --ojsonl put '$target = $item' "${src}/test-grouped.jsonl" | mlr --ijsonl --ojsonl cut -f user,target) \
    | mlr --ijsonl --ojsonl cut -f user,source,target > "${dst}"
  msg "Built ${dst}"
}

build_train() {
  dst=$1
  src=$2
  mlr --ijsonl --ojsonl join -j user -f \
    <(mlr --ijsonl --ojson cat "${src}/train-with-fold-in.jsonl" | jq -c 'group_by(.user) | map({user: (.[0].user), source: [.[] | .item] })[]') \
    <(mlr --ijsonl --ojson cat "${src}/train-with-fold-in.jsonl" | jq -c 'group_by(.user) | map({user: (.[0].user), target: [.[] | .item] })[]') \
    | mlr --ijsonl --ojsonl cut -f user,source,target > "${dst}"
  msg "Built ${dst}"
}

build_full_train() {
  dst=$1
  src=$2
  mlr --ijsonl --ojsonl join -j user -f \
    <(mlr --ijsonl --ojson cat "${src}/full-train-with-fold-in.jsonl" | jq -c 'group_by(.user) | map({user: (.[0].user), source: [.[] | .item] })[]') \
    <(mlr --ijsonl --ojson cat "${src}/full-train-with-fold-in.jsonl" | jq -c 'group_by(.user) | map({user: (.[0].user), target: [.[] | .item] })[]') \
    | mlr --ijsonl --ojsonl cut -f user,source,target > "${dst}"
  msg "Built ${dst}"
}

if ! command -v mlr &> /dev/null; then
    die "miller (mlr) is not installed. Aborting"
fi

out_dir=${PWD}

parse_params "$@"

dir_path=${argv[0]}
if ! [ -e "${dir_path}" ]; then
    die "${dir_path}: No such file or directory"
fi
mkdir -p "${out_dir}"

build_train "${out_dir}/multae-train-with-fold-in.jsonl" "${dir_path}"
build_eval "${out_dir}/multae-eval-with-fold-in.jsonl" "${dir_path}"
build_full_train "${out_dir}/multae-full-train-with-fold-in.jsonl" "${dir_path}"
build_test "${out_dir}/multae-test-with-fold-in.jsonl" "${dir_path}"
