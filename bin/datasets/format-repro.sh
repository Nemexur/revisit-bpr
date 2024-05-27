#!/usr/bin/env bash

# shellcheck disable=SC2016,SC2086

set -Eeuo pipefail

script_name=$(basename "${BASH_SOURCE[0]}")
args=(DIR_PATH)

usage() {
    cat << EOF
Format Revisit iALS datasets to jsonl.

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

build_jsonl() {
  dst=$1
  src=$2
  bin/datasets/jsonl.sh \
    <(mlr --csv put '$item = $sid + 1; $user = $uid + 1' + cut -f user,item ${src}) \
    > "${dst}"
  msg "Built ${dst}"
}

build_user_seen() {
  dst=$1
  src=$2
  bin/datasets/jsonl.sh -g \
    <(mlr --csv put '$item = $sid + 1; $user = $uid + 1' + cut -f user,item ${src}) \
    | mlr --jsonl put '$seen_items = $item' + cut -f user,seen_items > "${dst}"
  msg "Built ${dst}"
}

build_grouped() {
  dst=$1
  src=$2
  bin/datasets/jsonl.sh -g \
    <(mlr --csv put '$item = $sid + 1; $user = $uid + 1' + cut -f user,item ${src}) \
    > "${dst}"
  msg "Built ${dst}"
}

shuf_dataset() {
    dst=$1
    src=$2
    shuf --random-source=<(RANDOM=13; while true; do echo $RANDOM; done) "${src}" > "${dst}"
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

# Train
build_jsonl "${out_dir}/train.jsonl" "${dir_path}/train.csv"
shuf_dataset "${out_dir}/shuf-train.jsonl" "${out_dir}/train.jsonl"
build_user_seen "${out_dir}/train-user-seen-items.jsonl" "${dir_path}/train.csv"
build_grouped "${out_dir}/train-grouped.jsonl" "${dir_path}/train.csv"

# Fold-in Eval
build_jsonl "${out_dir}/eval-fold-in.jsonl" "${dir_path}/validation_tr.csv"
shuf_dataset "${out_dir}/shuf-eval-fold-in.jsonl" "${out_dir}/eval-fold-in.jsonl"
build_user_seen "${out_dir}/eval-fold-in-user-seen-items.jsonl" "$(echo "${dir_path}"/{train.csv,validation_tr.csv})"

# Eval
build_grouped "${out_dir}/eval-grouped.jsonl" "${dir_path}/validation_te.csv"
build_user_seen \
  "${out_dir}/eval-user-seen-items.jsonl" \
  "$(echo "${dir_path}"/{train.csv,validation_tr.csv,validation_te.csv})"

# Train with fold-in
build_jsonl "${out_dir}/train-with-fold-in.jsonl" "$(echo "${dir_path}"/{train.csv,validation_tr.csv})"
shuf_dataset "${out_dir}/shuf-train-with-fold-in.jsonl" "${out_dir}/train-with-fold-in.jsonl"
build_user_seen \
  "${out_dir}/train-with-fold-in-user-seen-items.jsonl" \
  "$(echo "${dir_path}"/{train.csv,validation_tr.csv})"
build_grouped "${out_dir}/train-with-fold-in-grouped.jsonl" "$(echo "${dir_path}"/{train.csv,validation_tr.csv})"

# Fold-in Test
build_jsonl "${out_dir}/test-fold-in.jsonl" "${dir_path}/test_tr.csv"
shuf_dataset "${out_dir}/shuf-test-fold-in.jsonl" "${out_dir}/test-fold-in.jsonl"
build_user_seen \
  "${out_dir}/test-fold-in-user-seen-items.jsonl" \
  "$(echo "${dir_path}"/{train.csv,validation_tr.csv,validation_te.csv,test_tr.csv})"

# Test
build_grouped "${out_dir}/test-grouped.jsonl" "${dir_path}/test_te.csv"
build_user_seen \
  "${out_dir}/test-user-seen-items.jsonl" \
  "$(echo "${dir_path}"/{train.csv,validation_tr.csv,validation_te.csv,test_tr.csv,test_te.csv})"

# Full Train
build_jsonl "${out_dir}/full-train.jsonl" "$(echo "${dir_path}"/{train.csv,validation_tr.csv,validation_te.csv})"
shuf_dataset "${out_dir}/shuf-full-train.jsonl" "${out_dir}/full-train.jsonl"
build_user_seen \
  "${out_dir}/full-train-user-seen-items.jsonl" \
  "$(echo "${dir_path}"/{train.csv,validation_tr.csv,validation_te.csv})"
build_grouped \
    "${out_dir}/full-train-grouped.jsonl" \
    "$(echo "${dir_path}"/{train.csv,validation_tr.csv,validation_te.csv})"

# Full Train with fold-in
build_jsonl \
  "${out_dir}/full-train-with-fold-in.jsonl" \
  "$(echo "${dir_path}"/{train.csv,validation_tr.csv,validation_te.csv,test_tr.csv})"
shuf_dataset "${out_dir}/shuf-full-train-with-fold-in.jsonl" "${out_dir}/full-train-with-fold-in.jsonl"
build_user_seen \
  "${out_dir}/full-train-with-fold-in-user-seen-items.jsonl" \
  "$(echo "${dir_path}"/{train.csv,validation_tr.csv,validation_te.csv,test_tr.csv})"
build_grouped \
    "${out_dir}/full-train-with-fold-in-grouped.jsonl" \
    "$(echo "${dir_path}"/{train.csv,validation_tr.csv,validation_te.csv,test_tr.csv})"
