#!/bin/sh
# test.sh
# Summary: Validation suite for ipa schema compiler and runtime.
# Author:  KaisarCode
# Website: https://kaisarcode.com
# License: https://www.gnu.org/licenses/gpl-3.0.html

# Prints one failure line.
# @param $1 Failure message.
# @return 1 on failure.
kc_test_fail() {
    printf '\033[31m[FAIL]\033[0m %s\n' "$1"
    return 1
}

# Prints one success line.
# @param $1 Success message.
# @return 0 on success.
kc_test_pass() {
    printf '\033[32m[PASS]\033[0m %s\n' "$1"
}

# Detects the artifact architecture for the current machine.
# @return Architecture name on stdout.
kc_test_arch() {
    case "$(uname -m)" in
        x86_64 | amd64)            printf '%s\n' "x86_64" ;;
        aarch64 | arm64)           printf '%s\n' "aarch64" ;;
        armv7l | armv7)            printf '%s\n' "armv7" ;;
        i386 | i486 | i586 | i686) printf '%s\n' "i686" ;;
        ppc64le | powerpc64le)     printf '%s\n' "powerpc64le" ;;
        *)                         uname -m ;;
    esac
}

# Detects the artifact platform for the current machine.
# @return Platform name on stdout.
kc_test_platform() {
    case "$(uname -s)" in
        Linux) printf '%s\n' "linux" ;;
        *)     uname -s | tr '[:upper:]' '[:lower:]' ;;
    esac
}

# Returns the CLI path for the current architecture and platform.
# @return CLI path on stdout.
kc_test_binary_path() {
    printf './bin/%s/%s/ipa\n' "$(kc_test_arch)" "$(kc_test_platform)"
}

# Verifies the binary exists and is executable.
# @return 0 on success, 1 on failure.
kc_test_check_binary() {
    if [ ! -x "$BIN" ]; then
        kc_test_fail "binary not found: $BIN"
        return 1
    fi
    return 0
}

# Builds a .ipa from a schema.json, expecting success.
# @param $1 Test description.
# @param $2 Path to schema.json.
# @return 0 on success, 1 on failure.
kc_test_build_ok() {
    desc="$1"
    json="$2"
    if ! "$BIN" "$json" -b > /dev/null 2>&1; then
        kc_test_fail "$desc"
        return 1
    fi
    kc_test_pass "$desc"
}

# Runs build mode, expecting a non-zero exit code.
# @param $1 Test description.
# @param $2 Path to schema.json.
# @return 0 on success, 1 on failure.
kc_test_build_fail() {
    desc="$1"
    json="$2"
    if "$BIN" "$json" -b > /dev/null 2>&1; then
        kc_test_fail "$desc (expected failure)"
        return 1
    fi
    kc_test_pass "$desc"
}

# Queries the schema with argv text and checks node id in output.
# @param $1 Test description.
# @param $2 Path to .ipa file.
# @param $3 Input text.
# @param $4 Expected node id string.
# @return 0 on success, 1 on failure.
kc_test_match() {
    desc="$1"
    ipa="$2"
    input="$3"
    expected_id="$4"
    out=$("$BIN" "$ipa" "$input" 2>/dev/null) || true
    if printf '%s' "$out" | grep -q "\"id\":\"${expected_id}\""; then
        kc_test_pass "$desc → id=$expected_id"
        return 0
    fi
    kc_test_fail "$desc → expected id=$expected_id in: $out"
    return 1
}

# Queries the schema via stdin and checks node id in output.
# @param $1 Test description.
# @param $2 Path to .ipa file.
# @param $3 Input text.
# @param $4 Expected node id string.
# @return 0 on success, 1 on failure.
kc_test_match_stdin() {
    desc="$1"
    ipa="$2"
    input="$3"
    expected_id="$4"
    out=$(printf '%s' "$input" | "$BIN" "$ipa" 2>/dev/null) || true
    if printf '%s' "$out" | grep -q "\"id\":\"${expected_id}\""; then
        kc_test_pass "$desc → id=$expected_id"
        return 0
    fi
    kc_test_fail "$desc → expected id=$expected_id in: $out"
    return 1
}

# Queries the schema and checks that ok is false (threshold rejection).
# @param $1 Test description.
# @param $2 Path to .ipa file.
# @param $3 Input text.
# @return 0 on success, 1 on failure.
kc_test_no_match() {
    desc="$1"
    ipa="$2"
    input="$3"
    out=$("$BIN" "$ipa" "$input" 2>/dev/null) || true
    if printf '%s' "$out" | grep -q '"ok":false'; then
        kc_test_pass "$desc → ok=false"
        return 0
    fi
    kc_test_fail "$desc → expected ok=false in: $out"
    return 1
}

# Queries the schema and checks that an emit key appears in the output.
# @param $1 Test description.
# @param $2 Path to .ipa file.
# @param $3 Input text.
# @param $4 Expected emit key string.
# @return 0 on success, 1 on failure.
kc_test_emit() {
    desc="$1"
    ipa="$2"
    input="$3"
    expected_key="$4"
    out=$("$BIN" "$ipa" "$input" 2>/dev/null) || true
    if printf '%s' "$out" | grep -q "\"${expected_key}\""; then
        kc_test_pass "$desc → emit key=$expected_key"
        return 0
    fi
    kc_test_fail "$desc → expected emit key=$expected_key in: $out"
    return 1
}

# Queries the schema and checks that a child match appears in the output.
# @param $1 Test description.
# @param $2 Path to .ipa file.
# @param $3 Input text.
# @param $4 Expected child node id string.
# @return 0 on success, 1 on failure.
kc_test_child_match() {
    desc="$1"
    ipa="$2"
    input="$3"
    expected_child_id="$4"
    out=$("$BIN" "$ipa" "$input" 2>/dev/null) || true
    if printf '%s' "$out" | grep -q "\"children\":" && printf '%s' "$out" | grep -q "\"id\":\"${expected_child_id}\""; then
        kc_test_pass "$desc → child id=$expected_child_id"
        return 0
    fi
    kc_test_fail "$desc → expected child id=$expected_child_id in: $out"
    return 1
}

# Verifies that kc_ipa_open rejects a malformed .ipa file.
# @param $1 Test description.
# @return 0 on success, 1 on failure.
kc_test_malformed_ipa() {
    desc="$1"
    bad="$TMPDIR/bad.ipa"
    printf 'JUNK' > "$bad"
    if "$BIN" "$bad" "hello" > /dev/null 2>&1; then
        rm -f "$bad"
        kc_test_fail "$desc (expected failure)"
        return 1
    fi
    rm -f "$bad"
    kc_test_pass "$desc"
}

# Writes the test schema.json to a temp file.
# @param $1 Destination path.
# @return 0 on success.
kc_test_write_schema() {
    cat > "$1" << 'SCHEOF'
{
    "schema": "ipa.map.v1",
    "id": "test",
    "vendor": "test",
    "version": "0.1.0",
    "defaults": {
        "node_threshold": 0.80,
        "min_ngram": 1,
        "max_ngram": 5
    },
    "nodes": [
        {
            "id": "install_command",
            "descriptions": [
                "install firefox",
                "install chrome",
                "setup software",
                "add application"
            ],
            "emit": {
                "action": "install"
            },
            "children": [
                {
                    "id": "software",
                    "descriptions": [
                        "firefox",
                        "chrome",
                        "brave",
                        "vlc"
                    ],
                    "emit": {
                        "software": "$raw"
                    }
                }
            ]
        },
        {
            "id": "greeting",
            "descriptions": [
                "hello",
                "hi there",
                "good morning",
                "hey"
            ],
            "emit": {
                "type": "greeting"
            }
        }
    ]
}
SCHEOF
}

# Writes an invalid schema.json missing the required schema field.
# @param $1 Destination path.
# @return 0 on success.
kc_test_write_invalid_schema() {
    cat > "$1" << 'SCHEOF'
{
    "id": "bad",
    "vendor": "test",
    "version": "0.1.0",
    "nodes": []
}
SCHEOF
}

# Runs the full validation suite.
# @return 0 on success, non-zero on any failure.
kc_test_main() {
    failed=0

    BIN=$(kc_test_binary_path)
    TMPDIR=$(mktemp -d)
    SCHEMA_JSON="${TMPDIR}/schema.json"
    SCHEMA_IPA="${TMPDIR}/schema.ipa"
    BAD_JSON="${TMPDIR}/bad.json"

    kc_test_check_binary || exit 1

    kc_test_write_schema "$SCHEMA_JSON"
    kc_test_write_invalid_schema "$BAD_JSON"

    kc_test_build_ok   "build valid schema.json"      "$SCHEMA_JSON"  || failed=1
    kc_test_build_fail "reject invalid schema"        "$BAD_JSON"     || failed=1

    kc_test_match       "argv parse node match"       "$SCHEMA_IPA" "install firefox"    "install_command" || failed=1
    kc_test_match_stdin "stdin parse node match"      "$SCHEMA_IPA" "install chrome"     "install_command" || failed=1
    kc_test_match       "emb parse node match"        "$SCHEMA_IPA" "add an application" "install_command" || failed=1
    kc_test_match       "greeting node match"         "$SCHEMA_IPA" "hello"              "greeting"        || failed=1
    kc_test_emit        "emit key present"            "$SCHEMA_IPA" "install firefox"    "action"          || failed=1
    kc_test_child_match "child refinement match"      "$SCHEMA_IPA" "install firefox"    "software"        || failed=1
    kc_test_no_match    "threshold rejection"         "$SCHEMA_IPA" "zzz xyz qwerty"                       || failed=1
    kc_test_malformed_ipa "malformed .ipa rejected"                                                        || failed=1

    rm -rf "$TMPDIR"
    return $failed
}

kc_test_main
