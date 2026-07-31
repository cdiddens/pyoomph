#!/usr/bin/env bash
#
# Nightly regression run for the develop branch, meant to be started from cron on a
# dedicated test machine:
#
#     0 1 * * *  /path/to/pyoomph/citools/nightly_develop.sh
#
# It fast-forwards an existing checkout to origin/develop, rebuilds, runs the full pytest
# suite and the full tutorial pipeline, and mails the result to MAIL_TO. If origin/develop
# has not moved since the last run it does nothing at all and sends no mail, so a quiet
# inbox means "nobody pushed", not "the nightly died" -- check $LOG_DIR/nightly.log if in
# doubt, every wake-up leaves a line there.
#
# Configuration lives outside the repository, in ~/.pyoomph_nightly.conf (override with
# PYOOMPH_NIGHTLY_CONF). See nightly_develop.conf.example; it holds the SMTP password, so
# chmod 600 it.
#
#   --force      run even if origin/develop has not moved (for testing the setup)
#   --test-mail  send a one-line mail and exit, to prove the mail path works
#
# Two things are deliberate and easy to get wrong when editing this:
#
#  * The script re-execs itself from a copy in /tmp before touching git. Bash reads a
#    script incrementally while running it, so pulling a new version of this very file out
#    from under the running shell makes it resume at a byte offset in different text.
#
#  * test_all_tutorial_scripts.py exits 0 even when scripts fail -- it only prints
#    "SOME TESTS FAILED". Its output is parsed rather than its return code; do not
#    "simplify" that away.

set -uo pipefail

# --------------------------------------------------------------------- re-exec ---

SELF="$(readlink -f "$0")"
if [ "${PYOOMPH_NIGHTLY_REEXEC:-}" != "1" ]; then
    copy="$(mktemp "${TMPDIR:-/tmp}/pyoomph_nightly.XXXXXXXX.sh")" || exit 1
    cat "$SELF" >"$copy" && chmod +x "$copy" || exit 1
    export PYOOMPH_NIGHTLY_REEXEC=1 PYOOMPH_NIGHTLY_SELF_COPY="$copy"
    exec "$copy" "$@"
fi

cleanup() { [ -n "${PYOOMPH_NIGHTLY_SELF_COPY:-}" ] && rm -f "$PYOOMPH_NIGHTLY_SELF_COPY"; }
trap cleanup EXIT

# ---------------------------------------------------------------- configuration ---

REPO_DIR="$HOME/code/pyoomph"
LOG_DIR="$HOME/pyoomph_nightly_logs"
KEEP_RUNS=30
ENV_SETUP=""
PYTHON="python3"
PETSC_PYTHONPATH=""
TIMEOUT_BUILD=7200
TIMEOUT_PYTEST=7200
TIMEOUT_TUTORIALS=28800
MAIL_TO="c.diddens@utwente.nl"
MAIL_FROM="pyoomph-nightly@$(hostname -f 2>/dev/null || hostname)"
SMTP_HOST=""
SMTP_PORT=587
SMTP_USER=""
SMTP_PASSWORD=""
SMTP_SECURITY="starttls"
MAIL_LOG_TAIL=60
BRANCH="develop"

CONFIG="${PYOOMPH_NIGHTLY_CONF:-$HOME/.pyoomph_nightly.conf}"
if [ -f "$CONFIG" ]; then
    # shellcheck disable=SC1090
    . "$CONFIG" || { echo "cannot read $CONFIG" >&2; exit 1; }
fi

FORCE=0
TEST_MAIL=0
for arg in "$@"; do
    case "$arg" in
        --force) FORCE=1 ;;
        --test-mail) TEST_MAIL=1 ;;
        *) echo "unknown option: $arg" >&2; exit 2 ;;
    esac
done

mkdir -p "$LOG_DIR" || exit 1
NIGHTLY_LOG="$LOG_DIR/nightly.log"
STATE_FILE="$LOG_DIR/last_tested.sha"

note() { printf '%s %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*" >>"$NIGHTLY_LOG"; }

# ------------------------------------------------------------------------ mail ---

# smtplib if SMTP_HOST is configured, otherwise the local mail/sendmail command. Most
# workstations have no working MTA, hence the SMTP path; but a machine that does have one
# should not need credentials in a config file, hence the fallback.
send_mail() {
    local subject="$1" bodyfile="$2" rc

    if [ -n "$SMTP_HOST" ]; then
        MAIL_SUBJECT="$subject" MAIL_BODY_FILE="$bodyfile" \
        MAIL_TO="$MAIL_TO" MAIL_FROM="$MAIL_FROM" \
        SMTP_HOST="$SMTP_HOST" SMTP_PORT="$SMTP_PORT" SMTP_USER="$SMTP_USER" \
        SMTP_PASSWORD="$SMTP_PASSWORD" SMTP_SECURITY="$SMTP_SECURITY" \
        "$PYTHON" - <<'PYEOF'
import os, smtplib, sys
from email.message import EmailMessage

msg = EmailMessage()
msg["Subject"] = os.environ["MAIL_SUBJECT"]
msg["From"] = os.environ["MAIL_FROM"]
msg["To"] = os.environ["MAIL_TO"]
with open(os.environ["MAIL_BODY_FILE"], "r", encoding="utf-8", errors="replace") as f:
    msg.set_content(f.read())

host = os.environ["SMTP_HOST"]
port = int(os.environ.get("SMTP_PORT") or 0)
security = (os.environ.get("SMTP_SECURITY") or "starttls").lower()
try:
    if security == "ssl":
        server = smtplib.SMTP_SSL(host, port or 465, timeout=120)
    else:
        server = smtplib.SMTP(host, port or 25, timeout=120)
    with server:
        if security == "starttls":
            server.starttls()
        if os.environ.get("SMTP_USER"):
            server.login(os.environ["SMTP_USER"], os.environ.get("SMTP_PASSWORD", ""))
        server.send_message(msg)
except Exception as e:
    print("smtplib failed: %s" % e, file=sys.stderr)
    sys.exit(1)
PYEOF
        rc=$?
        if [ $rc -eq 0 ]; then
            note "mail sent via smtplib: $subject"
            return 0
        fi
        note "smtplib failed, trying local mailer"
    fi

    if command -v mail >/dev/null 2>&1; then
        if mail -s "$subject" "$MAIL_TO" <"$bodyfile"; then
            note "mail sent via mail(1): $subject"
            return 0
        fi
    fi

    local sendmail=""
    if command -v sendmail >/dev/null 2>&1; then sendmail="sendmail"
    elif [ -x /usr/sbin/sendmail ]; then sendmail="/usr/sbin/sendmail"; fi
    if [ -n "$sendmail" ]; then
        if { printf 'To: %s\nFrom: %s\nSubject: %s\n\n' "$MAIL_TO" "$MAIL_FROM" "$subject"
             cat "$bodyfile"; } | "$sendmail" -t; then
            note "mail sent via sendmail: $subject"
            return 0
        fi
    fi

    note "NO WORKING MAILER -- report kept at $bodyfile"
    return 1
}

if [ "$TEST_MAIL" = 1 ]; then
    t="$(mktemp)"
    printf 'pyoomph nightly test mail from %s at %s.\n' "$(hostname)" "$(date)" >"$t"
    send_mail "[pyoomph nightly] test mail" "$t"; rc=$?
    rm -f "$t"
    exit $rc
fi

# ------------------------------------------------------------------ single run ---

exec 9>"$LOG_DIR/.lock"
if ! flock -n 9; then
    note "another nightly run is still going, skipping"
    exit 0
fi

# ------------------------------------------------------------- is there anything --

if [ ! -d "$REPO_DIR/.git" ]; then
    note "REPO_DIR=$REPO_DIR is not a git checkout -- fix $CONFIG"
    exit 1
fi

git -C "$REPO_DIR" fetch --prune origin >>"$NIGHTLY_LOG" 2>&1 || {
    note "git fetch failed"; exit 1; }

REMOTE_SHA="$(git -C "$REPO_DIR" rev-parse --verify --quiet "origin/$BRANCH")"
if [ -z "$REMOTE_SHA" ]; then
    # The branch does not exist yet. Nothing to test and nothing worth mailing about
    # every single night -- the trace in nightly.log is enough.
    note "origin/$BRANCH does not exist yet, nothing to do"
    exit 0
fi

LAST_SHA=""
[ -f "$STATE_FILE" ] && LAST_SHA="$(cat "$STATE_FILE")"

if [ "$REMOTE_SHA" = "$LAST_SHA" ] && [ "$FORCE" = 0 ]; then
    note "origin/$BRANCH unchanged at ${REMOTE_SHA:0:12}, nothing to do"
    exit 0
fi

RUN_DIR="$LOG_DIR/$(date '+%Y%m%d-%H%M%S')"
mkdir -p "$RUN_DIR" || exit 1
note "testing origin/$BRANCH ${REMOTE_SHA:0:12} (logs in $RUN_DIR)"

REPORT="$RUN_DIR/report.txt"
: >"$REPORT"
say() { printf '%s\n' "$*" >>"$REPORT"; }

quote_log() { # file, header
    local f="$1" hdr="$2"
    [ -f "$f" ] || return 0
    say ""
    say "--- $hdr (last $MAIL_LOG_TAIL lines of $(basename "$f")) ---"
    tail -n "$MAIL_LOG_TAIL" "$f" >>"$REPORT"
    say "--- end $hdr ---"
}

FAILED_STEPS=()

# ------------------------------------------------------------------- update git ---

DIRTY="$(git -C "$REPO_DIR" status --porcelain --untracked-files=no)"
if [ -n "$DIRTY" ]; then
    say "The nightly checkout $REPO_DIR has uncommitted changes, so it was not updated"
    say "and nothing was tested. Clean it up:"
    say ""
    printf '%s\n' "$DIRTY" >>"$REPORT"
    send_mail "[pyoomph nightly] ABORTED: $(hostname) checkout is dirty" "$REPORT"
    exit 1
fi

PREV_SHA="$(git -C "$REPO_DIR" rev-parse --verify --quiet HEAD)"

{
    if git -C "$REPO_DIR" show-ref --verify --quiet "refs/heads/$BRANCH"; then
        git -C "$REPO_DIR" checkout "$BRANCH"
    else
        git -C "$REPO_DIR" checkout -b "$BRANCH" --track "origin/$BRANCH"
    fi && git -C "$REPO_DIR" merge --ff-only "origin/$BRANCH"
} >"$RUN_DIR/git.log" 2>&1
if [ $? -ne 0 ]; then
    say "Could not fast-forward $REPO_DIR to origin/$BRANCH (${REMOTE_SHA:0:12})."
    say "Most likely the local $BRANCH branch has commits of its own, or something else"
    say "is checked out with local work on it. Nothing was built or tested."
    quote_log "$RUN_DIR/git.log" "git"
    send_mail "[pyoomph nightly] ABORTED: cannot update to origin/$BRANCH" "$REPORT"
    exit 1
fi

# From here on the commit counts as tested, whatever the outcome: a commit that breaks the
# build would otherwise be retried -- and mailed about -- every night until someone pushes
# a fix. The two aborts above deliberately happen before this, because a dirty or diverged
# checkout is a problem with the test machine, not a verdict on the commit, and it should
# keep nagging until it is cleaned up.
printf '%s\n' "$REMOTE_SHA" >"$STATE_FILE"

NEW_COMMITS=""
if [ -n "$LAST_SHA" ] && git -C "$REPO_DIR" cat-file -e "$LAST_SHA^{commit}" 2>/dev/null; then
    NEW_COMMITS="$(git -C "$REPO_DIR" log --oneline --no-decorate "$LAST_SHA..$REMOTE_SHA")"
else
    NEW_COMMITS="$(git -C "$REPO_DIR" log --oneline --no-decorate -5 "$REMOTE_SHA")"
fi

# ------------------------------------------------------------------ environment ---

if [ -n "$ENV_SETUP" ]; then
    # Either a file to source or the commands themselves -- whichever the config used.
    if [ -f "$ENV_SETUP" ]; then
        # shellcheck disable=SC1090
        . "$ENV_SETUP" >>"$RUN_DIR/env.log" 2>&1 || note "sourcing ENV_SETUP=$ENV_SETUP failed"
    else
        eval "$ENV_SETUP" >>"$RUN_DIR/env.log" 2>&1 || note "ENV_SETUP failed"
    fi
fi
if [ -n "$PETSC_PYTHONPATH" ]; then
    export PYTHONPATH="$PETSC_PYTHONPATH${PYTHONPATH:+:$PYTHONPATH}"
fi
# Keep matplotlib and friends from looking for an X server under cron.
export MPLBACKEND="${MPLBACKEND:-Agg}"

# ------------------------------------------------------------------- run a step ---

STEP_LINES=()
run_step() { # name, logfile, timeout, shell command -> sets STEP_RC
    local name="$1" log="$2" tmo="$3" cmd="$4" start rc secs
    start=$SECONDS
    timeout --signal=TERM --kill-after=120 "$tmo" bash -c "$cmd" >"$log" 2>&1
    rc=$?
    secs=$((SECONDS - start))
    if [ $rc -eq 124 ] || [ $rc -eq 137 ]; then
        note "$name TIMED OUT after ${tmo}s"
    fi
    STEP_RC=$rc
    STEP_SECS=$secs
    return $rc
}

hms() { printf '%dh%02dm%02ds' $(($1 / 3600)) $((($1 % 3600) / 60)) $(($1 % 60)); }

# ---------------------------------------------------------------------- build ---

run_step build "$RUN_DIR/build.log" "$TIMEOUT_BUILD" \
    "cd $(printf '%q' "$REPO_DIR") && ./build_for_develop.sh"
BUILD_RC=$STEP_RC
BUILD_SECS=$STEP_SECS
if [ "$BUILD_RC" -ne 0 ]; then
    FAILED_STEPS+=("build")
    STEP_LINES+=("build           FAILED (rc=$BUILD_RC)   $(hms "$BUILD_SECS")")
else
    STEP_LINES+=("build           ok                  $(hms "$BUILD_SECS")")
fi

# --------------------------------------------------------------------- pytest ---
#
# Only if the build succeeded: testing the previous build would produce a mail full of
# failures that have nothing to do with the commit under test.

PYTEST_RC=0
PYTEST_SECS=0
PYTEST_FAILURES=""
PYTEST_SUMMARY=""
if [ "$BUILD_RC" -eq 0 ]; then
    # The documented invocation is `python -m pytest *.py --full` from inside tests/
    # (see tests/README.md); --full is what adds the slow 3D campaign and the MPI modules.
    run_step pytest "$RUN_DIR/pytest.log" "$TIMEOUT_PYTEST" \
        "cd $(printf '%q' "$REPO_DIR")/tests && $(printf '%q' "$PYTHON") -m pytest *.py --full -rfE"
    PYTEST_RC=$STEP_RC
    PYTEST_SECS=$STEP_SECS
    PYTEST_FAILURES="$(grep -E '^(FAILED|ERROR) ' "$RUN_DIR/pytest.log" 2>/dev/null)"
    PYTEST_SUMMARY="$(grep -E '^=+.*(passed|failed|error|no tests ran).*=+$' "$RUN_DIR/pytest.log" 2>/dev/null | tail -n 1)"
    if [ "$PYTEST_RC" -ne 0 ]; then
        FAILED_STEPS+=("pytest")
        STEP_LINES+=("pytest --full   FAILED (rc=$PYTEST_RC)   $(hms "$PYTEST_SECS")")
    else
        STEP_LINES+=("pytest --full   ok                  $(hms "$PYTEST_SECS")")
    fi
else
    STEP_LINES+=("pytest --full   skipped (build failed)")
fi

# ------------------------------------------------------------------ tutorials ---

TUT_RC=0
TUT_SECS=0
TUT_FAILURES=""
TUT_BAD=0
if [ "$BUILD_RC" -eq 0 ]; then
    run_step tutorials "$RUN_DIR/tutorials.log" "$TIMEOUT_TUTORIALS" \
        "cd $(printf '%q' "$REPO_DIR") && $(printf '%q' "$PYTHON") -u citools/test_all_tutorial_scripts.py"
    TUT_RC=$STEP_RC
    TUT_SECS=$STEP_SECS

    # The runner always exits 0; the verdict is in its output.
    TUT_FAILURES="$(grep -E '=+ FAILED ' "$RUN_DIR/tutorials.log" 2>/dev/null)"
    [ "$TUT_RC" -ne 0 ] && TUT_BAD=1
    [ -n "$TUT_FAILURES" ] && TUT_BAD=1
    grep -q '^SOME TESTS FAILED' "$RUN_DIR/tutorials.log" 2>/dev/null && TUT_BAD=1
    grep -q '^ALL TESTS PASSED' "$RUN_DIR/tutorials.log" 2>/dev/null || TUT_BAD=1

    # The next run wipes citools/pyoomph_tutorial_scripts/, so the per-script logs of the
    # failures have to be copied out now.
    TUT_LOGDIR="$RUN_DIR/tutorial_logs"
    if [ -d "$REPO_DIR/citools/pyoomph_tutorial_scripts" ]; then
        mkdir -p "$TUT_LOGDIR"
        (cd "$REPO_DIR/citools/pyoomph_tutorial_scripts" && find . -name '*.log' -print0 |
            while IFS= read -r -d '' f; do
                cp "$f" "$TUT_LOGDIR/$(printf '%s' "${f#./}" | tr '/' '_')" 2>/dev/null
            done)
        rmdir "$TUT_LOGDIR" 2>/dev/null
    fi

    if [ "$TUT_BAD" -ne 0 ]; then
        FAILED_STEPS+=("tutorials")
        STEP_LINES+=("tutorials       FAILED              $(hms "$TUT_SECS")")
    else
        STEP_LINES+=("tutorials       ok                  $(hms "$TUT_SECS")")
    fi
else
    STEP_LINES+=("tutorials       skipped (build failed)")
fi

# --------------------------------------------------------------------- report ---

if [ ${#FAILED_STEPS[@]} -eq 0 ]; then
    VERDICT="PASS"
    SUBJECT="[pyoomph nightly] $BRANCH ${REMOTE_SHA:0:12} PASS"
else
    VERDICT="FAIL"
    SUBJECT="[pyoomph nightly] $BRANCH ${REMOTE_SHA:0:12} FAILED: $(IFS=', '; echo "${FAILED_STEPS[*]}")"
fi

say "pyoomph nightly on $(hostname) -- $VERDICT"
say ""
say "branch    : $BRANCH"
say "commit    : $REMOTE_SHA"
say "previously: ${LAST_SHA:-(no previous run)}"
say "checkout  : $REPO_DIR (was at ${PREV_SHA:0:12} before the run)"
say "logs      : $RUN_DIR"
say "started   : $(date -d "@$(( $(date +%s) - SECONDS ))" '+%Y-%m-%d %H:%M:%S' 2>/dev/null)"
say "duration  : $(hms "$SECONDS")"
say ""
say "Steps"
say "-----"
for line in "${STEP_LINES[@]}"; do say "  $line"; done
say ""
say "New commits"
say "-----------"
printf '%s\n' "${NEW_COMMITS:-(none)}" >>"$REPORT"

if [ "$VERDICT" = "PASS" ]; then
    say ""
    [ -n "$PYTEST_SUMMARY" ] && say "pytest: $PYTEST_SUMMARY"
    say "Tutorial pipeline: all scripts ran. (preCICE runs still need a manual check.)"
else
    if [ "$BUILD_RC" -ne 0 ]; then
        say ""
        say "BUILD FAILED"
        say "============"
        quote_log "$RUN_DIR/build.log" "build"
    fi

    if [ "$PYTEST_RC" -ne 0 ]; then
        say ""
        say "PYTEST FAILED"
        say "============="
        [ -n "$PYTEST_SUMMARY" ] && say "$PYTEST_SUMMARY"
        say ""
        if [ -n "$PYTEST_FAILURES" ]; then
            printf '%s\n' "$PYTEST_FAILURES" >>"$REPORT"
        else
            say "(no FAILED/ERROR lines -- pytest itself did not get that far)"
            quote_log "$RUN_DIR/pytest.log" "pytest"
        fi
    fi

    if [ "$TUT_BAD" -ne 0 ]; then
        say ""
        say "TUTORIAL PIPELINE FAILED"
        say "========================"
        if [ -n "$TUT_FAILURES" ]; then
            printf '%s\n' "$TUT_FAILURES" >>"$REPORT"
        else
            say "(the runner reported a failure without naming a script -- see tutorials.log)"
            quote_log "$RUN_DIR/tutorials.log" "tutorials"
        fi
        # One tail per failing script, capped: a broken core commit fails all 127 of them
        # and the mail would be unreadable. The rest are on disk in $RUN_DIR.
        if [ -d "$RUN_DIR/tutorial_logs" ]; then
            n=0
            for f in "$RUN_DIR/tutorial_logs"/*.log; do
                [ -f "$f" ] || continue
                n=$((n + 1))
                if [ "$n" -gt 10 ]; then
                    say ""
                    say "(further failing-script logs omitted -- all of them are in $RUN_DIR/tutorial_logs)"
                    break
                fi
                quote_log "$f" "tutorial $(basename "$f" .log)"
            done
        fi
    fi
fi

say ""
say "-- citools/nightly_develop.sh"

send_mail "$SUBJECT" "$REPORT"
note "run finished: $VERDICT ($(hms "$SECONDS"))"

# ------------------------------------------------------------------- housekeeping --

ls -1dt "$LOG_DIR"/20* 2>/dev/null | tail -n "+$((KEEP_RUNS + 1))" | while IFS= read -r old; do
    rm -rf "$old"
done

[ "$VERDICT" = "PASS" ] && exit 0
exit 1
