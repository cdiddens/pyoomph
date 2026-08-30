#!/usr/bin/env bash
#
# Nightly regression run for the develop branch, meant to be started from cron on a
# dedicated test machine:
#
#     0 1 * * *  /path/to/pyoomph/citools/nightly_develop.sh
#
# It fast-forwards an existing checkout to origin/develop, rebuilds, runs the full pytest
# suite, builds the documentation in a virtualenv of its own, runs the tutorial pipeline twice
# -- serially and under mpirun -- and mails the result
# to MAIL_TO. If origin/develop has not moved since the last run it does nothing at all and
# sends no mail, so a quiet inbox means "nobody pushed", not "the nightly died" -- check
# $LOG_DIR/nightly.log if in doubt, every wake-up leaves a line there.
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
    export PYOOMPH_NIGHTLY_REEXEC=1 PYOOMPH_NIGHTLY_SELF_COPY="$copy" PYOOMPH_NIGHTLY_ORIGIN="$SELF"
    exec "$copy" "$@"
fi

cleanup() { [ -n "${PYOOMPH_NIGHTLY_SELF_COPY:-}" ] && rm -f "$PYOOMPH_NIGHTLY_SELF_COPY"; }
trap cleanup EXIT

# ---------------------------------------------------------------- configuration ---

REPO_DIR="$HOME/code/pyoomph"
LOG_DIR="$HOME/pyoomph_nightly_logs"
KEEP_RUNS=30
NIGHTLY_TMPDIR=""   # scratch for the run; empty means $LOG_DIR/tmp (resolved after the config)
ENV_SETUP=""
PYTHON="python3"
TIMEOUT_BUILD=7200
TIMEOUT_PYTEST=7200
TIMEOUT_DOCS=5400
TIMEOUT_TUTORIALS=28800
TIMEOUT_TUTORIALS_MPI=28800
DOCS_ENABLED=1
DOCS_PDF=1   # also run xelatex over the generated tutorial; skipped by itself without latexmk
# Sphinx warnings the docs step is not allowed to fail on, as an extended regular expression.
# Only preCICE by default: its Python bindings need the preCICE library, docs/requirements.txt
# deliberately does not ask for it, and Read the Docs does not have it either -- so that one
# warning is the state RTD builds in rather than something develop broke.
DOCS_IGNORE_WARNINGS='Failed to import pyoomph\.solvers\.precice_adapter|No module named .precice.|module .pyoomph\.solvers. has no attribute .precice_adapter.'
TUTORIAL_MPI_RANKS=4   # second tutorial pass under mpirun -n N; 0 switches it off
MAIL_TO=""   # empty means "do not send"; set it in the config file, not here
MAIL_FROM="pyoomph-nightly@$(hostname -f 2>/dev/null || hostname)"
SMTP_HOST=""
SMTP_PORT=587
SMTP_USER=""
SMTP_PASSWORD=""
SMTP_SECURITY="starttls"
SMTP_FORCE_IPV4=""
MAIL_LOG_TAIL=60
PYTEST_DETAIL_LINES=150
TIMING_TOP=10   # how many scripts the timing section of the mail names; 0 switches the section off
BRANCH="develop"

CONFIG="${PYOOMPH_NIGHTLY_CONF:-$HOME/.pyoomph_nightly.conf}"
if [ -f "$CONFIG" ]; then
    # shellcheck disable=SC1090
    . "$CONFIG" || { echo "cannot read $CONFIG" >&2; exit 1; }
    # Configs written before d2cf20c set this to point the run at a PETSc build. Nothing has read it
    # since, and a config that still has it looks configured while the run has no PETSc at all --
    # which is what left the MPI tests untested for months. Say so rather than ignoring it silently.
    if [ -n "${PETSC_PYTHONPATH:-}" ]; then
        echo "note: PETSC_PYTHONPATH in $CONFIG is obsolete and ignored; PETSC_DIR and" >&2
        echo "      PETSC_ARCH_REAL/PETSC_ARCH_COMPLEX (from ~/.bashrc) are used instead." >&2
    fi
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

# Under cron there is no terminal and the log is the only record. Run by hand, a silent
# script is indistinguishable from a broken one, so the same lines go to stderr.
note() {
    local msg
    msg="$(date '+%Y-%m-%d %H:%M:%S') $*"
    printf '%s\n' "$msg" >>"$NIGHTLY_LOG"
    [ -t 2 ] && printf '%s\n' "$msg" >&2
    return 0
}

# ------------------------------------------------------------------------ mail ---

# smtplib if SMTP_HOST is configured, otherwise the local mail/sendmail command. Most
# workstations have no working MTA, hence the SMTP path; but a machine that does have one
# should not need credentials in a config file, hence the fallback.
MAIL_TRANSPORT=""
send_mail() {
    local subject="$1" bodyfile="$2" rc out
    MAIL_TRANSPORT=""

    # An empty MAIL_TO turns mail off. Worth having: on a host that cannot send at all --
    # networks commonly refuse direct delivery from their workstation ranges as a matter of
    # policy -- the alternative is two doomed SMTP attempts every night followed by a report
    # handed to a local MTA that queues it forever. The reports are on disk either way, and
    # that is then where they are read from.
    if [ -z "$MAIL_TO" ]; then
        MAIL_TRANSPORT="disabled"
        note "mail disabled (MAIL_TO is empty) -- report kept at $bodyfile"
        return 0
    fi

    if [ -n "$SMTP_HOST" ]; then
        out="$(MAIL_SUBJECT="$subject" MAIL_BODY_FILE="$bodyfile" \
        MAIL_TO="$MAIL_TO" MAIL_FROM="$MAIL_FROM" \
        SMTP_HOST="$SMTP_HOST" SMTP_PORT="$SMTP_PORT" SMTP_USER="$SMTP_USER" \
        SMTP_PASSWORD="$SMTP_PASSWORD" SMTP_SECURITY="$SMTP_SECURITY" \
        SMTP_FORCE_IPV4="$SMTP_FORCE_IPV4" \
        "$PYTHON" - 2>&1 <<'PYEOF'
import os, smtplib, socket, sys
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

# Microsoft rejects IPv6 senders that have no reverse-DNS record ("4.7.25 ... must have
# reverse DNS record (S820)") while accepting the same machine over IPv4, and a host with
# a AAAA route will pick IPv6 by itself. Filtering getaddrinfo rather than connecting to a
# literal address keeps the hostname for EHLO and for the TLS handshake.
_getaddrinfo = socket.getaddrinfo

def _ipv4_only(*a, **kw):
    infos = [ai for ai in _getaddrinfo(*a, **kw) if ai[0] == socket.AF_INET]
    if not infos:
        raise OSError("%s has no IPv4 address" % (a[0],))
    return infos

def deliver():
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

forced = os.environ.get("SMTP_FORCE_IPV4", "") not in ("", "0", "false", "False")
errors = []
for ipv4_only in ([True] if forced else [False, True]):
    socket.getaddrinfo = _ipv4_only if ipv4_only else _getaddrinfo
    try:
        deliver()
    except Exception as e:
        errors.append(("IPv4-only" if ipv4_only else "default", e))
        continue
    if ipv4_only and not forced:
        print("note: the default route failed and IPv4 worked; set SMTP_FORCE_IPV4=1 in the "
              "config to stop retrying the failing one every time")
    sys.exit(0)

for label, e in errors:
    print("smtplib failed (%s): %s" % (label, e))
sys.exit(1)
PYEOF
)"
        rc=$?
        [ -n "$out" ] && note "$out"
        if [ $rc -eq 0 ]; then
            MAIL_TRANSPORT="smtplib"
            note "mail sent via smtplib to $SMTP_HOST: $subject"
            return 0
        fi
        note "smtplib failed, trying the local mailer"
    fi

    # Both fallbacks only prove that a local MTA ACCEPTED the message. On a workstation
    # with no relay configured that is where it stops -- accepted, queued, never delivered,
    # exit status 0. Hence the wording of the note, and the advice under --test-mail.
    if command -v mail >/dev/null 2>&1; then
        if out="$(mail -s "$subject" "$MAIL_TO" <"$bodyfile" 2>&1)"; then
            MAIL_TRANSPORT="mail"
            note "handed to mail(1) (accepted locally, delivery not confirmed): $subject"
            return 0
        fi
        [ -n "$out" ] && note "mail(1) failed: $out"
    fi

    local sendmail=""
    if command -v sendmail >/dev/null 2>&1; then sendmail="sendmail"
    elif [ -x /usr/sbin/sendmail ]; then sendmail="/usr/sbin/sendmail"; fi
    if [ -n "$sendmail" ]; then
        if { printf 'To: %s\nFrom: %s\nSubject: %s\n\n' "$MAIL_TO" "$MAIL_FROM" "$subject"
             cat "$bodyfile"; } | "$sendmail" -t; then
            MAIL_TRANSPORT="sendmail"
            note "handed to sendmail (accepted locally, delivery not confirmed): $subject"
            return 0
        fi
    fi

    note "NO WORKING MAILER for \"$subject\" -- report kept at $bodyfile"
    return 1
}

if [ "$TEST_MAIL" = 1 ]; then
    echo "pyoomph nightly -- mail check"
    echo
    if [ -f "$CONFIG" ]; then
        echo "  config    : $CONFIG"
    else
        echo "  config    : $CONFIG  *** DOES NOT EXIST -- built-in defaults are in use ***"
    fi
    if [ -z "$MAIL_TO" ]; then
        echo "  to        : (MAIL_TO is empty -- mail is switched off, reports stay in $LOG_DIR)"
    else
        echo "  to        : $MAIL_TO"
    fi
    # The rest only describes how mail would be sent, which is noise once it is off.
    if [ -n "$MAIL_TO" ]; then
        echo "  from      : $MAIL_FROM"
        # A sender whose domain does not resolve is rejected or silently dropped by most
        # relays, and the default here is built from `hostname`, which on a workstation is
        # usually a bare name with no domain at all.
        case "${MAIL_FROM#*@}" in
            *.*) ;;
            *) echo "              ^^ that domain has no dot and will not resolve. Most relays drop"
               echo "                 such a sender without a word. Set MAIL_FROM in the config to a"
               echo "                 real mailbox, e.g. $MAIL_TO" ;;
        esac
        if [ -n "$SMTP_HOST" ]; then
            echo "  smtp      : $SMTP_HOST:$SMTP_PORT ($SMTP_SECURITY), user=${SMTP_USER:-<none>}," \
                 "password $([ -n "$SMTP_PASSWORD" ] && echo set || echo 'NOT set')"
            # Submission servers generally insist the From match the account that
            # authenticated, and reject or silently rewrite it otherwise.
            case "$SMTP_USER" in
                "" | "$MAIL_FROM") ;;
                *@*) echo "              ^^ this does not match MAIL_FROM ($MAIL_FROM). Most submission"
                     echo "                 servers require the sender to be the account that logged in;"
                     echo "                 set MAIL_FROM to $SMTP_USER and MAIL_TO to where you read it." ;;
            esac
        else
            echo "  smtp      : SMTP_HOST is empty -- falling back to the local mail/sendmail command"
        fi
        echo "  mail(1)   : $(command -v mail || echo 'not installed')"
        echo "  sendmail  : $(command -v sendmail || { [ -x /usr/sbin/sendmail ] && echo /usr/sbin/sendmail; } || echo 'not installed')"
    fi
    echo
    t="$(mktemp)"
    printf 'pyoomph nightly test mail from %s at %s.\n' "$(hostname)" "$(date)" >"$t"
    send_mail "[pyoomph nightly] test mail" "$t"; rc=$?
    rm -f "$t"
    echo
    case "${MAIL_TRANSPORT:-none}" in
        disabled)
            echo "Mail is switched off. Each run still writes its full report to a dated directory"
            echo "under $LOG_DIR, and $LOG_DIR/nightly.log records"
            echo "every wake-up including the nights nothing was pushed."
            ;;
        smtplib)
            echo "Accepted by $SMTP_HOST, so the message did leave this machine."
            echo "If it does not arrive, look in the spam folder and at the mail server's logs."
            ;;
        mail|sendmail)
            echo "Handed to the local '$MAIL_TRANSPORT' command. That only means a local MTA accepted"
            echo "it -- NOT that it was delivered. A workstation with no relay configured queues or"
            echo "drops it silently and still exits 0, which is the usual reason no mail arrives."
            echo
            echo "    mailq                 # is it stuck in the queue?"
            echo "    tail /var/log/mail.log /var/log/maillog"
            echo
            echo "Setting SMTP_HOST in $CONFIG avoids the local MTA entirely and is the more reliable"
            echo "option on a machine that is not a mail server."
            ;;
        *)
            echo "No mail transport worked at all. Set SMTP_HOST (and SMTP_USER/SMTP_PASSWORD) in"
            echo "$CONFIG -- that path needs nothing installed beyond Python."
            ;;
    esac
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

# Reused across the hand-over to an updated version of this script (see "new version" below), so
# that the hand-over does not leave an orphaned log directory behind.
RUN_DIR="${PYOOMPH_NIGHTLY_RUN_DIR:-$LOG_DIR/$(date '+%Y%m%d-%H%M%S')}"
mkdir -p "$RUN_DIR" || exit 1
note "testing origin/$BRANCH ${REMOTE_SHA:0:12} (logs in $RUN_DIR)"

# Scratch of our own, next to the logs, rather than the machine's /tmp. On 2026-08-11 /tmp here was
# a 7.8 GB tmpfs that interactive work had filled to the last 52 kB before the run started; pytest
# then reported 395 failures that were all ENOSPC on tmp_path, and every one of the 128 tutorial
# scripts died in the mpirun pass with a SIGBUS out of pmix_shmem_segment_create -- what mmap does
# when tmpfs cannot back the page. Nothing about develop was broken and the report said nothing
# about disks. Cleared per run because pytest keeps its last three tmp_path trees itself, which is
# how ~1 GB of them had accumulated. Safe to clear here: we hold the lock, so no other run is using
# it. Only the small PMIx/ORTE session dirs follow TMPDIR -- vader's shared-memory segments live in
# /dev/shm regardless -- so putting this on disk does not slow the MPI passes down.
NIGHTLY_TMPDIR="${NIGHTLY_TMPDIR:-$LOG_DIR/tmp}"
if rm -rf "$NIGHTLY_TMPDIR" && mkdir -p "$NIGHTLY_TMPDIR"; then
    export TMPDIR="$NIGHTLY_TMPDIR" TMP="$NIGHTLY_TMPDIR" TEMP="$NIGHTLY_TMPDIR"
else
    note "cannot use NIGHTLY_TMPDIR=$NIGHTLY_TMPDIR -- falling back to ${TMPDIR:-/tmp}"
fi

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

# ---------------------------------------------------------------- new version ---
#
# What is running is the copy taken at the top, i.e. the version of this file from *before* the
# fast-forward. A commit that changes the nightly and the thing it drives at the same time would
# therefore be tested with the old nightly: on 2026-08-09 the tutorial runner had just started to
# require $PETSC_ARCH_REAL/$PETSC_ARCH_COMPLEX, the script that would have provided them was in the
# same push, and the tutorial step died in its first second. So hand over to the new version once.
#
# The state file is deliberately still unwritten here, so the new instance walks into exactly the
# situation this one did -- origin/$BRANCH ahead of last_tested.sha -- and simply carries on. Its
# `exec 9>` reopens the lock file, which drops this instance's flock along with the old descriptor
# before it takes its own, so the hand-over does not lock the run out of its own lock.
if [ "${PYOOMPH_NIGHTLY_UPDATED:-}" != "1" ] && [ -n "${PYOOMPH_NIGHTLY_ORIGIN:-}" ] &&
   [ -r "$PYOOMPH_NIGHTLY_ORIGIN" ] && ! cmp -s "$PYOOMPH_NIGHTLY_ORIGIN" "$SELF"; then
    note "the update changed $PYOOMPH_NIGHTLY_ORIGIN -- restarting with the new version"
    newcopy="$(mktemp "${TMPDIR:-/tmp}/pyoomph_nightly.XXXXXXXX.sh")" || exit 1
    if cat "$PYOOMPH_NIGHTLY_ORIGIN" >"$newcopy" && chmod +x "$newcopy"; then
        # exec does not run the EXIT trap, so this copy is unlinked by hand. Bash holds the file
        # open while it reads it, so unlinking the running script is safe.
        rm -f "$SELF"
        export PYOOMPH_NIGHTLY_UPDATED=1 PYOOMPH_NIGHTLY_SELF_COPY="$newcopy" PYOOMPH_NIGHTLY_RUN_DIR="$RUN_DIR"
        exec "$newcopy" "$@"
    fi
    rm -f "$newcopy"
    note "could not copy the new version -- carrying on with the old one"
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

# Everything the machine's own setup provides -- the compiler environment, mpirun, and above all the
# two PETSc builds -- comes from the same place the user's shell gets it: .bashrc. In particular
# test_all_tutorial_scripts.py wants PETSC_DIR plus PETSC_ARCH_REAL and PETSC_ARCH_COMPLEX -- it runs
# the ordinary tutorials against the real-scalar build and only the ones with complex spectra (the
# normal-mode stability and the periodic-orbit/Floquet scripts, which the runner picks out by itself)
# against the complex one -- and duplicating those paths in the nightly config was one more thing to
# forget when a PETSc is rebuilt under a new arch name.
#
# It is fetched from a genuinely interactive `bash -i` rather than by sourcing ~/.bashrc into this
# shell, because sourcing it does not work: rc files guard themselves against non-interactive use,
# and the stock Debian guard is
#     case $- in *i*) ;; *) return;; esac
# which tests the shell's interactive *flag*. That flag can only be set when bash starts, so the
# `PS1='nightly$ '` trick this script used until 2026-08-09 -- which does defeat the older
# `[ -z "$PS1" ] && return` idiom -- returned at line 9 of .bashrc and quietly delivered nothing.
# The nightly ran for months with no PETSc in its environment; it only became a failure when the
# tutorial runner started to require the two arches by name, and then failed in the first second.
#
# The probe shell inherits this environment and runs the rc file on top of it, so what it prints is
# exactly "what the nightly has, plus what .bashrc adds" -- including variables that are merely
# assigned rather than exported, since it prints the values rather than passing the environment on.
# Only the variables named here are taken over; a wholesale import would also carry PWD, SHLVL and
# friends. `bash -i` without a terminal complains about job control, and rc files are chatty: that
# goes to env.log. </dev/null and the timeout are in case an rc file wants to talk to a user who is
# not there (a `read`, an auto-attaching tmux).
#
# setsid is not decoration. An interactive bash that HAS a controlling terminal but is not the
# terminal's foreground process group stops itself on purpose -- initialize_job_control() sends
# itself SIGTTIN and waits to be brought to the foreground, which nobody is going to do here. Under
# cron that never happens, there is no terminal at all; started by hand from a shell it deadlocks on
# the spot, as it did on 2026-08-09. A session of its own means the probe cannot open /dev/tty, so
# it settles for "no job control in this shell" and gets on with it -- the same situation cron gives
# it. --kill-after for the same reason: a process stopped that way never handles the plain SIGTERM.
NIGHTLY_ENV_VARS=(PATH LD_LIBRARY_PATH PYTHONPATH PETSC_DIR PETSC_ARCH_REAL PETSC_ARCH_COMPLEX)
if [ -r "$HOME/.bashrc" ]; then
    _probe='printf "__PYOOMPH_ENV__\n"'
    for _v in "${NIGHTLY_ENV_VARS[@]}"; do
        _probe="$_probe; printf '%s\\n' \"\${$_v:-}\""
    done
    _detach=""
    command -v setsid >/dev/null 2>&1 && _detach="setsid -w"
    # shellcheck disable=SC2086  # unquoted on purpose: empty means "no setsid on this machine"
    _probe_out="$($_detach timeout --kill-after=10 60 bash -ic "$_probe" </dev/null 2>>"$RUN_DIR/env.log")"
    if printf '%s\n' "$_probe_out" | grep -qx '__PYOOMPH_ENV__'; then
        # Everything before the marker is the rc file's own chatter on stdout.
        _i=0
        while IFS= read -r _line; do
            [ -n "$_line" ] && export "${NIGHTLY_ENV_VARS[$_i]}=$_line"
            _i=$((_i + 1))
            [ "$_i" -ge "${#NIGHTLY_ENV_VARS[@]}" ] && break
        done <<<"$(printf '%s\n' "$_probe_out" | sed -n '/^__PYOOMPH_ENV__$/,$p' | tail -n +2)"
    else
        note "could not read the interactive environment from ~/.bashrc -- see env.log"
    fi
fi
PETSC_MISSING=""
for _v in PETSC_DIR PETSC_ARCH_REAL PETSC_ARCH_COMPLEX; do
    if [ -n "${!_v:-}" ]; then
        export "${_v?}"
    else
        PETSC_MISSING="${PETSC_MISSING:+$PETSC_MISSING }\$$_v"
    fi
done
if [ -n "$PETSC_MISSING" ]; then
    # Not fatal here: the tutorial step will say so itself, loudly, and the build and pytest steps
    # are still worth running. But the report should not have to be read backwards to find out why.
    note "PETSc not configured: $PETSC_MISSING unset after ~/.bashrc -- the tutorial step will fail"
else
    note "PETSc: real=$PETSC_DIR/$PETSC_ARCH_REAL complex=$PETSC_DIR/$PETSC_ARCH_COMPLEX"
fi

# The PYTHONPATH the pytest step runs with, built from $PETSC_DIR/$PETSC_ARCH_COMPLEX rather than
# taken from ~/.bashrc.
#
# Without an importable petsc4py the MPI half of the suite does not test anything, and does not say
# so clearly either: the modules that guard on it (test_mpi_adaptivity{,_3d}, test_mpi_eigenvalues,
# test_mpi_bifurcation_tracking) skip with "petsc4py not available", while the ones that do not run
# mpirun regardless, fall back to Pardiso in the workers and fail on its "cannot be used under MPI"
# guard. The 2026-08-09 run was 82 failed / 1184 passed / 99 skipped for exactly that reason, with
# every one of those failures an artefact of the environment. It went unnoticed because the nightly
# has never had PETSc here: PYTHONPATH comes from the interactive shell, which does not set it (the
# petsc_real/petsc_complex aliases do, per invocation), and the PETSC_PYTHONPATH that older configs
# set for this has not been read by this script since d2cf20c.
#
# The complex build, not the real one: it runs everything the real one does, and it is additionally
# what the eigenvalue tests need -- test_tensor_index_conventions and
# test_mpi_eigenvalues::test_distributed_axisymmetric_flow fail against a real-scalar PETSc, the
# latter with "Your PETSc/SLEPc installation cannot handle a complex eigenvalue problem". One arch
# for the whole suite is therefore enough, and it is the complex one.
#
# Scoped to the pytest step instead of exported: the tutorial step needs BOTH builds and picks them
# itself from PETSC_ARCH_REAL/PETSC_ARCH_COMPLEX, so it has no use for an inherited choice.
PYTEST_PYTHONPATH=""
if [ -n "${PETSC_DIR:-}" ] && [ -n "${PETSC_ARCH_COMPLEX:-}" ]; then
    PYTEST_PYTHONPATH="$PETSC_DIR/$PETSC_ARCH_COMPLEX/lib"
    # A nonexistent entry in PYTHONPATH is not an error, it just silently leaves the run without
    # PETSc -- which is the failure this whole block exists to prevent, so check rather than assume.
    if [ ! -d "$PYTEST_PYTHONPATH/petsc4py" ]; then
        note "no petsc4py under $PYTEST_PYTHONPATH -- the MPI tests will skip or fail; check \$PETSC_ARCH_COMPLEX"
    fi
    # Prepended, not replacing: whatever the interactive shell provides stays reachable behind it.
    [ -n "${PYTHONPATH:-}" ] && PYTEST_PYTHONPATH="$PYTEST_PYTHONPATH:$PYTHONPATH"
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

step_line() { # label, status, seconds (omit for a step that did not run)
    if [ -n "${3:-}" ]; then
        STEP_LINES+=("$(printf '%-22s %-22s %s' "$1" "$2" "$(hms "$3")")")
    else
        STEP_LINES+=("$(printf '%-22s %s' "$1" "$2")")
    fi
}

# ---------------------------------------------------------------------- build ---

run_step build "$RUN_DIR/build.log" "$TIMEOUT_BUILD" \
    "cd $(printf '%q' "$REPO_DIR") && ./build_for_develop.sh"
BUILD_RC=$STEP_RC
BUILD_SECS=$STEP_SECS
if [ "$BUILD_RC" -ne 0 ]; then
    FAILED_STEPS+=("build")
    step_line "build" "FAILED (rc=$BUILD_RC)" "$BUILD_SECS"
else
    step_line "build" "ok" "$BUILD_SECS"
fi

# --------------------------------------------------------------------- pytest ---
#
# Only if the build succeeded: testing the previous build would produce a mail full of
# failures that have nothing to do with the commit under test.

PYTEST_RC=0
PYTEST_SECS=0
PYTEST_FAILURES=""
PYTEST_SKIPS=""
PYTEST_SUMMARY=""
if [ "$BUILD_RC" -eq 0 ]; then
    # The documented invocation is `python -m pytest *.py --full` from inside tests/
    # (see tests/README.md); --full is what adds the slow 3D campaign and the MPI modules.
    # -rfEs: failures, errors AND skips. The skips matter -- the test_mpi_* modules skipif()
    # themselves out when mpirun or an MPI-capable solver is missing, so a cron environment
    # without mpirun turns the entire MPI half into a silent green. They are reported even on
    # a PASS for that reason. (PYTEST_PYTHONPATH above is what now supplies that solver; the
    # skips are still worth printing, because mpirun itself can go missing the same way.)
    _pytest_env=""
    # Only when non-empty: PYTHONPATH= with nothing after it is not the same as leaving it
    # unset, it puts the current directory on sys.path.
    [ -n "$PYTEST_PYTHONPATH" ] && _pytest_env="PYTHONPATH=$(printf '%q' "$PYTEST_PYTHONPATH") "
    run_step pytest "$RUN_DIR/pytest.log" "$TIMEOUT_PYTEST" \
        "cd $(printf '%q' "$REPO_DIR")/tests && ${_pytest_env}$(printf '%q' "$PYTHON") -m pytest *.py --full -rfEs"
    PYTEST_RC=$STEP_RC
    PYTEST_SECS=$STEP_SECS
    PYTEST_FAILURES="$(grep -E '^(FAILED|ERROR) ' "$RUN_DIR/pytest.log" 2>/dev/null)"
    PYTEST_SKIPS="$(grep -E '^SKIPPED ' "$RUN_DIR/pytest.log" 2>/dev/null | sort -u)"
    PYTEST_SUMMARY="$(grep -E '^=+.*(passed|failed|error|no tests ran).*=+$' "$RUN_DIR/pytest.log" 2>/dev/null | tail -n 1)"
    if [ "$PYTEST_RC" -ne 0 ]; then
        FAILED_STEPS+=("pytest")
        step_line "pytest --full" "FAILED (rc=$PYTEST_RC)" "$PYTEST_SECS"
    else
        step_line "pytest --full" "ok" "$PYTEST_SECS"
    fi
else
    step_line "pytest --full" "skipped (build failed)"
fi

# ----------------------------------------------------------------------- docs ---
#
# The documentation, built the way Read the Docs builds it: an empty virtualenv with nothing in
# it but `pip install -r docs/requirements.txt`. That answers two questions the rest of the
# nightly never asks. Whether that requirements file is still sufficient -- it is RTD's only
# input, so a dependency that is satisfied here merely because the interactive environment
# happens to have it is a broken RTD build waiting for the next rebuild there. And whether the
# docstrings still survive autodoc, which imports and introspects every module in the package.
#
# One deliberate departure from RTD: the checkout goes in front of the venv on PYTHONPATH, so
# what gets documented is develop rather than the pyoomph wheel docs/requirements.txt pulls from
# PyPI. The requirement stays in place regardless -- it is what installs pyoomph's own runtime
# dependencies (numpy, scipy, meshio, pygmsh, matplotlib), which the checkout needs in order to
# import at all; only the package itself is shadowed. Without this the step would document the
# last release and could not possibly fail on anything develop did. It is not a hypothetical
# difference: against the 0.1.9 wheel, autodoc cannot import pyoomph.equations.additional,
# .stabilization, .stabilized_ns, .viscoelastic or pyoomph.meshes.tqmesh, because they are newer
# than the release -- which is also why those API pages are empty on readthedocs.io.
#
# mpi4py on top of the requirements, for the same reason: a development build has no
# pyoomph/NO_MPI marker, so `import pyoomph` takes generic/mpi.py's MPI branch. The wheel does
# have the marker, hence neither docs/requirements.txt nor RTD needs mpi4py.
#
# Both builders run. The HTML build excludes latex_tutorial.rst (exclude_patterns in conf.py),
# so the LaTeX build is the only thing that ever looks at the master document RTD turns into the
# tutorial PDF -- and running xelatex on top of it is the only thing that catches what a caption
# does to hyperref. A truncated PDF is not a build failure to LaTeX: it writes the pages it got
# through and reports the error only in its exit status, which is why DOCS_PDF is worth having.
#
# Sphinx is not given -W. Its warnings are collected from the log and matched against
# DOCS_IGNORE_WARNINGS instead, so that the one warning RTD also lives with (no preCICE) does not
# have to turn the whole nightly red, while anything new still does.

DOCS_RC=0
DOCS_SECS=0
DOCS_WARNINGS=""
DOCS_IGNORED=""
DOCS_SKIPPED=""
DOCS_PDF_SKIPPED=""
DOCS_VENV=""
DOCS_DOCUMENTED=""
if [ "$BUILD_RC" -ne 0 ]; then
    step_line "docs" "skipped (build failed)"
    DOCS_SKIPPED="the build failed"
elif [ "${DOCS_ENABLED:-1}" = "0" ]; then
    step_line "docs" "skipped (switched off)"
    DOCS_SKIPPED="DOCS_ENABLED=0 switches the docs step off"
else
    DOCS_VENV="${NIGHTLY_TMPDIR:-${TMPDIR:-/tmp}}/docs-venv"

    # The compiled half of the package, staged into the checkout for the duration of the build.
    #
    # PYTHONPATH=$REPO_DIR is enough to import the branch only if the checkout is self-contained,
    # and with the editable install this machine uses it is not: scikit-build-core leaves the pure
    # Python in the checkout but puts _pyoomph_core.abi3.so in site-packages, and joins the two with
    # a .pth plus a finder that injects both directories into pyoomph.__path__. A deliberately empty
    # venv has neither, so "import pyoomph" got as far as expressions/__init__.py and died on
    # "No module named pyoomph._pyoomph_core" - before sphinx ran at all. Giving the venv
    # --system-site-packages would fix the import and destroy the point of the step, which is to
    # prove docs/requirements.txt sufficient on its own; copying the one extension module in keeps
    # the venv otherwise untouched.
    #
    # Skipped when the .so already lives in the checkout (an in-tree build), so nothing is
    # overwritten and nothing that was already there is removed. *.so is in .gitignore, so the
    # staged copy cannot make the next run abort on a dirty checkout, and the trap below removes it
    # even when the step fails or times out - a stale .so left behind would shadow later rebuilds.
    DOCS_CORE_SO=""
    DOCS_CORE_STAGED=""
    _docs_core="$("$PYTHON" -c 'import os,pyoomph._pyoomph_core as c
print(os.path.realpath(c.__file__))' 2>/dev/null)"
    _docs_repo_real="$(cd "$REPO_DIR" 2>/dev/null && pwd -P)"
    if [ -n "$_docs_core" ] && [ -e "$_docs_core" ] && [ -n "$_docs_repo_real" ]; then
        case "$_docs_core" in
            "$_docs_repo_real"/*) : ;;   # already in the checkout: nothing to stage
            *)
                if [ -e "$REPO_DIR/pyoomph/$(basename "$_docs_core")" ]; then
                    note "docs: $REPO_DIR/pyoomph/$(basename "$_docs_core") exists already -- not staging over it"
                else
                    DOCS_CORE_SO="$_docs_core"
                    DOCS_CORE_STAGED="$REPO_DIR/pyoomph/$(basename "$_docs_core")"
                fi
                ;;
        esac
    fi
    _docs_stage_cmd=""
    if [ -n "$DOCS_CORE_SO" ]; then
        _docs_stage_cmd="
trap 'rm -f $(printf '%q' "$DOCS_CORE_STAGED")' EXIT
echo '=== staging the compiled extension into the checkout ==='
cp $(printf '%q' "$DOCS_CORE_SO") $(printf '%q' "$DOCS_CORE_STAGED") || exit 1"
    else
        DOCS_CORE_MISSING=1
    fi

    _docs_pdf_cmd=""
    if [ "${DOCS_PDF:-1}" != "0" ]; then
        if command -v latexmk >/dev/null 2>&1 && command -v xelatex >/dev/null 2>&1 &&
           command -v make >/dev/null 2>&1; then
            # The Makefile sphinx writes next to the .tex already knows the engine (xelatex, see
            # latex_engine in conf.py) and the number of passes. -f is deliberately NOT passed:
            # forcing latexmk past an error is exactly how a half-typeset PDF comes to look like a
            # successful build.
            _docs_pdf_cmd="echo '=== latexmk (xelatex) ==='
make -C $(printf '%q' "$RUN_DIR/docs_latex") all-pdf"
        else
            DOCS_PDF_SKIPPED="latexmk, xelatex or make is not on PATH"
        fi
    else
        DOCS_PDF_SKIPPED="DOCS_PDF=0 switches the PDF pass off"
    fi

    # LC_ALL=C: sphinx translates its own console output, and this log is both parsed and quoted
    # into the report -- on this machine it otherwise ends with "build abgeschlossen" and the mail
    # comes out in two languages. LANGUAGE=en, the usual way to ask for that, does nothing here:
    # sphinx picks its catalogue from locale.getlocale() rather than through gettext's own
    # environment search, and only LC_ALL/LANG reach that. PYTHONUTF8=1 comes with it because
    # plain LC_ALL=C would otherwise make ASCII the default encoding for every file sphinx opens
    # without saying so, and C.UTF-8 is not present everywhere.
    #
    # The venv lives under the per-run scratch rather than in the checkout: the next run wipes that
    # directory anyway, and a stray directory inside REPO_DIR would make the following night abort
    # on a dirty checkout.
    run_step docs "$RUN_DIR/docs.log" "$TIMEOUT_DOCS" "
set -o pipefail
export LC_ALL=C PYTHONUTF8=1
# The nightly exports the PYTHONPATH it read out of ~/.bashrc, and a virtualenv does not shadow
# it -- entries on PYTHONPATH come before the venv's own site-packages. Leaving it in place makes
# 'fresh environment' untrue: here it put the PETSc build's petsc4py and slepc4py in front of
# everything, which pip duly resolved against ('slepc4py requires numpy<2'). Clear it, and set it
# to the checkout alone once the installing is done.
unset PYTHONPATH$_docs_stage_cmd
rm -rf $(printf '%q' "$DOCS_VENV") || exit 1
$(printf '%q' "$PYTHON") -m venv $(printf '%q' "$DOCS_VENV") || exit 1
echo '=== pip install -r docs/requirements.txt ==='
$(printf '%q' "$DOCS_VENV/bin/python") -m pip install -r $(printf '%q' "$REPO_DIR/docs/requirements.txt") || exit 1
echo '=== pip install mpi4py (the checkout is an MPI build; the wheel is not) ==='
$(printf '%q' "$DOCS_VENV/bin/python") -m pip install mpi4py || exit 1
export PYTHONPATH=$(printf '%q' "$REPO_DIR")
echo '=== which pyoomph is being documented ==='
$(printf '%q' "$DOCS_VENV/bin/python") -c 'import os,sys,pyoomph,pyoomph._version
f=os.path.realpath(pyoomph.__file__); repo=os.path.realpath(sys.argv[1])
print(\"documenting\", f, pyoomph._version.__version__)
if not f.startswith(repo+os.sep):
    sys.exit(\"NOT the checkout under \"+repo+\" -- PYTHONPATH did not win over the installed wheel, \"
             \"so this would have documented the last release rather than the branch under test\")' $(printf '%q' "$REPO_DIR") || exit 1
echo '=== sphinx -b html ==='
$(printf '%q' "$DOCS_VENV/bin/python") -m sphinx -b html $(printf '%q' "$REPO_DIR/docs/source") $(printf '%q' "$RUN_DIR/docs_html") || exit 1
echo '=== sphinx -b latex ==='
$(printf '%q' "$DOCS_VENV/bin/python") -m sphinx -b latex $(printf '%q' "$REPO_DIR/docs/source") $(printf '%q' "$RUN_DIR/docs_latex") || exit 1
$_docs_pdf_cmd"
    DOCS_RC=$STEP_RC
    DOCS_SECS=$STEP_SECS

    DOCS_DOCUMENTED="$(grep -m1 '^documenting ' "$RUN_DIR/docs.log" 2>/dev/null)"
    # Only from the first sphinx marker onwards, and only sphinx's own two shapes -- a bare
    # "WARNING: ..." and a "<file>:<line>: WARNING: ...". pip writes WARNING: lines of its own
    # (a new version of itself is available, a hash could not be checked), and those would
    # otherwise fail the run for nothing. LaTeX writes "LaTeX Warning:", which does not match.
    _docs_warn_all="$(sed -n '/^=== sphinx -b html ===$/,$p' "$RUN_DIR/docs.log" 2>/dev/null |
                      grep -E '(^|: )WARNING:' | sort -u)"
    # An empty DOCS_IGNORE_WARNINGS means "ignore nothing", which is not what an empty pattern
    # does to grep -- that matches every line and would silence the whole step.
    _docs_ignore="${DOCS_IGNORE_WARNINGS:-\$^}"
    if [ -n "$_docs_warn_all" ]; then
        DOCS_IGNORED="$(printf '%s\n' "$_docs_warn_all" | grep -E "$_docs_ignore")"
        DOCS_WARNINGS="$(printf '%s\n' "$_docs_warn_all" | grep -vE "$_docs_ignore")"
    fi

    if [ "$DOCS_RC" -ne 0 ]; then
        FAILED_STEPS+=("docs")
        step_line "docs" "FAILED (rc=$DOCS_RC)" "$DOCS_SECS"
    elif [ -n "$DOCS_WARNINGS" ]; then
        FAILED_STEPS+=("docs")
        step_line "docs" "FAILED ($(printf '%s\n' "$DOCS_WARNINGS" | wc -l) warnings)" "$DOCS_SECS"
    else
        step_line "docs" "ok" "$DOCS_SECS"
        # Only on success -- a failed step is worth being able to look at and re-run by hand. The
        # venv is 220 MB and the two output trees another 200 MB; kept for all KEEP_RUNS runs that
        # would be several gigabytes of rendered HTML nobody reads, while docs.log, which is what
        # the report quotes, is small and stays.
        rm -rf "$DOCS_VENV" "$RUN_DIR/docs_html" "$RUN_DIR/docs_latex"
    fi
fi

# ------------------------------------------------------------------ tutorials ---
#
# Two passes over the same set of scripts: serially, and -- unless TUTORIAL_MPI_RANKS is 0 or
# mpirun is not on PATH -- under `mpirun -n $TUTORIAL_MPI_RANKS` without --distribute, so every
# rank builds the whole mesh and solves it. That is what a user gets who starts a script under
# mpirun without thinking about distribution, and it exercises a different half of the parallel
# code than the distributed pytest suites do.
#
# The passes cannot be run back to back and their logs collected afterwards: the runner rebuilds
# citools/pyoomph_tutorial_scripts/ from the tutorial sources at startup, deleting the per-script
# logs the previous pass left there. Hence the copy at the end of each pass.
#
# Together they are by far the longest part of the nightly. A machine that cannot afford both sets
# TUTORIAL_MPI_RANKS=0 in the config; the flock at the top means a run that overruns into the next
# night costs a night rather than two overlapping runs.

PASS_LABELS=()
PASS_SECS=()
PASS_FAILURES=()
PASS_SKIPS=()
PASS_SELFSKIPS=()
PASS_BAD=()
PASS_LOG=()
PASS_LOGDIR=()
PASS_TIMES=()
PASS_TIMETOTAL=()
PASS_TIMEMISSING=()

run_tutorial_pass() { # label, tag, timeout, extra arguments for the runner
    local label="$1" tag="$2" tmo="$3" extra="$4" log logdir rc bad failures skips selfskips secs times
    log="$RUN_DIR/tutorials_$tag.log"
    logdir="$RUN_DIR/tutorial_logs_$tag"

    # The runner drops the previous pass's per-script logs when it rebuilds the bundle -- but only
    # if it gets that far. When it dies before that (on 2026-08-09: the PETSc arches were not in its
    # environment, so it raised in its first second), the harvest below picks up whatever logs the
    # previous pass, or a previous night, or a hand-started run left behind, and the report quotes
    # them as failures of this pass. Clearing first makes the harvest show only this pass's work.
    if [ -d "$REPO_DIR/citools/pyoomph_tutorial_scripts" ]; then
        find "$REPO_DIR/citools/pyoomph_tutorial_scripts" -name '*.log' -delete 2>/dev/null
    fi

    run_step "tutorials ($label)" "$log" "$tmo" \
        "cd $(printf '%q' "$REPO_DIR") && $(printf '%q' "$PYTHON") -u citools/test_all_tutorial_scripts.py $extra"
    rc=$STEP_RC
    secs=$STEP_SECS

    # The runner always exits 0; the verdict is in its output. PROBLEM: lines are the bundle's own
    # consistency check (duplicated scripts that have drifted apart); they also fail the run, but
    # they are printed before the first script and would be out of reach of the log tail below.
    failures="$(grep -E '^PROBLEM: |=+ FAILED ' "$log" 2>/dev/null)"
    # Scripts the runner did not count against the verdict because an optional package is missing
    # (preCICE). Not a failure, but not coverage either, so it goes in the report the same way the
    # pytest skips do.
    skips="$(sed -n '/^SKIPPED FOR MISSING OPTIONAL DEPENDENCIES:/,/^$/p' "$log" \
             2>/dev/null | grep -E '^ +\S+ needs ')"
    # And the ones the runner itself refuses to start under mpirun -- parallel_running.py spawns its
    # own mpirun, the deflation scripts use a custom assembler that is not MPI-capable yet, and the
    # Crouzeix-Raviart condensation needs --distribute, which this pass deliberately does not pass.
    # Reported for the same reason: the MPI pass covers slightly less than the serial one.
    selfskips="$(grep -E '^ +SKIPPING .* -- ' "$log" 2>/dev/null | sed 's/^ *//' | sort -u)"

    # "TIME <seconds> s <folder>/<script>" per script, from the runner's SIMULATION TIMES section:
    # the elapsed time each run recorded for itself, which is what makes the serial and the mpirun
    # pass comparable at all. The full table is kept on disk; the mail only names the slowest few.
    times="$RUN_DIR/timings_$tag.txt"
    grep -E '^ +TIME +[0-9]' "$log" 2>/dev/null | sed 's/^ *//' >"$times"
    [ -s "$times" ] || { rm -f "$times"; times=""; }
    PASS_TIMES+=("$times")
    PASS_TIMETOTAL+=("$(grep -E '^ +TIME TOTAL ' "$log" 2>/dev/null | sed 's/^ *TIME TOTAL *//')")
    PASS_TIMEMISSING+=("$(grep -E '^ +TIME MISSING for ' "$log" 2>/dev/null | sed 's/^ *TIME MISSING for *//')")

    bad=0
    [ "$rc" -ne 0 ] && bad=1
    [ -n "$failures" ] && bad=1
    grep -q '^SOME TESTS FAILED' "$log" 2>/dev/null && bad=1
    grep -q '^ALL TESTS PASSED' "$log" 2>/dev/null || bad=1

    if [ -d "$REPO_DIR/citools/pyoomph_tutorial_scripts" ]; then
        mkdir -p "$logdir"
        (cd "$REPO_DIR/citools/pyoomph_tutorial_scripts" && find . -name '*.log' -print0 |
            while IFS= read -r -d '' f; do
                cp "$f" "$logdir/$(printf '%s' "${f#./}" | tr '/' '_')" 2>/dev/null
            done)
        rmdir "$logdir" 2>/dev/null
    fi

    PASS_LABELS+=("$label")
    PASS_SECS+=("$secs")
    PASS_FAILURES+=("$failures")
    PASS_SKIPS+=("$skips")
    PASS_SELFSKIPS+=("$selfskips")
    PASS_BAD+=("$bad")
    PASS_LOG+=("$log")
    PASS_LOGDIR+=("$logdir")

    if [ "$bad" -ne 0 ]; then
        FAILED_STEPS+=("tutorials ($label)")
        step_line "tutorials $label" "FAILED" "$secs"
    else
        step_line "tutorials $label" "ok" "$secs"
    fi
}

MPI_TUTORIALS_SKIPPED=""
if [ "$BUILD_RC" -eq 0 ]; then
    run_tutorial_pass "serial" serial "$TIMEOUT_TUTORIALS" ""

    case "${TUTORIAL_MPI_RANKS:-0}" in
        ''|*[!0-9]*)
            MPI_TUTORIALS_SKIPPED="TUTORIAL_MPI_RANKS=\"${TUTORIAL_MPI_RANKS:-}\" is not a number -- fix $CONFIG"
            step_line "tutorials mpi" "skipped (bad config)" ;;
        0)
            MPI_TUTORIALS_SKIPPED="TUTORIAL_MPI_RANKS=0 switches the MPI pass off"
            step_line "tutorials mpi" "skipped (switched off)" ;;
        *)
            if command -v mpirun >/dev/null 2>&1; then
                run_tutorial_pass "mpirun -n $TUTORIAL_MPI_RANKS" mpi "$TIMEOUT_TUTORIALS_MPI" \
                    "--mpirun $TUTORIAL_MPI_RANKS"
            else
                MPI_TUTORIALS_SKIPPED="mpirun is not on PATH"
                step_line "tutorials mpi" "skipped (no mpirun)"
            fi ;;
    esac
else
    step_line "tutorials" "skipped (build failed)"
fi

# --------------------------------------------------------------------- report ---

# A run where the MPI modules skipped themselves is not a green run, it is an untested
# one, and it must not look identical to a real pass in the subject line. The same goes for
# tutorial scripts the runner had to skip for want of an optional package, and for the whole
# MPI tutorial pass when it could not be started.
MPI_SKIPPED=0
printf '%s' "$PYTEST_SKIPS" | grep -q 'test_mpi_' && MPI_SKIPPED=1

# The same script is skipped by both passes, so the union rather than the sum.
TUT_SKIPS_ALL=""
for i in "${!PASS_SKIPS[@]}"; do
    [ -n "${PASS_SKIPS[$i]}" ] && TUT_SKIPS_ALL="${TUT_SKIPS_ALL}${PASS_SKIPS[$i]}"$'\n'
done
TUT_SKIPS_ALL="$(printf '%s' "$TUT_SKIPS_ALL" | grep -v '^$' | sort -u)"

if [ ${#FAILED_STEPS[@]} -eq 0 ]; then
    VERDICT="PASS"
    SKIPNOTE=""
    [ "$MPI_SKIPPED" -ne 0 ] && SKIPNOTE="the MPI tests"
    if [ -n "$MPI_TUTORIALS_SKIPPED" ] && [ "$BUILD_RC" -eq 0 ]; then
        SKIPNOTE="${SKIPNOTE:+$SKIPNOTE and }the MPI tutorial pass"
    fi
    # The step not running at all, but not its PDF pass alone: TeX is a far less usual thing for a
    # numerics machine to have than mpirun, and a permanent "but ... were SKIPPED" in every subject
    # line teaches people to stop reading it. Coverage below names the PDF either way.
    if [ -n "$DOCS_SKIPPED" ] && [ "$BUILD_RC" -eq 0 ]; then
        SKIPNOTE="${SKIPNOTE:+$SKIPNOTE and }the docs build"
    fi
    if [ -n "$TUT_SKIPS_ALL" ]; then
        SKIPNOTE="${SKIPNOTE:+$SKIPNOTE and }$(printf '%s\n' "$TUT_SKIPS_ALL" | wc -l) tutorial script(s)"
    fi
    if [ -n "$SKIPNOTE" ]; then
        SUBJECT="[pyoomph nightly] $BRANCH ${REMOTE_SHA:0:12} PASS -- but $SKIPNOTE were SKIPPED"
    else
        SUBJECT="[pyoomph nightly] $BRANCH ${REMOTE_SHA:0:12} PASS"
    fi
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

# Reported whatever the verdict: what was NOT run is as much a part of the result as what
# failed. pytest --full selects the whole suite, but the MPI modules still skipify
# themselves out when mpirun or an MPI-capable solver is missing.
if [ "$BUILD_RC" -eq 0 ]; then
    say ""
    say "Coverage"
    say "--------"
    if command -v mpirun >/dev/null 2>&1; then
        say "  mpirun: $(command -v mpirun)"
    else
        say "  mpirun: NOT ON PATH -- every MPI test skipped itself, and the MPI tutorial pass could"
        say "          not run either. The nightly takes PATH from an interactive bash, so either"
        say "          ~/.bashrc does not put mpirun on it, or ENV_SETUP in $CONFIG has to."
    fi
    if [ -n "$MPI_TUTORIALS_SKIPPED" ]; then
        say "  tutorials under mpirun: NOT RUN -- $MPI_TUTORIALS_SKIPPED"
    else
        say "  tutorials: run twice, serially and under mpirun -n $TUTORIAL_MPI_RANKS (no --distribute,"
        say "             i.e. every rank builds the whole mesh)"
    fi
    if [ -n "$PETSC_MISSING" ]; then
        say "  PETSc: $PETSC_MISSING unset after ~/.bashrc -- the tutorial step could not start."
    else
        say "  PETSc: real $PETSC_DIR/$PETSC_ARCH_REAL, complex $PETSC_DIR/$PETSC_ARCH_COMPLEX"
        say "         pytest ran against the complex build (PYTHONPATH=$PYTEST_PYTHONPATH)"
    fi
    if [ -n "$DOCS_SKIPPED" ]; then
        say "  docs: NOT BUILT -- $DOCS_SKIPPED"
    else
        # Which pyoomph autodoc actually got hold of. The step refuses to continue when it is not
        # the checkout, but the line is worth having in the report either way: "documented the last
        # release" and "documented develop" produce the same green tick otherwise.
        say "  docs: ${DOCS_DOCUMENTED:-(the build did not get as far as importing pyoomph)}"
        say "        built in a fresh virtualenv from docs/requirements.txt (+ mpi4py), HTML and LaTeX"
        if [ -n "$DOCS_PDF_SKIPPED" ]; then
            say "        PDF: NOT RUN -- $DOCS_PDF_SKIPPED. Nothing then typesets latex_tutorial.rst,"
            say "             which the HTML build excludes, so only its RST is checked."
        else
            say "        PDF: xelatex run over the generated tutorial as well"
        fi
        if [ -n "$DOCS_IGNORED" ]; then
            say "        warnings ignored (DOCS_IGNORE_WARNINGS):"
            printf '%s\n' "$DOCS_IGNORED" | sed 's/^/          /' >>"$REPORT"
        fi
    fi
    if [ -n "$PYTEST_SKIPS" ]; then
        say "  skipped by pytest:"
        printf '%s\n' "$PYTEST_SKIPS" | sed 's/^/    /' >>"$REPORT"
    else
        say "  skipped by pytest: nothing"
    fi
    if [ -n "$TUT_SKIPS_ALL" ]; then
        say "  tutorial scripts skipped for a missing optional package:"
        printf '%s\n' "$TUT_SKIPS_ALL" | sed 's/^ */    /' >>"$REPORT"
    else
        say "  tutorial scripts skipped: nothing"
    fi
    for i in "${!PASS_LABELS[@]}"; do
        if [ -n "${PASS_SELFSKIPS[$i]}" ]; then
            say "  not run in the ${PASS_LABELS[$i]} pass (the runner excludes them there):"
            printf '%s\n' "${PASS_SELFSKIPS[$i]}" | sed 's/^/    /' >>"$REPORT"
        fi
    done
fi

# ---------------------------------------------------------------------- timing ---
#
# On a green run as much as on a red one: a change that halves a solve, an MPI pass that turns out
# slower than the serial one, or a machine that has quietly become slower show up nowhere else, and
# only as a series over several nights -- so the numbers have to be in every report, not just in the
# ones somebody goes looking for.
HAVE_TIMINGS=0
for i in "${!PASS_LABELS[@]}"; do
    [ -n "${PASS_TIMES[$i]}" ] && HAVE_TIMINGS=1
done
if [ "$HAVE_TIMINGS" = 1 ] && [ "${TIMING_TOP:-0}" -gt 0 ] 2>/dev/null; then
    say ""
    say "Timing"
    say "------"
    say "  What each script's own run recorded as its elapsed time: problem setup, code generation"
    say "  and compilation and every solve, but not the interpreter start-up and the imports. The"
    say "  complete per-script tables are in $RUN_DIR/timings_*.txt."
    for i in "${!PASS_LABELS[@]}"; do
        say ""
        say "  ${PASS_LABELS[$i]}: ${PASS_TIMETOTAL[$i]:-(nothing recorded)}"
        [ -n "${PASS_TIMEMISSING[$i]}" ] && say "    no time recorded for ${PASS_TIMEMISSING[$i]}"
        if [ -n "${PASS_TIMES[$i]}" ]; then
            say "    slowest (up to $TIMING_TOP):"
            # The runner already sorts them, slowest first.
            head -n "$TIMING_TOP" "${PASS_TIMES[$i]}" |
                awk '{ printf "      %10.2f s  %s\n", $2, $4 }' >>"$REPORT"
        fi
    done
    # Every later pass against the first (i.e. the serial one), which is what says whether running
    # under mpirun actually bought anything. Scripts only one of the passes ran are left out; the
    # Coverage section above says which those are.
    for i in "${!PASS_LABELS[@]}"; do
        [ "$i" -gt 0 ] || continue
        [ -n "${PASS_TIMES[0]}" ] && [ -n "${PASS_TIMES[$i]}" ] || continue
        say ""
        say "  ${PASS_LABELS[0]} -> ${PASS_LABELS[$i]}, largest changes:"
        awk 'NR==FNR { a[$4]=$2+0; next }
             ($4 in a) {
                 d=$2+0-a[$4]; ad=(d<0 ? -d : d)
                 printf "%.6f\t      %10.2f s -> %10.2f s  %+7.1f%%  %s\n", \
                        ad, a[$4], $2+0, (a[$4]>0 ? 100*d/a[$4] : 0), $4
             }' "${PASS_TIMES[0]}" "${PASS_TIMES[$i]}" |
            sort -rn | head -n "$TIMING_TOP" | cut -f2- >>"$REPORT"
    done
fi

if [ "$VERDICT" = "PASS" ]; then
    say ""
    [ -n "$PYTEST_SUMMARY" ] && say "pytest: $PYTEST_SUMMARY"
    if [ -n "$MPI_TUTORIALS_SKIPPED" ]; then
        say "Tutorial pipeline: all scripts ran serially; the mpirun pass did not run."
    else
        say "Tutorial pipeline: all scripts ran, serially and under mpirun -n $TUTORIAL_MPI_RANKS."
    fi
    if [ -z "$DOCS_SKIPPED" ]; then
        say "Documentation: built from docs/requirements.txt in a clean virtualenv, no new warnings."
    fi
    say "(preCICE runs and the interactive GUI tutorial scripts still need a manual check.)"
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
            # The summary lines truncate the reason to "AssertionError: ...", which is not
            # enough to act on -- and when a whole subsystem fails at once, the first
            # traceback usually explains all of it. Quote the head of the FAILURES section.
            if [ "$PYTEST_DETAIL_LINES" -gt 0 ] 2>/dev/null; then
                detail="$(sed -n '/^=\+ FAILURES =\+$/,/^=\+ \(short test summary\|warnings summary\|[0-9]\)/p' \
                          "$RUN_DIR/pytest.log" 2>/dev/null | head -n "$PYTEST_DETAIL_LINES")"
                if [ -n "$detail" ]; then
                    say ""
                    say "--- first $PYTEST_DETAIL_LINES lines of the pytest FAILURES section ---"
                    printf '%s\n' "$detail" >>"$REPORT"
                    say "--- end (the whole section is in $RUN_DIR/pytest.log) ---"
                fi
            fi
        else
            say "(no FAILED/ERROR lines -- pytest itself did not get that far)"
            quote_log "$RUN_DIR/pytest.log" "pytest"
        fi
    fi

    if [ "$DOCS_RC" -ne 0 ] || [ -n "$DOCS_WARNINGS" ]; then
        say ""
        say "DOCS FAILED"
        say "==========="
        if [ -n "$DOCS_WARNINGS" ]; then
            say "Sphinx warnings that are not on the DOCS_IGNORE_WARNINGS list:"
            say ""
            printf '%s\n' "$DOCS_WARNINGS" >>"$REPORT"
            say ""
        fi
        if [ "$DOCS_RC" -ne 0 ]; then
            # pip needs the network, and a machine that has none fails here every night with a
            # perfectly ordinary-looking error a long way up the log.
            say "The step itself exited $DOCS_RC. If it died in the pip install, check that this"
            say "machine can reach PyPI at all before looking at docs/requirements.txt."
            quote_log "$RUN_DIR/docs.log" "docs"
        fi
        [ -n "$DOCS_VENV" ] && [ -d "$DOCS_VENV" ] &&
            say "(the virtualenv is left at $DOCS_VENV until the next run, to re-run the build by hand)"
    fi

    for i in "${!PASS_LABELS[@]}"; do
        [ "${PASS_BAD[$i]}" -ne 0 ] || continue
        say ""
        say "TUTORIAL PIPELINE FAILED (${PASS_LABELS[$i]})"
        say "========================"
        if [ -n "${PASS_FAILURES[$i]}" ]; then
            printf '%s\n' "${PASS_FAILURES[$i]}" >>"$REPORT"
        else
            say "(the runner reported a failure without naming a script -- see $(basename "${PASS_LOG[$i]}"))"
            quote_log "${PASS_LOG[$i]}" "tutorials ${PASS_LABELS[$i]}"
        fi
        # One tail per failing script, capped: a broken core commit fails all 127 of them
        # and the mail would be unreadable. The rest are on disk in $RUN_DIR.
        if [ -d "${PASS_LOGDIR[$i]}" ]; then
            n=0
            for f in "${PASS_LOGDIR[$i]}"/*.log; do
                [ -f "$f" ] || continue
                n=$((n + 1))
                if [ "$n" -gt 10 ]; then
                    say ""
                    say "(further failing-script logs omitted -- all of them are in ${PASS_LOGDIR[$i]})"
                    break
                fi
                quote_log "$f" "tutorial $(basename "$f" .log)"
            done
        fi
    done
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
