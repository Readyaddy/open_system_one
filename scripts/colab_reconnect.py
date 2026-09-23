"""Reconnects a locally-named colab-cli session to a Colab VM that's still
alive server-side but got pruned from local tracking
(~/.config/colab-cli/sessions.json).

Why this exists: colab-cli's `sync_sessions()` auto-removes any locally
tracked session whose endpoint doesn't show up in a `list_assignments()`
call (colab_cli/common.py's `prune_session`) -- this can fire on a single
transient/flaky listing even though the VM itself, and anything running on
it (e.g. a detached `subprocess.Popen(..., start_new_session=True)`
training job), was never touched. After a prune, `colab status -s <name>`
and `colab exec -s <name>` both fail with "Session not found", even though
`colab sessions` still shows the VM alive, just unnamed (`[?]`).

The bundled COLAB_SKILL.md's own "Recovery" section only covers the case
where the backend genuinely killed the VM ("re-create with `colab new`").
Running `colab new -s <name>` here would be wrong and dangerous: it
allocates a genuinely NEW VM rather than reusing the orphaned one -- this
happened by accident at least twice on this project before the correct fix
was worked out (see experiments/exp6_diverse_data_qqp_aux/SESSION_LOG.md,
"Never run `colab new -s <name>` to 'reconnect'").

This script instead uses colab-cli's OWN internal Client/StateStore classes
(not a hand-rolled HTTP call to the assignments API -- an earlier attempt at
that 400'd; the exact request shape colab-cli's Client._issue_request sends,
including the `authuser=0` param, isn't otherwise documented) to:
  1. List currently-active server-side assignments (`colab sessions`'
     underlying call).
  2. Find the one whose endpoint isn't in any locally-tracked session
     (an "orphan") -- or, if --endpoint is given explicitly, use that one.
  3. Write a fresh SessionState entry into sessions.json under the name you
     ask for, using that assignment's real token/url. kernel_id/session_id
     are left None -- colab-cli looks those up fresh on next use, exactly
     as documented in SESSION_LOG.md's original manual fix.

Usage:
  python3 colab_reconnect.py <name>                       # auto-detect the one orphan
  python3 colab_reconnect.py <name> --endpoint <endpoint>  # explicit, when there are several orphans
  python3 colab_reconnect.py <name> --auth oauth2          # non-default auth strategy
  python3 colab_reconnect.py <name> --config /tmp/x.json   # isolated session-state file

Exit codes: 0 on success, 1 if no orphan (or the named endpoint) could be
found, 2 on ambiguous auto-detection (more than one orphan, no --endpoint given).
"""
import argparse
import sys

from colab_cli.auth import AuthProvider
from colab_cli.common import State
from colab_cli.state import SessionState


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("name", help="Local session name to (re)assign, e.g. jepa-exp7-train.")
    ap.add_argument("--endpoint", default=None,
                     help="Explicit server-side endpoint to attach to (from `colab sessions`'s "
                          "`[?] <endpoint>` listing). Required only when there's more than one "
                          "orphaned assignment and auto-detection would be ambiguous.")
    ap.add_argument("--auth", choices=["adc", "oauth2"], default="adc",
                     help="Same meaning as the colab CLI's own --auth flag. Default matches the "
                          "CLI's default (adc) -- see COLAB_SKILL.md's Authentication section for why "
                          "ADC is preferred for headless/agent use.")
    ap.add_argument("--config", default=None,
                     help="Path to sessions.json, if using an isolated state file (the CLI's own "
                          "--config flag). Defaults to ~/.config/colab-cli/sessions.json.")
    args = ap.parse_args()

    state = State()
    state.auth_provider = AuthProvider(args.auth)
    if args.config:
        state.config_path = args.config

    print(f"Listing active server-side assignments...")
    assignments = state.client.list_assignments()
    if not assignments:
        print("No active assignments at all -- there is no VM to reconnect to.")
        sys.exit(1)

    tracked = state.store.list()
    tracked_endpoints = {s.endpoint for s in tracked.values()}
    existing = tracked.get(args.name)

    for a in assignments:
        marker = "orphan" if a.endpoint not in tracked_endpoints else "already tracked"
        print(f"  endpoint={a.endpoint}  variant={a.variant.name}  "
              f"accelerator={a.accelerator.value}  [{marker}]")

    if args.endpoint:
        match = next((a for a in assignments if a.endpoint == args.endpoint), None)
        if match is None:
            print(f"\nERROR: endpoint '{args.endpoint}' is not among the active assignments above.")
            sys.exit(1)
    else:
        orphans = [a for a in assignments if a.endpoint not in tracked_endpoints]
        if not orphans:
            if existing:
                print(f"\n'{args.name}' is already tracked locally and no orphans exist -- "
                      f"nothing to reconnect (try `colab status -s {args.name}` directly).")
            else:
                print(f"\nNo orphaned assignments found, and '{args.name}' isn't tracked either -- "
                      f"the VM may genuinely be gone. Check `colab sessions` yourself.")
            sys.exit(1)
        if len(orphans) > 1:
            print(f"\n{len(orphans)} orphaned assignments found -- ambiguous. Re-run with "
                  f"--endpoint <one of the endpoints listed above as [orphan]>.")
            sys.exit(2)
        match = orphans[0]

    sess = SessionState(
        name=args.name,
        token=match.runtime_proxy_info.token,
        url=match.runtime_proxy_info.url,
        endpoint=match.endpoint,
        variant=match.variant.name,
        accelerator=match.accelerator.value,
        kernel_id=None,   # looked up fresh on next `colab exec`/`status`, per SESSION_LOG.md's fix
        session_id=None,
        last_execution=None,
        running=None,
        keep_alive_pid=None,
    )
    state.store.add(sess)
    print(f"\nReconnected: '{args.name}' -> {match.endpoint}")
    print(f"  url:   {match.runtime_proxy_info.url}")
    print(f"  token: expires in {match.runtime_proxy_info.token_expires_in_seconds}s "
          f"(colab-cli refreshes this itself on next use)")
    print(f"\nVerify with: colab --auth={args.auth} status -s {args.name}")


if __name__ == "__main__":
    main()
