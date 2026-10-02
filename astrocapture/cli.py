"""Command line interface.

Usage::

    python -m astrocapture --config examples/sim_session.yaml   # run a session
    python -m astrocapture plan --config examples/sim_session.yaml
    python -m astrocapture --list-drivers
"""

from __future__ import annotations

import argparse
import sys

from astrocapture import __version__, config
from astrocapture.drivers import available_drivers, make_camera, make_mount
from astrocapture.sequencer import Sequencer


def cmd_run(args: argparse.Namespace) -> int:
    plan = config.load_plan(args.config)
    print(config.plan_summary(plan))
    print()
    mount = make_mount(plan.mount.driver, **plan.mount.options)
    camera = make_camera(plan.camera.driver, **plan.camera.options)
    seq = Sequencer(plan, mount, camera)
    try:
        final = seq.run()
    except KeyboardInterrupt:
        print("\nCtrl-C: aborting…")
        seq.abort()
        final = seq.run() if seq.state.name == "IDLE" else seq.state
    print(f"\nSession directory: {seq.session.dir}")
    print(f"Final state: {final.value}")
    return 0 if str(final.value) == "done" else 1


def cmd_plan(args: argparse.Namespace) -> int:
    try:
        plan = config.load_plan(args.config)
    except config.PlanError as exc:
        print(f"INVALID PLAN\n{exc}", file=sys.stderr)
        return 2
    print("Plan is valid.\n")
    print(config.plan_summary(plan))
    return 0


def cmd_list_drivers(_args: argparse.Namespace) -> int:
    avail = available_drivers()
    print(f"AstroCapture {__version__}")
    print("Mount drivers :", ", ".join(avail["mounts"]) or "(none)")
    print("Camera drivers:", ", ".join(avail["cameras"]) or "(none)")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="astrocapture",
        description="Telescope + camera control and capture sequencing.",
    )
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    p.add_argument("--list-drivers", action="store_true",
                   help="list available mount/camera drivers and exit")
    p.add_argument("--config", default="examples/sim_session.yaml",
                   help="plan YAML to run (default: simulator session)")
    sub = p.add_subparsers(dest="command")
    sp = sub.add_parser("plan", help="validate a plan file and print a summary")
    sp.add_argument("--config", required=True, help="plan YAML to validate")
    sp.set_defaults(func=cmd_plan)
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.list_drivers:
        return cmd_list_drivers(args)
    if getattr(args, "func", None) is not None:
        return args.func(args)
    try:
        return cmd_run(args)
    except config.PlanError as exc:
        print(f"INVALID PLAN\n{exc}", file=sys.stderr)
        return 2
    except (ValueError, ImportError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
