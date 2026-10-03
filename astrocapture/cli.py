"""Command line interface.

Usage::

    python -m astrocapture --config examples/sim_session.yaml   # run a session
    python -m astrocapture plan --config examples/sim_session.yaml
    python -m astrocapture night --config examples/night_queue.yaml
    python -m astrocapture catalog "M51"             # look up a deep-sky object
    python -m astrocapture tonight --lat 43.38 --lon -80.96
    python -m astrocapture process sessions/m51-sim-20250101-120000
    python -m astrocapture process sessions/<name> --min-quality 0.5 --denoise
    python -m astrocapture ask "image M51 tonight, 2 hours of data"
    python -m astrocapture export-training-data sessions/<name> --output training/
    python -m astrocapture eaa --config examples/sim_session.yaml --frames 6
    python -m astrocapture --list-drivers
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone

from astrocapture import __version__, ai_assistant, catalog, config
from astrocapture.dash import serve as dash_serve
from astrocapture.drivers import available_drivers, make_camera, make_mount
from astrocapture.eaa import run_eaa
from astrocapture.process import process_session
from astrocapture.quality import export_training_data
from astrocapture.scheduler import NightScheduler
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


def cmd_dash(args: argparse.Namespace) -> int:
    return dash_serve(args.config, port=args.port)


def cmd_night(args: argparse.Namespace) -> int:
    """Run a multi-target night queue until astronomical dawn."""
    try:
        plan = config.load_plan(args.config)
    except config.PlanError as exc:
        print(f"INVALID PLAN\n{exc}", file=sys.stderr)
        return 2
    if plan.night is None:
        print("ERROR: plan has no 'night:' block — add one "
              "(see examples/night_queue.yaml)", file=sys.stderr)
        return 2
    print(config.plan_summary(plan))
    print()

    def factory():
        return (make_mount(plan.mount.driver, **plan.mount.options),
                make_camera(plan.camera.driver, **plan.camera.options))

    try:
        return NightScheduler(plan, factory).run()
    except (ValueError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


def cmd_process(args: argparse.Namespace) -> int:
    denoise: bool | dict = False
    if args.denoise_model:
        denoise = {
            "model": args.denoise_model,
            "providers": (args.denoise_providers.split(",")
                          if args.denoise_providers else None),
        }
    elif args.denoise:
        denoise = True
    try:
        stats = process_session(
            args.session_dir,
            args.output,
            quality_min_score=args.min_quality,
            denoise=denoise,
            trails=args.trails,
        )
    except (ImportError, FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(f"Stacked {stats.get('n_stacked', 0)}/{stats.get('n_lights', 0)} "
          f"light frames -> {args.output}")
    for w in stats.get("warnings", []):
        print(f"WARNING: {w}")
    if stats.get("n_quality_rejected"):
        print(f"Quality filter rejected {stats['n_quality_rejected']} frame(s)")
    if stats.get("n_trail_rejected"):
        print(f"Trail rejection dropped {stats['n_trail_rejected']} frame(s)")
    if stats.get("denoised"):
        print(f"Denoised with {stats.get('denoise_backend')} "
              f"-> {args.output}/stacked_denoised.fits")
    if stats.get("n_lights", 0) == 0:
        return 1
    return 0


def cmd_ask(args: argparse.Namespace) -> int:
    """Build an imaging plan from natural language."""
    print(f"Planning: {args.text!r}")
    print(f"({ai_assistant.describe_source(args.text, use_llm=not args.no_llm)})")
    try:
        plan_dict = ai_assistant.plan_dict_from_text(
            args.text, use_llm=not args.no_llm)
        plan = ai_assistant.validate_plan_dict(plan_dict)
    except config.PlanError as exc:
        print(f"INVALID PLAN\n{exc}", file=sys.stderr)
        return 2
    print()
    print(config.plan_summary(plan))
    notes = plan_dict.get("notes")
    if notes:
        print(f"\nAssistant notes: {notes}")
    if args.dry_run:
        print("\n(dry run — no file written)")
        return 0
    out = args.output or f"{plan.session_name}.yaml"
    ai_assistant.write_plan_yaml(plan_dict, out)
    print(f"\nWrote {out} — validate with: "
          f"python -m astrocapture plan --config {out}")
    return 0


def cmd_export_training_data(args: argparse.Namespace) -> int:
    out = export_training_data(args.session_dir, args.output)
    n = len(list((out / "thumbnails").glob("*.png")))
    print(f"Exported {n} thumbnails + labels.csv -> {out}")
    print("Next: fill in the 'label' column (1 = keep, 0 = reject), "
          "then train a QualityModel (see README 'AI layer').")
    return 0


def cmd_eaa(args: argparse.Namespace) -> int:
    return run_eaa(args.config, max_frames=args.frames)


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


def _hms(ra_deg: float) -> str:
    hours = ra_deg / 15.0
    h = int(hours)
    m = int((hours - h) * 60)
    s = (hours - h - m / 60) * 3600
    return f"{h:02d}h {m:02d}m {s:04.1f}s"


def _dms(dec_deg: float) -> str:
    sign = "+" if dec_deg >= 0 else "-"
    a = abs(dec_deg)
    d = int(a)
    m = int((a - d) * 60)
    s = (a - d - m / 60) * 3600
    return f"{sign}{d:02d}° {m:02d}' {s:04.1f}\""


def cmd_catalog(args: argparse.Namespace) -> int:
    try:
        obj = catalog.lookup(args.name)
    except catalog.UnknownObjectError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    mag = f"{obj['mag']:.2f}" if obj["mag"] is not None else "—"
    size = f"{obj['size_arcmin']:.1f}'" if obj["size_arcmin"] is not None else "—"
    const = obj["constellation"] or "—"
    print(f"{obj['ids'][0]} — {obj['name']}")
    print(f"  Type : {obj['type']}   Constellation: {const}")
    print(f"  RA   : {obj['ra']:.4f}° ({_hms(obj['ra'])})   "
          f"Dec: {obj['dec']:+.4f}° ({_dms(obj['dec'])})  (J2000)")
    print(f"  Mag  : {mag}   Size: {size}")
    print(f"  IDs  : {', '.join(obj['ids'])}")
    return 0


def cmd_tonight(args: argparse.Namespace) -> int:
    when = None
    if args.date:
        try:
            when = datetime.strptime(args.date, "%Y-%m-%d").replace(
                hour=12, tzinfo=timezone.utc)
        except ValueError:
            print(f"ERROR: --date must be YYYY-MM-DD, got {args.date!r}",
                  file=sys.stderr)
            return 2
    try:
        best = catalog.tonight_best(
            args.lat, args.lon, when=when,
            min_alt_deg=args.min_alt, limit=args.limit,
        )
    except (ValueError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    if not best:
        print(f"No catalog objects reach {args.min_alt:g}° tonight.")
        return 0
    date_lbl = args.date or "tonight"
    print(f"Tonight's best ({date_lbl}, {args.lat:.2f}°, {args.lon:.2f}°) "
          f"— {len(best)} objects ≥ {args.min_alt:g}°:")
    print(f"{'#':>3}  {'Object':26s} {'Type':16s} {'Mag':>5} "
          f"{'Cst':>3} {'Peak':>5} {'Peak(UTC)':>8} {'Hrs':>4}")
    for i, o in enumerate(best, 1):
        label = o["name"]
        if o["ids"] and o["ids"][0] != o["name"]:
            label = f"{label} [{o['ids'][0]}]"
        mag = f"{o['mag']:.1f}" if o["mag"] is not None else "—"
        peak_t = o["peak_time_utc"][11:16]  # HH:MM of the ISO timestamp
        print(f"{i:>3}  {label[:26]:26s} {o['type'][:16]:16s} {mag:>5} "
              f"{(o['constellation'] or '—'):>3} {o['peak_alt_deg']:>5.1f} "
              f"{peak_t:>8} {o['hours_above']:>4.1f}")
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
    sd = sub.add_parser("dash", help="run a session with the live web dashboard")
    sd.add_argument("--config", required=True, help="plan YAML to run")
    sd.add_argument("--port", type=int, default=8765,
                    help="dashboard port (default: 8765)")
    sd.set_defaults(func=cmd_dash)
    sn = sub.add_parser("night", help="run a multi-target night queue until dawn")
    sn.add_argument("--config", required=True,
                    help="plan YAML with a night: block "
                         "(e.g. examples/night_queue.yaml)")
    sn.set_defaults(func=cmd_night)
    sc = sub.add_parser("catalog", help="look up a deep-sky object by name")
    sc.add_argument("name", help='object name, e.g. "M51", "NGC 7000"')
    sc.set_defaults(func=cmd_catalog)
    st = sub.add_parser("tonight", help="rank tonight's best deep-sky targets")
    st.add_argument("--lat", type=float, required=True,
                    help="observer latitude (deg)")
    st.add_argument("--lon", type=float, required=True,
                    help="observer longitude (deg)")
    st.add_argument("--date", default=None,
                    help="observing date YYYY-MM-DD (default: today)")
    st.add_argument("--limit", type=int, default=20, help="max objects to list")
    st.add_argument("--min-alt", type=float, default=30.0,
                    help="minimum altitude in degrees")
    st.set_defaults(func=cmd_tonight)
    sp2 = sub.add_parser(
        "process",
        help="calibrate and stack a finished session directory",
    )
    sp2.add_argument("session_dir", help="session directory to process")
    sp2.add_argument("--output", default="processed",
                     help="output directory (default: processed)")
    sp2.add_argument("--min-quality", type=float, default=None,
                     help="drop light frames scoring below this 0..1 "
                          "heuristic quality (default: off)")
    sp2.add_argument("--denoise", action="store_true",
                     help="denoise the final stack (classical bilateral "
                          "filter, no downloads)")
    sp2.add_argument("--denoise-model", default=None,
                     help="ONNX denoise model path (needs onnxruntime)")
    sp2.add_argument("--denoise-providers", default=None,
                     help="comma-separated ONNX providers, e.g. "
                          "CoreMLExecutionProvider,CPUExecutionProvider")
    sp2.add_argument("--trails", default="off",
                     choices=["off", "reject", "mask"],
                     help="satellite/airplane trail handling "
                          "(default: off)")
    sp2.set_defaults(func=cmd_process)
    sa = sub.add_parser(
        "ask",
        help="build an imaging plan from natural language",
    )
    sa.add_argument("text",
                    help='e.g. "image M51 tonight, 2 hours of data"')
    sa.add_argument("--dry-run", action="store_true",
                    help="print the plan without writing a file")
    sa.add_argument("--output", default=None,
                    help="plan YAML path (default: <session-name>.yaml)")
    sa.add_argument("--no-llm", action="store_true",
                    help="force the offline rule-based parser")
    sa.set_defaults(func=cmd_ask)
    se2 = sub.add_parser(
        "export-training-data",
        help="export frame thumbnails + label CSV for training a "
             "future quality CNN",
    )
    se2.add_argument("session_dir", help="session directory to export")
    se2.add_argument("--output", default="training",
                     help="output directory (default: training)")
    se2.set_defaults(func=cmd_export_training_data)
    se = sub.add_parser(
        "eaa",
        help="live-stack a plan's light frames (EAA mode)",
    )
    se.add_argument("--config", required=True, help="plan YAML to run")
    se.add_argument("--frames", type=int, default=None,
                    help="max light frames to stack (default: all)")
    se.set_defaults(func=cmd_eaa)
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
