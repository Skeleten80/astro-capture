# AstroCapture

Telescope + camera control and automated capture sequencing for
astrophotography. You write an imaging plan in YAML; AstroCapture slews
the mount, runs the exposure sequence (lights / darks / flats / bias),
dithers between frames, and saves calibrated FITS files with proper
headers into a timestamped session directory.

**Status: v0.1.0 scaffold. The simulator runs end-to-end today. The INDI
backend is a pure-Python protocol client (no compiled dependencies —
runs on Linux, macOS and Windows) but has NOT been tested against real
hardware** — see [Caveats](#caveats).

## Quick start (no hardware needed)

```bash
cd astro-capture
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Validate the example plan
python -m astrocapture plan --config examples/sim_session.yaml

# Run a full simulated session (~30 s): slew, 4 lights + dither,
# 2 darks, 2 flats, 3 biases, park. Output lands in sessions/
python -m astrocapture --config examples/sim_session.yaml

# What drivers are available on this machine?
python -m astrocapture --list-drivers
```

## Install

**Recommended architecture:** `indiserver` runs on a small Linux box at
the scope (a Raspberry Pi is the classic choice — it holds the USB
cables to the mount and camera). AstroCapture runs on whatever laptop
you have and talks to it over the network, since INDI is a network
protocol:

```
  [laptop: macOS / Windows / Linux] --TCP 7624--> [Pi at the scope]
   AstroCapture (pure-Python            indiserver + drivers
   INDI client, zero compiled deps)     (indi_celestron_gps, indi_gphoto_cc, ...)
```

On the Pi (Ubuntu/Debian):

```bash
# INDI server + drivers
sudo apt install indi-bin
#   Celestron NexStar: sudo apt install indi-bin   # indi_celestron_gps ships with indi-bin
#   EQMod mounts:      sudo apt install indi-eqmod
#   ZWO cameras:       sudo apt install indi-asi
#   DSLR via INDI:     sudo apt install indi-gphoto

# DSLR USB support (also needed by indi-gphoto)
sudo apt install libgphoto2-6
```

On the laptop (Linux, macOS Intel/Apple Silicon, or Windows 11):

```bash
pip install -r requirements.txt
# that's it for driver: indi — the INDI client is pure Python (stdlib
# socket + XML), no PyIndi-Client, no compiler needed.
pip install gphoto2   # only needed for driver: dslr (direct USB, Linux/macOS)
```

## Pointing it at real hardware

1. Start the INDI server with the drivers for *your* gear (on the Pi,
   or on the laptop itself if it's Linux):
   ```bash
   indiserver indi_celestron_gps indi_gphoto_cc -p 7624 &
   ```
2. List the exact device names the server exposes:
   ```bash
   indi_getprop | grep -E "CONNECTION" | head
   ```
   (Device names look like `"Celestron GPS"`, `"Canon DSLR"` — copy
   them exactly, quotes and all.)
3. Copy `examples/indi_session.yaml`, fill in your device names,
   telescope/camera strings, the server's `host`/`port`, and the
   exposure plan, then:
   ```bash
   python -m astrocapture --config my_session.yaml
   ```

AstroCapture's INDI backend (`drivers/indi.py`) speaks the INDI v1.7
XML protocol directly over the socket — `<getProperties>`,
`newNumberVector`/`newSwitchVector`/`newTextVector` commands,
`setXXXVector` state updates, and base64 BLOB frames after
`<enableBLOB>`. No PyIndi-Client needed (it currently fails to build
against modern libindi anyway), so the client runs anywhere Python
runs.

### DSLR-over-USB notes

Two routes, in order of preference:

1. **INDI (`driver: indi`, device `indi_gphoto_cc`)** — preferred. The
   INDI driver handles bulb exposures, ISO, live view, and wraps frames
   as FITS for you. AstroCapture treats it like any other camera.
2. **Direct gphoto2 (`driver: dslr`)** — no INDI server in the loop. This
   backend is a *sketch*: standard exposures work, but bulb capture
   (>30 s) is model-specific (Canon `eosremoterelease`, Nikon/Sony
   differ) and `abort_exposure` is not implemented. Good starting point
   if you want to bypass INDI; not recommended for unattended all-night
   runs.

Either way: set your DSLR to **manual mode, manual focus, RAW**, disable
auto power-off and in-camera long-exposure noise reduction (you take
darks for that), and use a dummy battery / AC adapter.

## Mathias's setup (NexStar 6SE + Rebel T7i)

`examples/mathias_6se_t7i.yaml` is a ready-to-edit first-light plan for
this exact gear — 30× 25 s lights at ISO 1600 on M13, plus matching
darks/flats/biases.

**Physical chain**

- 12V power → mount (its own supply; the hand controller draws from it).
- USB cable (mini-USB) — or a USB-to-serial adapter — from the port on the
  **base of the hand controller** → laptop. 9600 baud; add yourself to the
  `dialout` group: `sudo usermod -aG dialout $USER` (log out/in after).
- Camera USB → laptop (separate cable; don't run the camera through a hub
  with the mount if you can avoid it).
- Camera → telescope: T-ring (Canon EF) + 1.25″ nosepiece, or an SCT
  T-adapter threaded straight onto the 6SE's rear cell. Diagonal out.

**Software**

```bash
sudo apt install indi-bin indi-gphoto libgphoto2-6
# indi_celestron_gps ships with indi-bin

indiserver indi_celestron_gps indi_gphoto_cc -p 7624 &
indi_getprop   # confirm device names: "Celestron GPS", "Canon DSLR"
ls /dev/ttyUSB* /dev/ttyACM*   # your hand-controller serial port

# edit examples/mathias_6se_t7i.yaml (serial_port!), then:
python -m astrocapture plan --config examples/mathias_6se_t7i.yaml
python -m astrocapture --config examples/mathias_6se_t7i.yaml
```

**First-light checklist**

1. Power on the mount and **star-align from the hand controller first**
   (SkyAlign / auto two-star). INDI cannot do the initial alignment —
   connect only after the handbox reports a successful alignment.
2. Camera: M mode, manual focus, RAW only, auto power-off OFF, mirror
   lockup ON, long-exposure NR OFF (full checklist: `T7I_PROFILE` in
   `astrocapture/drivers/dslr.py`).
3. Focus with Live View at 10× on a bright star before the run.
4. Take 2–3 short test exposures and check the frame before committing
   to the full 30-frame sequence.

**The honest caveat:** the 6SE is an **alt-az** fork mount, so the sky
rotates in the frame (field rotation). Without an equatorial wedge, keep
subs to roughly **20–30 s** depending on target declination — longer and
stars trail at the frame edges even with perfect tracking. Real
long-exposure deep sky wants a wedge or an equatorial mount. The
sequencer works either way; when you get a wedge or EQ mount, just raise
the exposure times in the plan.

## Autonomous imaging

Five optional plan blocks turn a basic capture run into an unattended
session. All are off by default — a plan without them behaves exactly as
before. See `examples/mathias_6se_t7i.yaml` for commented examples with
honest prerequisites.

**1. Target catalog.** `astrocapture/data/catalog.json` vendors **5,045**
deep-sky objects (**110** Messier, **109** Caldwell, plus NGC/IC and
more) with J2000 coordinates. `target: {name: "M51"}` resolves RA/Dec
automatically at plan load; `python -m astrocapture catalog "M51"` looks
up an object and `tonight --lat/--lon` ranks what's well placed.

**2. Plate-solve recentering.** A per-target `platesolve:` block runs a
closed loop after each slew: short exposure → local astrometry.net
`solve-field` → slew to the error-corrected coordinates → re-solve,
until the residual is within `tolerance_arcmin` (or `max_iterations`
runs out). A missing `solve-field` binary is *inconclusive, not failed*:
the block logs "skipped" and imaging continues; only a solver that runs
but can't converge counts toward the watchdog's failure budget.

**3. Autofocus.** A top-level `autofocus:` block runs a V-curve autofocus
(sweep the focuser, measure median HFR per position, fit a parabola,
move to the vertex) before the first light frame, every `every_minutes`,
and whenever the median light-frame HFR degrades by
`hfr_degradation_trigger` vs the post-focus baseline. The honest note:
the stock NexStar 6SE has **no motorized focuser**, so with no focuser
device configured the sequencer prints the Bahtinov-mask manual focusing
guide once at session start and continues (manual-assist mode) instead of
pretending to autofocus.

**4. PHD2 guiding.** A top-level `guiding:` block connects to PHD2's
JSON-over-TCP API at session start, starts guiding after the first slew,
and replaces the blind timed dither with `dither()` + a real settle
handshake. If PHD2 is unreachable the sequencer logs a warning and falls
back to unguided blind dithers — never a crash. A lost guide star goes
through the watchdog's retry budget before parking.

**5. Safety watchdog.** Every run gets a `Watchdog`: consecutive
plate-solve failures park the mount and stop the sequence; an exhausted
guide-star retry budget does the same; any unexpected exception parks
the mount, releases the camera, and alerts (swallow-after-parking —
unattended hardware in an unknown state gets parked, not debugged);
`max_session_hours` stops the run gracefully on a wall-clock limit.
Alerts go to `session.log`, or POST to a webhook (`alerts.webhook_url`).

Watch it live:

```bash
python -m astrocapture dash --config examples/sim_session.yaml --port 8765
# then open http://localhost:8765 — live frame feed, FITS thumbnails,
# session log tail, and per-frame stats as the run progresses
```

## How it's built

```
astrocapture/
  drivers/
    base.py    # Mount + Camera abstract interfaces
    sim.py     # SimMount (great-circle slews) + SimCamera (synthetic starfield)
    indi.py    # pure-Python INDI v1.7 protocol client (stdlib only:
               # socket + XML, no PyIndi-Client, works on Linux/macOS/Windows)
    dslr.py    # gphoto2 sketch (import guarded)
  config.py    # plan YAML loading + validation (fails fast, lists all errors)
  sequencer.py # state machine: SLEWING -> EXPOSING -> DITHERING ... DONE
               # thread-safe pause()/resume()/abort(); wires platesolve,
               # autofocus, PHD2 guiding, and the safety watchdog
  session.py   # session dirs, FITS writer, dither/plate-solve/meridian hooks
  catalog.py   # vendored night-sky catalog (5,045 objects; name lookup)
  platesolve.py# solve-field wrapper + closed-loop recenter(); FakeSolver
  focus.py     # HFR measurement, V-curve autofocus, SimFocuser/INDIFocuser,
               # manual Bahtinov-mask assist_mode()
  phd2.py      # PHD2 JSON-over-TCP client (guiding, dither + settle)
  safety.py    # Watchdog (park/stop/alert policy), AlertLog, WebhookAlert
  dash.py      # live web dashboard (frame feed, thumbnails, log tail)
  cli.py       # python -m astrocapture
examples/
  sim_session.yaml    # runs with zero hardware
  indi_session.yaml   # template for real gear (fill in YOUR device names)
  mathias_6se_t7i.yaml  # NexStar 6SE + Rebel T7i first-light plan
tests/                # pytest tests, all passing
```

**Design rules:** the sequencer only talks to the abstract `Mount` /
`Camera`, so a new backend (ASCOM Alpaca, a vendor SDK) is one new file
plus a registry entry. The INDI backend has zero optional dependencies
— `import astrocapture` works on any machine, and only the
direct-gphoto2 sketch raises a clear error when its package is missing.

**Simulator fidelity notes:** `SimMount` slews along the great circle at
a configurable deg/sec (verified against slew-rate math in tests).
`SimCamera` renders a seeded star catalog with Gaussian PSFs, photon
shot noise, read noise, hot pixels, and exposure-scaled counts — frames
stack plausibly. Dithering shifts the catalog so hot pixels don't stack
coherently, like the real thing.

## Session output

```
sessions/m51-sim-20261002-172424/
  session.log          # every action, timestamped
  lights/  darks/  flats/  bias/
    M51_light_L_001.fits ...
```

FITS headers include `OBJECT`, `RA`/`DEC` (J2000 deg), `EXPTIME`,
`IMAGETYP`, `DATE-OBS`, `INSTRUME`, `TELESCOP`, `FILTER`, `GAIN`/`ISO`,
`XBINNING`/`YBINNING`, `CCD-TEMP`, `BUNIT=ADU`.

## Caveats

- **Not yet tested against real hardware.** The INDI backend speaks the
  INDI v1.7 XML protocol directly (verified against the spec, and
  exercised end-to-end against a scripted fake server in
  `tests/test_indi_proto.py`), using standard INDI property names
  (`EQUATORIAL_EOD_COORD`, `TELESCOPE_PARK`, `CCD_EXPOSURE`, …) — but
  property names vary between drivers, so expect a debugging session
  with `indi_getprop` on first light. Known variation points are marked
  in `drivers/indi.py` comments. (The old PyIndi-Client approach was
  dropped because that package currently fails to build against modern
  libindi.)
- The gphoto2 backend's bulb path is per-model and untested.
- Meridian-flip *detection* exists (`check_meridian_flip`); the flip
  itself currently pauses the sequence for you to flip manually.
- Plate solving shells out to a local `solve-field`; without
  astrometry.net installed the per-target recenter block logs "skipped"
  and continues (a missing binary is inconclusive, never a failure).
- The PHD2 client speaks the documented JSON-over-TCP API but has not
  been exercised against a live PHD2 yet; guiding paths are covered by
  test doubles in `tests/test_sequencer_integration.py`.

## Roadmap

- [x] Autoguiding via the PHD2 API (dither handshake + settle, instead
      of the timed settle used now)
- [x] Autofocus V-curve routine (sweep focuser, fit HFR curve, move to
      best focus; temperature-compensation hooks)
- [x] Plate-solve re-centering loop: solve → slew to correct residual →
      re-solve until within tolerance
- [ ] Automated meridian flip: re-slew, re-center, resume guiding
- [ ] ASCOM Alpaca backend (Windows/remote-driver option)
- [ ] First-light test log against real mount + camera

## Tests

```bash
.venv/bin/python -m pytest tests/ -q
# 108 passed
```

Covers: plan validation (incl. multi-error reporting and the new
autonomous-imaging blocks), sim slew math (monotonic approach, rate
timing, RA wrap, park), synthetic image properties (shape/dtype,
exposure scaling, dither shift, FITS round-trip), sequencer state
machine (full run, pause/resume, abort), FITS header contents, the
pure-Python INDI client (handshake, goto RA/Dec XML, park switch, slew
Busy→Ok, exposure → real FITS bytes via a scripted fake INDI server),
the night-sky catalog (5,045 objects, lookup, tonight ranking), HFR
measurement + V-curve autofocus (SimFocuser, INDIFocuser against a fake
server, manual assist mode), plate solving + closed-loop recentering
(FakeSolver), the PHD2 client (protocol framing, dither/settle,
star-lost), the safety watchdog (solve-failure parking, guide retry
budget, exceptions, session limits, webhooks), and the sequencer
integration of all five (platesolve success/failure, multi-target runs,
autofocus, PHD2 fallback, guide-lost parking, exception parking, config
validation).
