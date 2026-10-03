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

**6. Dew heater control.** A top-level `dew:` block keeps the corrector
plate above the dew point — the #1 night-killer for an SCT in humid
Ontario. Once per light frame the controller reads ambient temperature
and humidity, computes the dew point (Magnus formula), and ramps heater
duty with a proportional law as the air approaches your `margin_c`
safety margin. A sensor or heater fault logs a warning and holds the
last duty — dew control can **never** abort imaging. Real use needs
physical hardware (12 V dew strap + PWM controller, driven via INDI);
the INDI device/property names are configurable and must be verified
with `indi_getprop` on first light. Try the scripted humid night in sim:
`examples/dew_demo.yaml`.

**7. Night scheduler.** `python -m astrocapture night --config
examples/night_queue.yaml` works your `targets:` list unattended until
astronomical dawn. Each target gets the plan's full light sequence as
its quota; the scheduler continuously picks the observable target
scoring highest on priority × altitude × Moon separation, runs it in its
own session directory, then moves on. It won't start a new target when
less than one frame plus overhead remains before dawn, parks the mount
if anything aborts mid-target, and writes `night_summary.md` (per-target
frames, statuses, failures, watchdog alerts). Configure with the
`night:` block (site coordinates, minimum altitude / Moon separation);
`time_accel` fast-forwards waits in sim demos.

Watch it live:

```bash
python -m astrocapture dash --config examples/sim_session.yaml --port 8765
# then open http://localhost:8765 — live frame feed, FITS thumbnails,
# session log tail, and per-frame stats as the run progresses
```

## From photons to picture (processing + live stacking)

Two post-capture workflows close the loop from FITS files to finished
image. Both run on any platform — no hardware needed.

**Calibrate + stack.** `python -m astrocapture process
sessions/<name>-<stamp> --output processed/` builds master
bias/dark/flat frames (sigma-clipped median combine; darks are
exposure-scaled when they don't match the lights, with a logged note
that scaling is approximate), calibrates each light as
`(data − bias − dark) / flat`, registers frames by star-matching, and
sigma-clips them into a single `stacked.fits` (32-bit float, with a
header recording which masters were used) plus an auto-stretched
`stacked.png` preview and a `process.log` with per-frame shifts and
rejection counts. Registration is translation-only — fine for short
alt-az subs; field rotation is *not* corrected (a known follow-up).
Missing or unusable calibration frames produce a warning, never a
crash (the sim's flats are short starfield exposures, so the pipeline
detects the degenerate flat and skips flat-fielding with a clear log
line instead of dividing by nonsense).

**Live stacking (EAA mode).** `python -m astrocapture eaa --config
examples/sim_session.yaml --frames 6` captures light frames and stacks
them as they arrive, printing a running SNR estimate — great for
outreach nights and for checking data quality while you image. Every
frame registers against the *first* frame (never the running stack, so
alignment can't drift). The web dashboard gains a **Live** panel
(`/api/live.png`) showing the accumulating stack during any session.

## AI layer

Four optional AI-flavored features. The honest summary up front: **no
trained models are shipped** — the LLM path needs your own
`OPENAI_API_KEY`, the ONNX denoise slot accepts a model file *you*
supply, and everywhere a learned model would go there is a classical
fallback that works out of the box. What's real is documented below;
what's stubbed says so.

**1. Natural-language planning** (`astrocapture/ai_assistant.py`).
`python -m astrocapture ask "image the Whirlpool Galaxy tonight, 2 hours
of data" [--dry-run]` turns English into a validated plan YAML. With
`OPENAI_API_KEY` set (and the optional `openai` package installed) an
LLM drafts the plan under a system prompt that encodes your exact rig —
6SE alt-az 25 s sub cap, T7i ISO 1600, dither every 3, matching
calibration frames — and must return STRICT JSON, which is then
schema-checked *and* run through the real plan validator (target names
resolve through the night-sky catalog). Raw LLM output is never
trusted. Without a key, the offline rule-based parser takes over:
catalog designation/common-name lookup, "2 hours" → frame counts, sub
lengths clamped to 25 s.

**2. Bad-frame quality scoring** (`astrocapture/quality.py`). Heuristic
v1 — explicitly *not* a neural net: per-frame scores 0..1 from star
trailing (median stellar eccentricity via second moments), cloud/haze
(background + noise drift vs. the session baseline), and focus softness
(median HFR vs. the session's best). The overall score is the geometric
mean, so one bad axis tanks the frame. `python -m astrocapture process
sessions/<name> --min-quality 0.5` drops low scorers before stacking
(reported per-frame in `process.log`). `python -m astrocapture
export-training-data sessions/<name> --output training/` writes frame
thumbnails plus a `labels.csv` with an empty `label` column — fill it in
(1 = keep, 0 = reject) and you have the dataset for the small CNN that
will one day implement the `QualityModel` interface (`score(frame) ->
float`); the heuristic columns ship as free input features.

**3. AI denoise, classical fallback** (`astrocapture/denoise.py`).
`python -m astrocapture process sessions/<name> --denoise` runs an
edge-aware bilateral filter (pure numpy, measurably improves SNR) as the
final step, writing `stacked_denoised.fits/png`. `--denoise-model
model.onnx [--denoise-providers
CoreMLExecutionProvider,CPUExecutionProvider]` instead runs a
user-supplied ONNX model via onnxruntime (guarded import — a clear
error if it's not installed); the CoreML provider dispatches to the
Neural Engine on Apple Silicon, the same pattern as the car-logger
vision stack. We ship no trained astro denoise model; the ONNX slot is
ready for community models.

**4. Satellite/airplane trail detection** (`astrocapture/trails.py`).
Numpy-only: stars are masked (round detections only — the trail's own
flat-topped maxima are eccentricity-gated so the trail can't mask
itself), remaining bright pixels go through RANSAC line voting, and a
candidate counts only if its *longest continuous run* spans ≥ 60 px —
random star cores that merely align leave hundred-pixel gaps and are
rejected. `python -m astrocapture process sessions/<name> --trails
reject` drops trailed frames; `--trails mask` inpaints trail pixels
with the frame median and keeps the frame. Counts land in `process.log`.

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
  dew.py       # dew-point heater control (Magnus law; sim + INDI backends)
  scheduler.py # multi-target night queue until astronomical dawn
  imaging.py   # star centroids, translation registration, FITS stretch
  process.py   # calibration (bias/dark/flat) + sigma-clipped stacking
  eaa.py       # live stacking (EAA mode) with running SNR estimate
  dash.py      # live web dashboard (frame feed, thumbnails, log tail,
               # Live-stack panel)
  cli.py       # python -m astrocapture
examples/
  sim_session.yaml    # runs with zero hardware
  indi_session.yaml   # template for real gear (fill in YOUR device names)
  mathias_6se_t7i.yaml  # NexStar 6SE + Rebel T7i first-light plan
  night_queue.yaml    # multi-target scheduler demo (sim, fast)
  dew_demo.yaml       # scripted humid night for the dew controller (sim)
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
- [x] Calibration + stacking pipeline: masters, registration,
      sigma-clipped stack, `astrocapture process`
- [x] Live stacking / EAA mode with dashboard Live panel
- [x] Dew heater control (dew-point proportional law; needs strap hardware)
- [x] Night scheduler: multi-target queue until astronomical dawn
- [x] AI layer: natural-language planning (LLM + offline fallback),
      heuristic bad-frame scoring + training-data export, classical/ONNX
      denoise, satellite-trail detection
- [ ] Automated meridian flip: re-slew, re-center, resume guiding
- [ ] Field-rotation-aware registration (for longer alt-az subs)
- [ ] Learned quality CNN trained on exported labels (plugs into
      `quality.QualityModel`)
- [ ] ASCOM Alpaca backend (Windows/remote-driver option)
- [ ] First-light test log against real mount + camera

## Tests

```bash
.venv/bin/python -m pytest tests/ -q
# 208 passed, 1 skipped (onnxruntime-present variant; onnxruntime not installed)
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
validation), plus the AI layer: LLM planning (mocked client, strict
JSON, sub-cap enforcement, fence stripping), rule-based planning
(designations, common names, time parsing, sub clamping, offline
fallback), heuristic quality scoring (sharp/trailed/cloudy ordering,
unit range, training export), classical denoise (SNR gain, edge
preservation, ONNX error paths), and trail detection (angles, clean
frames, short-line and aligned-clump rejection, process
reject/mask integration).
