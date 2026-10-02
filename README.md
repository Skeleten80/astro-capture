# AstroCapture

Telescope + camera control and automated capture sequencing for
astrophotography. You write an imaging plan in YAML; AstroCapture slews
the mount, runs the exposure sequence (lights / darks / flats / bias),
dithers between frames, and saves calibrated FITS files with proper
headers into a timestamped session directory.

**Status: v0.1.0 scaffold. The simulator runs end-to-end today. The INDI
and DSLR backends are written but have NOT been tested against real
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

## Install (Ubuntu, for real hardware)

```bash
# INDI server + drivers
sudo apt install indi-bin
#   EQMod mounts:      sudo apt install indi-eqmod
#   ZWO cameras:       sudo apt install indi-asi
#   DSLR via INDI:     sudo apt install indi-gphoto

# DSLR USB support (also needed by indi-gphoto)
sudo apt install libgphoto2-6

# Python side
pip install -r requirements.txt
pip install PyIndi-Client   # only needed for driver: indi
pip install gphoto2         # only needed for driver: dslr (direct USB)
```

## Pointing it at real hardware

1. Start the INDI server with the drivers for *your* gear:
   ```bash
   indiserver indi_eqmod_telescope indi_gphoto_cc -p 7624 &
   ```
2. List the exact device names the server exposes:
   ```bash
   indi_getprop | grep -E "CONNECTION" | head
   ```
   (Device names look like `"EQMod Mount"`, `"Canon DSLR"` — copy them
   exactly, quotes and all.)
3. Copy `examples/indi_session.yaml`, fill in your device names,
   telescope/camera strings, and exposure plan, then:
   ```bash
   python -m astrocapture --config my_session.yaml
   ```

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

## How it's built

```
astrocapture/
  drivers/
    base.py    # Mount + Camera abstract interfaces
    sim.py     # SimMount (great-circle slews) + SimCamera (synthetic starfield)
    indi.py    # PyIndi-Client backend (import guarded)
    dslr.py    # gphoto2 sketch (import guarded)
  config.py    # plan YAML loading + validation (fails fast, lists all errors)
  sequencer.py # state machine: SLEWING -> EXPOSING -> DITHERING ... DONE
               # thread-safe pause()/resume()/abort()
  session.py   # session dirs, FITS writer, dither/plate-solve/meridian hooks
  cli.py       # python -m astrocapture
examples/
  sim_session.yaml    # runs with zero hardware
  indi_session.yaml   # template for real gear (fill in YOUR device names)
  mathias_6se_t7i.yaml  # NexStar 6SE + Rebel T7i first-light plan
tests/                # pytest tests, all passing
```

**Design rules:** the sequencer only talks to the abstract `Mount` /
`Camera`, so a new backend (ASCOM Alpaca, a vendor SDK) is one new file
plus a registry entry. Optional dependencies are guarded — `import
astrocapture` works on any machine; you get a clear error only when you
*select* a backend whose package is missing.

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

- **Not yet tested against real hardware.** The INDI backend is written
  against the PyIndi-Client API and standard INDI property names
  (`EQUATORIAL_EOD_COORD`, `TELESCOPE_PARK`, `CCD_EXPOSURE`, …), but
  property names vary between drivers — expect a debugging session with
  `indi_getprop` on first light. Known variation points are marked in
  `drivers/indi.py` comments.
- The gphoto2 backend's bulb path is per-model and untested.
- Meridian-flip *detection* exists (`check_meridian_flip`); the flip
  itself currently pauses the sequence for you to flip manually.
- Plate solving shells out to a local `solve-field`; without
  astrometry.net installed it logs "skipped" and continues.

## Roadmap

- [ ] Autoguiding via the PHD2 API (dither handshake + settle, instead
      of the timed settle used now)
- [ ] Autofocus V-curve routine (sweep focuser, fit HFR curve, move to
      best focus; temperature-compensation hooks)
- [ ] Plate-solve re-centering loop: solve → slew to correct residual →
      re-solve until within tolerance
- [ ] Automated meridian flip: re-slew, re-center, resume guiding
- [ ] ASCOM Alpaca backend (Windows/remote-driver option)
- [ ] First-light test log against real mount + camera

## Tests

```bash
.venv/bin/python -m pytest tests/ -q
# 36 passed
```

Covers: plan validation (incl. multi-error reporting), sim slew math
(monotonic approach, rate timing, RA wrap, park), synthetic image
properties (shape/dtype, exposure scaling, dither shift, FITS
round-trip), sequencer state machine (full run, pause/resume, abort),
and FITS header contents.
