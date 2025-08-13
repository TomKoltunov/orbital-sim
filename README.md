# orbital-sim — Constellation Ground-Track & Access Planner

orbital-sim is a pure-Python simulation and visualization tool for low- to medium-Earth-orbit satellite constellations.
It propagates orbits using two-body Keplerian motion, computes satellite ground tracks, calculates access windows to ground stations above a minimum elevation, and produces visualizations of both.

## Features

* Orbit propagation using two-body Keplerian dynamics (no external astrodynamics libraries).
* ECI ↔ ECEF conversions and geodetic coordinate calculations.
* Ground track plotting for one or more satellites.
* Access window computation for each satellite–ground-station pair.
* Timeline (Gantt-like) plotting of access windows.
* Minimal dependencies: numpy + matplotlib.

## Project Structure

```
main.py              # Main simulation logic & CLI entry point
scenario_demo.json   # Example scenario file (satellites + ground stations)
```

Key parts inside `main.py`:

* Data classes: `Satellite`, `GroundStation`, `Scenario`
* Orbital mechanics: Kepler solver, perifocal–ECI transforms
* Coordinate systems: ECI ↔ ECEF, geodetic ↔ ECEF, ENU transforms
* Access analysis: Computes above-horizon visibility windows
* Plotting: Ground track map + access window timeline

## Installation

Python 3.8+ is recommended.

```bash
git clone https://github.com/yourusername/orbital-sim.git
cd orbital-sim
pip install -r requirements.txt
```

If you do not have a `requirements.txt`, install the dependencies manually:

```bash
pip install numpy matplotlib
```

## Scenario File Format

Scenarios are stored as JSON and describe:

* Simulation timing and resolution
* Minimum elevation threshold
* A list of satellites with Keplerian elements
* A list of ground stations with positions

Example:

```json
{
  "t_start_utc": "2025-07-25T00:00:00Z",
  "duration_seconds": 7200,
  "time_step_seconds": 30,
  "min_elevation_deg": 10,
  "satellites": [
    {
      "name": "SAT-1",
      "a_km": 6871,
      "e": 0.001,
      "i_deg": 98.7,
      "raan_deg": 0,
      "argp_deg": 0,
      "nu0_deg": 0
    }
  ],
  "ground_stations": [
    {
      "name": "GS-1",
      "lat_deg": 32.0,
      "lon_deg": 34.8,
      "height_m": 50
    }
  ]
}
```

## Usage

Run the simulation from the command line:

```bash
python main.py --scenario scenario_demo.json --plot
```

**Options:**

* **`--scenario PATH`** (required) — path to your scenario JSON file.
* **`--plot`** — display plots interactively after the simulation.
* **`--save-plots DIR`** — save plots as PNG files in the specified directory.

**Examples:**

```bash
# Show plots for demo scenario
python main.py --scenario scenario_demo.json --plot

# Save plots instead of showing them
python main.py --scenario scenario_demo.json --save-plots outputs/

# Both save and show
python main.py --scenario scenario_demo.json --plot --save-plots outputs/
```

```first time running the code
python -m venv venv
source venv/bin/activate   # On macOS/Linux
venv\Scripts\activate      # On Windows
pip install -r requirements.txt
python main.py --scenario scenario_demo.json --plot


## Output

1. **Console** — Prints access windows for each satellite and ground station:

```
=== ACCESS WINDOWS ===

Ground station: GS-1
  SAT-1: 2025-07-25T00:15:00+00:00  -->  2025-07-25T00:25:00+00:00  (duration 600 s)
```

2. **Plots:**

   * **Ground Tracks**: Lat/Lon projection of satellite paths.
   * **Access Timeline**: Horizontal bars for visibility windows.

## How It Works

1. **Scenario loading**
   Parses scenario JSON into Python data classes.

2. **Time grid creation**
   Generates evenly spaced UTC timestamps.

3. **Orbit propagation**
   Uses mean motion, Kepler’s equation, and perifocal → ECI transforms.

4. **Frame conversions**
   ECI → ECEF → geodetic coordinates for plotting.

5. **Access computation**
   For each ground station, calculates elevation angles and detects above-threshold intervals.

6. **Plotting**
   Generates ground track and access timeline plots with matplotlib.

## Notes & Assumptions

* Earth is modeled as a perfect sphere (`R_EARTH`), no oblateness (J2) effects.
* Pure two-body dynamics — no atmospheric drag, SRP, or perturbations.
* No leap-second or high-precision Earth orientation corrections.
* All angles in scenario files are in degrees; internal calculations use radians.


## Author

Developed as an educational and planning tool for satellite visibility studies.
Ideal for learning orbital mechanics, ground track visualization, and basic satellite network analysis.
