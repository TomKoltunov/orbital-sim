#!/usr/bin/env python3
"""
הקובץ הראשי שמריץ את התוכנית — כל הלוגיקה והאלגוריתמיקה נמצאים בו
OrbitalSim: Constellation Ground-Track & Access Planner

Pure-Python (numpy + matplotlib) tool to:
- Propagate satellites with 2-body Keplerian motion
- Convert between ECI/ECEF and compute ground tracks
- Compute access windows to ground stations above min elevation
- Plot ground tracks and access timeline
"""

import argparse
import dataclasses
import datetime as dt
import json
import math
import os
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# Constants
# ============================================================

MU_EARTH = 3.986004418e14        # m^3 / s^2
R_EARTH = 6378137.0              # m (spherical Earth for simplicity)
OMEGA_EARTH = 7.2921159e-5       # rad / s (Earth rotation rate)

DEG2RAD = np.pi / 180.0
RAD2DEG = 180.0 / np.pi

# ============================================================
# Data classes
# ============================================================

@dataclasses.dataclass
class Satellite:
    name: str
    a_km: float
    e: float
    i_deg: float
    raan_deg: float
    argp_deg: float
    nu0_deg: float

@dataclasses.dataclass
class GroundStation:
    name: str
    lat_deg: float
    lon_deg: float
    height_m: float = 0.0

@dataclasses.dataclass
class Scenario:
    t_start_utc: dt.datetime
    duration_seconds: float
    time_step_seconds: float
    min_elevation_deg: float
    satellites: List[Satellite]
    ground_stations: List[GroundStation]

# ============================================================
# Utilities
# ============================================================

# מקבלת מחרוזת זמן בפורמט ISO (למשל "2025-07-25T00:00:00Z")
# ומחזירה אובייקט מסוג datetime בפייתון, שמייצג את אותו זמן עם אזור זמן UTC.
def parse_iso_utc(ts: str) -> dt.datetime:
    # Very small parser; expects ...Z -> האות Z היא קיצור ל־Zulu time (כלומר: UTC) לפי תקן ISO-8601.
    if ts.endswith("Z"): # בדיקה: אם המחרוזת מסתיימת ב־Z, אנחנו מורידים את האות הזו (חותכים את התו האחרון).
        ts = ts[:-1]
    # dt.datetime.fromisoformat(ts) — יוצר אובייקט datetime מתוך מחרוזת בפורמט ISO (ללא Z).
    # replace(tzinfo=dt.timezone.utc) — מוסיף מידע של אזור זמן (timezone.utc) לאובייקט.
    # התוצאה הסופית היא: datetime.datetime(2025, 7, 25, 0, 0, tzinfo=datetime.timezone.utc)
    return dt.datetime.fromisoformat(ts).replace(tzinfo=dt.timezone.utc)

def load_scenario(path: str) -> Scenario: # היא מקבלת מחרוזת path, שמייצגת את הנתיב לקובץ JSON.
    with open(path, "r") as f: # פותחים את קובץ הטקסט שצוין (לדוגמה scenario_demo.json) בקריאה ("r").
        # משתמשים ב־json.load(f) כדי לקרוא את תוכנו ולהמיר אותו ל־מילון פייתוני (dict) בשם d.
        # לדוגמה, אם בקובץ יש {"satellites": [...], "ground_stations": [...]}, אז:
        # d["satellites"] → רשימת מילונים של לוויינים
        # d["ground_stations"] → רשימת מילונים של תחנות
        # d["t_start_utc"] → מחרוזת תאריך
        d = json.load(f)

    # עבור כל מילון קטן בתוך d["satellites"] — יוצרים אובייקט מסוג Satellite.
    # הסימון **sd מפרק את המילון לשליחת פרמטרים בשם. לדוגמה: 
    # sd = { "name": "SAT-1", "a_km": 6871, "e": 0.001, ... } Satellite(**sd)
    # → זה בדיוק כמו לכתוב: Satellite(name="SAT-1", a_km=6871, e=0.001, ...)
    # התוצאה: רשימת אובייקטים sats, שכל אחד מהם הוא לוויין עם פרמטרים מסלוליים.
    sats = [Satellite(**sd) for sd in d["satellites"]]
    # אותו הדבר, אבל לתחנות קרקע. 
    # כל מילון מתוך d["ground_stations"] מומר לאובייקט מסוג GroundStation.
    gss  = [GroundStation(**gd) for gd in d["ground_stations"]]

    return Scenario( # מחזירים אובייקט מסוג Scenario, שמאגד את כל הנתונים של התרחיש.
        #  זמן התחלה. קוראים לפונקציה parse_iso_utc(...) שתהפוך את המחרוזת (2025-07-25T00:00:00Z) לאובייקט datetime.
        t_start_utc=parse_iso_utc(d["t_start_utc"]),
        duration_seconds=float(d["duration_seconds"]), # משך ההרצה (שעות → שניות)
        time_step_seconds=float(d["time_step_seconds"]), # שלב זמן בין דגימות
        min_elevation_deg=float(d["min_elevation_deg"]), # זווית גובה מינימלית
        satellites=sats, # רשימת כל הלוויינים
        ground_stations=gss # רשימת כל התחנות הקרקעיות
    )

# ליצור רשימה של זמנים (datetime) בהפרש קבוע לאורך משך זמן נתון
# בדומה לפונקציה np.linspace, אבל במקום מספרים — מדובר כאן על זמנים.
def linspace_times(t0: dt.datetime, duration: float, dt_seconds: float) -> List[dt.datetime]:
    n = int(np.floor(duration / dt_seconds)) + 1
    return [t0 + dt.timedelta(seconds=i * dt_seconds) for i in range(n)]

# הפונקציה מחשבת את התנועה הממוצעת של לוויין במסלול הקפלרי שלו (Keplerian orbit), 
# כלומר כמה רדיאנים לשנייה הוא עובר במסלול — באנגלית: Mean Motion.
def mean_motion(mu: float, a: float) -> float:
    return math.sqrt(mu / a**3)


# הפונקציה מחשבת את האנומליה האקצנטרית E מתוך האנומליה הממוצעת M, 
# עבור לוויין במסלול אליפטי בעל אקסצנטריות e.
# M: מערך של אנומליות ממוצעות  — כלומר מצב הלוויין בזמן נתון (ביחידות של רדיאנים).
# e: האקסצנטריות של המסלול (0 = עיגול, 0.7 = אליפסה משמעותית).
# tol: סף הדיוק לסיום האינטרציה (ברירת מחדל 10^-10)
# max_iter: מספר מקסימלי של איטרציות (ברירת מחדל: 100).
def kepler_E_from_M(M: np.ndarray, e: float, tol: float = 1e-10, max_iter: int = 100) -> np.ndarray:
    # מוודאים ש־M הוא numpy array עם ערכים מסוג float — כדי לתמוך בחישובים ווקטוריים (לא רק עבור ערך בודד).
    M = np.array(M, dtype=np.float64)
    # יוצרים מערך חדש E שישמש בתור האנומליה האקצנטרית שאנחנו מחפשים.
    # מתחילים את האיטרציה בערך ראשוני שהוא שווה ל־M (קירוב טוב).
    E = M.copy()
    # לולאה של עד 100 איטרציות (או כל max_iter) — כאן מתחילה שיטת ניוטון-רפסון.
    for _ in range(max_iter):
        # זו הפונקציה: M - sin(E) * e - E = f(E)
        # אנחנו רוצים למצוא את ה־E שעבורו הפונקציה הזו מתאפסת.
        f  = E - e * np.sin(E) - M
        # זה הנגזרת של הפונקציה: cos(E) * e - 1 = f'(E)
        fp = 1 - e * np.cos(E)
        # חישוב של שינוי עבור E לפי שיטת ניוטון: [f(E) / f'(E)] - E = E(new)
        dE = -f / fp
        # מעדכנים את ערך E לפי השינוי.
        E += dE
        # אם כל השינויים ב־E מספיק קטנים (כלומר ההתכנסות קרובה מאוד), שוברים את הלולאה מוקדם.
        if np.max(np.abs(dE)) < tol:
            break
    # בסיום, מחזירים את מערך ה־E שהתקבל — שהוא האנומליה האקצנטרית עבור כל אחת מה־M.
    return E

# היא מתרגמת את האנומליה האקצנטרית 𝐸 ל־אנומליה אמיתית 𝜈
# שהיא הזווית שבין הלוויין לבין נקודת הפריאפסיס במישור המסלול.
# 𝐸 אנומליה אקצנטרית (eccentric anomaly), מערך numpy.
# 𝑒: אקסצנטריות של המסלול (scalar).
def true_anomaly_from_E(E: np.ndarray, e: float) -> np.ndarray:
    # מחשב את קוסינוס של כל ערך ב־E (הכנה לנוסחאות הבאות שמסתמכות על זה)
    cosE = np.cos(E)
    # מחשב גם את הסינוס של כל ערך ב־E.
    sinE = np.sin(E)
    # מחשב את השורש הריבועי של הביטוי (e^2 - 1) - זה מופיע במעבר של הפיתוחים מ-E ל - v. (לא תלוי בזמן אלא קבוע עבור אותו מסלול.)
    sqrt1me2 = math.sqrt(1 - e * e)
    # מחשב את סינוס של האנומליה האמיתית לפי הנוסחה:
    sin_nu = sqrt1me2 * sinE / (1 - e * cosE)
    #  מחשב את קוסינוס של 𝜈 לפי הנוסחה:
    cos_nu = (cosE - e) / (1 - e * cosE)
    #  מחשב את הערך של 𝜈 (האנומליה האמיתית) לפי:
    nu = np.arctan2(sin_nu, cos_nu)
    # מחזיר את המערך nu, שהוא האנומליה האמיתית (ביחידות רדיאנים)
    # — כלומר המיקום האמיתי של הלוויין במסלול האליפטי.
    return nu

# הפונקציה ממירה את המיקום של הלוויין ממערכת פריפוקלית (Perifocal, נקראת גם PQW) למערכת ייחוס אינרציאלית 
# (ECI = Earth-Centered Inertial).
# r_pf: מערך בגודל 𝑁 × 3 של מיקומי הלוויין במערכת פריפוקלית.
# i: נטיית המסלול (inclination) ברדיאנים.
# raan: זווית קו העלייה (Right Ascension of Ascending Node) ברדיאנים.
# argp: זווית הפריאפסיס (Argument of Perigee) ברדיאנים.
def perifocal_to_eci(r_pf: np.ndarray, i: float, raan: float, argp: float) -> np.ndarray:
    # מחשב את קוסינוס וסינוס של זווית הנטייה (i).
    ci, si = np.cos(i), np.sin(i)
    # מחשב את קוסינוס וסינוס של RAAN.
    cO, sO = np.cos(raan), np.sin(raan)
    # מחשב את קוסינוס וסינוס של זווית הפריאפסיס.
    cw, sw = np.cos(argp), np.sin(argp)
    # מטריצת סיבוב סביב ציר Z ב־RAAN (נקראת R₃(Ω)). זוהי רוטציה במישור X-Y שמסובבת את המסלול לפי מיקום קו העלייה.
    R3_O = np.array([[ cO, sO, 0],
                     [-sO, cO, 0],
                     [  0,  0, 1]])
    # מטריצת סיבוב סביב ציר X ב־נטייה i (נקראת R₁(i)).
    # היא "מטה" את המסלול מהמשווה לזווית הנטייה שלו.
    R1_i = np.array([[1, 0, 0],
                     [0, ci, si],
                     [0,-si, ci]])
    # מטריצת סיבוב סביב ציר Z לפי זווית הפריאפסיס (נקראת R₃(ω)).
    # ממקמת את נקודת ההתחלה של הלוויין במסלול האליפטי.
    R3_w = np.array([[ cw, sw, 0],
                     [-sw, cw, 0],
                     [  0,  0, 1]])
    # כפל מטריצות — יוצרים את מטריצת הסיבוב הכוללת ממערכת PQW ל־ECI. לפי הסדר הבא (כפי שמקובל באסטרודינמיקה):
    Q = R3_O @ R1_i @ R3_w 
    # מבצעים את הסיבוב בפועל על כל וקטורי המיקום r_pf.
    # r_pf.T: מעבירים ל־ 3 × 𝑁 כדי להכפיל את מטריצת הסיבוב.
    # Q @ r_pf.T: התוצאה היא 3×N
    # .T נוסף כדי להחזיר חזרה ל־ 𝑁 × 3, כלומר כל שורה היא וקטור מיקום חדש ב־ECI.
    return (Q @ r_pf.T).T  

# מקבלת פרמטרים של מסלול (אורך מסלול, אקסצנטריות, נטייה וכו')
# ומחזירה את מיקומי הלוויין לאורך זמן, במערכת ECI (חלל קבוע).
# a: חצי הציר הגדול של המסלול (ק"מ)
# e: אקסצנטריות (0 = מעגל, >0 = אליפסה)
# i: נטיית המסלול (radians) — כמה הוא נוטה מהמשווה
# raan: הזווית בין נקודת האביב לקו העלייה (Right Ascension of Ascending Node)
# argp: זווית פריאפסיס (היכן שיא המרחק במסלול)
# nu0: זווית אנומליה אמיתית התחלתית (True anomaly) — איפה הלוויין מתחיל
# t: מערך של נקודות זמן (שניות מאז התחלה)
# mu: פרמטר כבידה של כדור הארץ (ברירת מחדל MU_EARTH)
# r_eci: מיקום הלוויין לאורך זמן במערכת ECI
def propagate_kepler(a: float, e: float, i: float, raan: float, argp: float,
                     nu0: float, t: np.ndarray, mu: float = MU_EARTH) -> np.ndarray:
    a_m = a * 1000.0 # ממירים את a מקילומטרים למטרים (כי mu הוא ביחידות של מטר).
    # מחשבים את התנועה הממוצעת במסלול (mean motion), שזה כמה רדיאנים לשנייה הלוויין מתקדם במסלולו.
    n = mean_motion(mu, a_m)
    # ממירים את האנומליה האמיתית (nu0) לאנומליה אקצנטרית (E0), שהיא זווית חשובה במסלול האליפטי.
    E0 = 2 * math.atan(math.tan(nu0 / 2) * math.sqrt((1 - e) / (1 + e)))
    if E0 < 0:
        E0 += 2 * np.pi # אם יצאנו עם זווית שלילית (בגלל טנגנס), הופכים אותה לחיובית ע"י הוספת 360° (2π).
    M0 = E0 - e * math.sin(E0) # מחשבים את האנומליה הממוצעת בזמן התחלה — זהו פרמטר מרכזי בחישוב מיקום במסלול.

    # מחשבים את האנומליה הממוצעת בכל אחד מהזמנים — כלומר באיזה מיקום הלוויין אמור להיות על פי הזמן.
    M = M0 + n * t 
    # פתרון של משוואת קפלר: מוצאים את האנומליה האקצנטרית E מתוך M.
    # הפונקציה kepler_E_from_M עושה פתרון נומרי (ניוטון-רפסון) של משוואת קפלר.
    E = kepler_E_from_M(M, e)
    # ממירים חזרה מ־E ל־אנומליה אמיתית nu (המיקום האמיתי של הלוויין במסלול).
    nu = true_anomaly_from_E(E, e)

    # עכשיו מחשבים את המיקום: זהו המרחק של הלוויין מהמרכז, לפי הנוסחה האליפטית
    r = a_m * (1 - e * np.cos(E))
    # יוצרים את וקטור המיקום במערכת פריפוקלית (Perifocal Frame) — מערכת צירים שממוקדת במסלול עצמו.
    r_pf = np.zeros((len(t), 3))
    r_pf[:, 0] = r * np.cos(nu)
    r_pf[:, 1] = r * np.sin(nu)
    r_pf[:, 2] = 0.0 # הציר Z הוא תמיד אפס כי הלוויין במסלול מישורי סביב מרכז כדור הארץ.

    # ואז מחשבים את המהירות במסלול:

    r_eci = perifocal_to_eci(r_pf, i, raan, argp)
    return r_eci

# ============================================================
# Frames, Geodesy
# ============================================================

# לתרגם את מיקומי הלוויין ממערכת ECI (Earth-Centered Inertial) למערכת ECEF (Earth-Centered, Earth-Fixed), 
# עבור כל נקודת זמן בסימולציה.
# ECI = מערכת גלובלית קבועה ביחס לכוכבים.
# ECEF = מערכת שמסתובבת יחד עם כדור הארץ.
# r_eci: מערך בגודל 𝑁×3, המייצג את מיקום הלוויין במערכת ECI בכל נקודת זמן.
# t_seconds: מערך של אורך 𝑁, המציין את הזמן שעבר מאז ההתחלה (בשניות) — עבור כל אינדקס.
# theta0: הזווית הראשונית של סיבוב כדור הארץ (ברירת מחדל: 0 רדיאנים).
def eci_to_ecef(r_eci: np.ndarray, t_seconds: np.ndarray, theta0: float = 0.0) -> np.ndarray:
    #  שומר את מספר נקודות הזמן בסימולציה.
    N = len(t_seconds)
    # יוצר מערך חדש בגודל זהה ל־r_eci, מלא באפסים — אליו נכניס את תוצאות ההמרה ל־ECEF.
    r_ecef = np.zeros_like(r_eci)
    # לולאה על כל נקודת זמן (למשל כל שנייה או כל 10 שניות), לפי מה שהוגדר בסימולציה.
    for k in range(N):
        # מחשב את זווית הסיבוב של כדור הארץ ברגע k:
        # OMEGA_EARTH הוא הקצב הזוויתי של סיבוב כדור הארץ (בערך 7.292×10⁻⁵ רדיאנים לשנייה).
        # כופלים בזמן שעבר כדי לדעת כמה כדור הארץ הסתובב מאז תחילת הסימולציה.
        # theta0 מאפשר לקבוע את המיקום ההתחלתי של גריניץ' (למשל אם מתחילים מ־GMT ≠ 0).
        theta = theta0 + OMEGA_EARTH * t_seconds[k]
        # מחשבים את קוסינוס וסינוס של הזווית — הכנה למטריצת סיבוב.
        c, s = np.cos(theta), np.sin(theta)
        # בונה את מטריצת הסיבוב סביב ציר Z:
        # סיבוב של מערכת ECI סביב ציר Z (כיוון מעלה) כדי לקבל את מערכת ECEF.
        R = np.array([[ c,  s, 0],
                      [-s,  c, 0],
                      [ 0,  0, 1]])
        # מבצע את הסיבוב בפועל עבור מיקום k:
        # כפל מטריצות בין R לבין הווקטור r_eci[k].
        # התוצאה: מיקום הלוויין במערכת ECEF.
        r_ecef[k] = R @ r_eci[k]
    # מחזיר את מערך המיקומים במערכת ECEF.
    return r_ecef

# לקבל וקטורי מיקום במערכת ECEF בגודל 𝑁×3 ולהחזיר את קו הרוחב (lat) ו־קו האורך (lon) של כל נקודה.
# r_ecef: מערך בגודל 𝑁×3 — מיקומי לוויין במערכת ECEF (x, y, z).
def ecef_to_geodetic_spherical(r_ecef: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    # מפרק את הקואורדינטות הקרטזיות (X, Y, Z) מתוך מערך הקלט.
    # כל אחד מהווקטורים x, y, z הוא באורך 𝑁 — כלומר ערכים עבור כל נקודת זמן.
    x, y, z = r_ecef[:, 0], r_ecef[:, 1], r_ecef[:, 2]
    # מחשב את קו האורך לפי: longitude=arctan2(y,x)
    # קו אורך נמדד לפי הזווית בין וקטור המיקום לבין ציר ה־X במישור XY (קווי האורך סובבים סביב ציר Z).
    # arctan2 מחזיר את הזווית הנכונה בין −𝜋 ל־+𝜋 לפי הרבע.
    lon = np.arctan2(y, x)
    # מחשב את המרחק של הנקודה ממישור XY (רדיוס במישור XY).
    # כלומר, המרחק בין הנקודה לציר Z (לצורך חישוב קו רוחב).
    rxy = np.sqrt(x * x + y * y)
    # מחשב את קו הרוחב לפי הנוסחה. זה בעצם הזווית בין הנקודה לבין מישור קו המשווה (XY).
    # קו רוחב 0: בקו המשווה, קו רוחב 𝜋/2: בקוטב הצפוני.
    lat = np.arctan2(z, rxy)
    # מחזיר את קווי הרוחב והאורך.
    return lat, lon

# לקבל: lat: קו רוחב (ברדיאנים), lon: קו אורך (ברדיאנים), h: גובה מעל פני השטח של כדור הארץ (במטרים)
# הפונקציה מקבלת את קו הרוחב, קו האורך, והגובה, ומחזירה וקטור [x, y, z] במערכת ECEF.
def geodetic_to_ecef(lat: float, lon: float, h: float) -> np.ndarray:
    # מחשב את קוסינוס וסינוס של קו הרוחב.
    clat, slat = np.cos(lat), np.sin(lat)
    # מחשב את קוסינוס וסינוס של קו האורך.
    clon, slon = np.cos(lon), np.sin(lon)
    # מחשב את הרדיוס הכולל של הנקודה:
    # R_EARTH הוא רדיוס כדור הארץ (בערך 6371 ק"מ).
    # מוסיפים לו את הגובה h של הנקודה מעל פני הקרקע.
    r = R_EARTH + h
    # מחשב את הקואורדינטה x לפי הנוסחה: x=r⋅cos(lat)⋅cos(lon)
    x = r * clat * clon
    # מחשב את הקואורדינטה y: y=r⋅cos(lat)⋅sin(lon)
    y = r * clat * slon
    # מחשב את הקואורדינטה z: z=r⋅sin(lat)
    z = r * slat
    # מחזיר את הווקטור [x, y, z] כמערך numpy — מיקום במערכת ECEF.
    return np.array([x, y, z])

# לחשב את מטריצת המעבר ממערכת קואורדינטות ECEF למערכת ENU (East-North-Up), עבור
# מיקום מסוים המוגדר לפי קו רוחב (lat) וקו אורך (lon). זאת אומרת: אם אתה נמצא בתחנת קרקע מסוימת ורוצה לדעת מאיזה כיוון הלוויין יופיע,
# אתה מתרגם את הכיוון שלו אליך דרך מטריצת ENU.
# ENU היא מערכת קואורדינטות מקומית: E(East), N(North), U(Up)
# היא מסודרת ביחס למיקום מסוים על פני כדור הארץ.
# הפונקציה מקבלת קו רוחב (lat) וקו אורך (lon) ברדיאנים, ומחזירה מטריצת numpy בגודל 3×3 שתתרגם וקטורים מ־ECEF ל־ENU.
def ecef_to_enu_matrix(lat: float, lon: float) -> np.ndarray:
    # מחשב את סינוס וקוסינוס של קו הרוחב.
    sL, cL = np.sin(lat), np.cos(lat)
    # מחשב את סינוס וקוסינוס של קו האורך.
    sO, cO = np.sin(lon), np.cos(lon)

    # Rows are [e; n; u] in ECEF coords
    return np.array([
        # שורה ראשונה — כיוון מזרח (East): זהו הווקטור שמצביע למזרח במערכת ECEF
        # .נובע מתנועה בכיוון חיובי של קו האורך.
        [-sO,          cO,         0],
        # שורה שנייה — כיוון צפון (North): זהו הווקטור
        #  שמצביע צפונה (לכיוון קו רוחב גבוה יותר). מורכב מהשפעה גם בציר X, גם בציר Y וגם ב-Z — לפי קו הרוחב.
        [-sL * cO, -sL * sO,  cL],
        # שורה שלישית — כיוון למעלה (Up):
        # כיוון מעלה (zenith) – כלומר וקטור שפונה ישירות מהמרכז של כדור הארץ החוצה.
        [ cL * cO,  cL * sO,  sL]
        # מטריצה בגודל 3×3, שכשכופלים אותה בוקטור ECEF מקבלים את הוקטור המתאים במערכת ENU:
        # enu=Q⋅(target−station). (ההפרש בין מיקום הלוויין למיקום התחנה מתורגם ל־ENU דרך מטריצה זו.)
    ])

# לקבל: מיקום תחנת הקרקע (קו רוחב, קו אורך, גובה), מיקום הלוויין במערכת ECEF (וקטור 3D)
# gs_lat, gs_lon, gs_h: קו רוחב, קו אורך, גובה התחנה (ברדיאנים ומטרים).
# r_sat_ecef: מיקום הלוויין במערכת ECEF (וקטור באורך 3).
def elevation_angle(gs_lat: float, gs_lon: float, gs_h: float,
                    r_sat_ecef: np.ndarray) -> float:
    # מחשב את מיקום התחנה במערכת ECEF.
    r_gs = geodetic_to_ecef(gs_lat, gs_lon, gs_h)
    # מחשב את וקטור הכיוון מהתחנה אל הלוויין (שנקרא "וקטור טופוצנטרי").
    # זהו פשוט ההפרש בין מיקום הלוויין למיקום התחנה, במערכת ECEF.
    rho = r_sat_ecef - r_gs
    # מחשב את מטריצת המעבר מ־ECEF ל־ENU עבור התחנה הזו.
    E = ecef_to_enu_matrix(gs_lat, gs_lon)
    # מתרגם את וקטור הכיוון מהתחנה אל הלוויין ממערכת ECEF למערכת ENU (East, North, Up).
    # עכשיו אנחנו יודעים מה הכיוון של הלוויין יחסית ל"צפון", "מזרח" ו"למעלה" של התחנה.
    enu = E @ rho
    # מחשב את זווית הגובה (elevation angle) לפי הנוסחה: 
    # enu[2] זה הרכיב של Up (ציר Z במערכת המקומית).
    # np.linalg.norm(enu) זה האורך של וקטור הכיוון הכולל.
    el = np.arcsin(enu[2] / np.linalg.norm(enu))
    # מחזיר את זווית הגובה (ברדיאנים).
    return el

# ============================================================
# Access computations
# ============================================================

# לחשב את חלונות הגישה (access windows) של כל לוויין מול כל תחנת קרקע לאורך זמן. כלומר: לכמה זמן,
# ובאיזה טווחים, כל תחנת קרקע רואה לוויין מסוים מעל קו האופק.
# times: רשימת הזמנים של הסימולציה (ב־datetime).
# r_ecef_all: מיקומי כל הלוויינים לאורך זמן (לפי שמות) במערכת ECEF.
# ground_stations: רשימת תחנות קרקע.
# min_el_rad: זווית הגובה המינימלית (ברדיאנים) שממנה נחשב שהלוויין "נראה".
def compute_access_windows(times: List[dt.datetime],
                           r_ecef_all: Dict[str, np.ndarray],
                           ground_stations: List[GroundStation],
                           min_el_rad: float) -> Dict[str, Dict[str, List[Tuple[dt.datetime, dt.datetime]]]]:
    # אתחול של מילון התוצאה: נשמור עבור כל תחנה → כל לוויין → רשימת חלונות גישה.
    access: Dict[str, Dict[str, List[Tuple[dt.datetime, dt.datetime]]]] = {}
    # נתחיל לעבור תחנה-תחנה.
    for gs in ground_stations:
        # ממירים את קו הרוחב והאורך של התחנה לרדיאנים.
        # יוצרים לה רשומת מילון ריקה ב־access.
        gs_lat = gs.lat_deg * DEG2RAD
        gs_lon = gs.lon_deg * DEG2RAD
        access[gs.name] = {}
        # עבור כל לוויין ומיקום הזמן שלו לאורך הסימולציה (ב־ECEF), נחשב זוויות גובה.
        for sat_name, r_ecef in r_ecef_all.items():
            # מחשב את זווית הגובה בין התחנה ללוויין בכל נקודת זמן. שומר את זה במערך el.
            el = np.zeros(len(times))
            for k in range(len(times)):
                el[k] = elevation_angle(gs_lat, gs_lon, gs.height_m, r_ecef[k])
            # יוצרים מערך בוליאני: True אם הלוויין מעל קו האופק (כלומר זווית > מינימום), False אחרת.
            above = el > min_el_rad
            # נתחיל תהליך של איתור "חלונות" רציפים של זמן שבהם הלוויין היה מעל האופק.
            windows = []
            # in_pass: האם אנחנו כרגע בתוך חלון גישה?
            in_pass = False
            # start_idx: מתי התחיל חלון הגישה הנוכחי?
            start_idx = None
            # ברגע שהלוויין עולה מעל האופק → מתחילים חלון חדש (in_pass = True).
            for k in range(len(times)):
                if above[k] and not in_pass:
                    in_pass = True
                    start_idx = k
                # אם הלוויין ירד מתחת לאופק (או שהגענו לסוף), סוגרים את חלון הגישה:
                # מוסיפים את הזוג (start_time, end_time) לרשימת ה־windows.
                # מאפסים את in_pass.
                if in_pass and (not above[k] or k == len(times) - 1):
                    end_idx = k if not above[k] else k
                    windows.append((times[start_idx], times[end_idx]))
                    in_pass = False
            # שומרים את כל חלונות הגישה בין הלוויין לתחנה למילון הראשי.
            access[gs.name][sat_name] = windows
    # מחזירים את כל טבלאות החלונות: לכל תחנה ולכל לוויין.
    return access

# ============================================================
# Plotting
# ============================================================

def plot_ground_tracks(latlon_by_sat: Dict[str, Tuple[np.ndarray, np.ndarray]], save_dir: str = None):
    plt.figure(figsize=(10, 5))
    # simple rectangular projection
    for sat, (lat, lon) in latlon_by_sat.items():
        # fix longitudes to [-180, 180]
        lon_deg = ((lon * RAD2DEG + 180) % 360) - 180
        lat_deg = lat * RAD2DEG
        plt.plot(lon_deg, lat_deg, '.', markersize=1, label=sat)
    plt.xlim([-180, 180])
    plt.ylim([-90, 90])
    plt.xlabel('Longitude [deg]')
    plt.ylabel('Latitude [deg]')
    plt.title('Ground Tracks')
    plt.grid(True, alpha=0.3)
    plt.legend(loc='upper right', fontsize=8)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, 'ground_tracks.png'), dpi=200, bbox_inches='tight')

def plot_access_timeline(access: Dict[str, Dict[str, List[Tuple[dt.datetime, dt.datetime]]]],
                         save_dir: str = None):
    """
    Simple Gantt-like timeline: each station gets its own subplot; each satellite's windows are horizontal bars
    """
    stations = list(access.keys())
    ns = len(stations)
    fig, axes = plt.subplots(ns, 1, figsize=(12, 2.5 * ns), sharex=True)
    if ns == 1:
        axes = [axes]

    for ax, station in zip(axes, stations):
        sat_to_windows = access[station]
        y_ticks = []
        y_labels = []
        for idx, (sat, wins) in enumerate(sat_to_windows.items()):
            for (t0, t1) in wins:
                ax.plot([t0, t1], [idx, idx], lw=6)
            y_ticks.append(idx)
            y_labels.append(sat)
        ax.set_yticks(y_ticks)
        ax.set_yticklabels(y_labels)
        ax.set_title(f"Access windows for {station}")
        ax.grid(True, axis='x', alpha=0.3)
    fig.autofmt_xdate()
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        fig.savefig(os.path.join(save_dir, 'access_timeline.png'), dpi=200, bbox_inches='tight')

# ============================================================
# Main simulation
# ============================================================

# scn: אובייקט מסוג Scenario — התרחיש הכולל
# make_plots: האם להציג את הגרפים על המסך?
# save_plots: מחרוזת עם שם תיקייה לשמירת הגרפים (None אם לא רוצים לשמור)
def run_sim(scn: Scenario, make_plots: bool, save_plots: str = None):
    # יוצרים רשימת נקודות זמן (timestamp) לאורך כל הסימולציה — מתחילים מהשעה הנתונה, בקפיצות קבועות (למשל כל 30 שניות).
    times = linspace_times(scn.t_start_utc, scn.duration_seconds, scn.time_step_seconds)
    # ממירים את הזמנים לרשימת מספרים (בשניות) — כלומר: כמה שניות עברו מהזמן הראשון.
    # לדוגמה: [0, 30, 60, 90, ...]
    t_seconds = np.array([(t - times[0]).total_seconds() for t in times])

    #  התחלה של לולאת לוויינים: מגדירים שלושה מילונים ריקים:
    r_eci_all: Dict[str, np.ndarray] = {} # מיקום הלוויין במערכת ECI (חלל)
    r_ecef_all: Dict[str, np.ndarray] = {} # מיקום הלוויין במערכת ECEF (יחסית לכדור הארץ)
    latlon_by_sat: Dict[str, Tuple[np.ndarray, np.ndarray]] = {} # זוגות של (קו רוחב, קו אורך) של הלוויין בכל רגע

    for sat in scn.satellites: # לולאה על כל לוויין בתרחיש. קלטים למסלול של הלוויין:
        a = sat.a_km # רדיוס מסלול ממוצע (ק"מ)
        e = sat.e # אקסצנטריות (כמה המסלול אליפטי)
        i = sat.i_deg * DEG2RAD # נטיית המסלול (כמה הוא נוטה מהמשווה). 
        raan = sat.raan_deg * DEG2RAD # פרמטר זוויתי למסלול
        argp = sat.argp_deg * DEG2RAD # פרמטר זוויתי למסלול
        nu0 = sat.nu0_deg * DEG2RAD # פרמטר זוויתי למסלול
        # ממירים את כל הזוויות מרדיאנים לדגריז (כדי לעבוד נכון מתמטית).

        # מחשבת את מיקום (r) של הלוויין בכל רגע בזמן. לפי פיזיקה של מסלול קפלר (2-body orbit)
        # התוצאה היא r_eci: מערך Nx3 של מיקומים בזמן (במערכת ECI — חלל קבוע)
        r_eci = propagate_kepler(a, e, i, raan, argp, nu0, t_seconds, MU_EARTH)
        # שומרים את מיקומי הלוויין הזה בתוך המילון r_eci_all, לפי שמו.
        r_eci_all[sat.name] = r_eci

        # ממירים את המיקומים ממערכת חללית (ECI) ל־ECEF — כלומר: איפה הלוויין מעל כדור הארץ המסתובב
        r_ecef = eci_to_ecef(r_eci, t_seconds, theta0=0.0)
        # שומרים את המיקומים החדשים לפי שם הלוויין.
        r_ecef_all[sat.name] = r_ecef
        # ממירים את כל המיקומים (XYZ) ל־קו רוחב / קו אורך / גובה.
        lat, lon = ecef_to_geodetic_spherical(r_ecef)
        # שומרים את הנתונים כדי לצייר אחר כך את מסלול הקרקע (Ground Track) של הלוויין.
        latlon_by_sat[sat.name] = (lat, lon)

    # ממירים את זווית הגובה המינימלית (למשל 10°) לרדיאנים.
    min_el_rad = scn.min_elevation_deg * DEG2RAD
    # מחשבים את כל חלונות הגישה בין לוויינים לתחנות קרקע.
    # הפונקציה בודקת מתי הלוויין נמצא מעל התחנה בזווית גובה > min_el.
    access = compute_access_windows(times, r_ecef_all, scn.ground_stations, min_el_rad)

    # עבור כל לוויין, מדפיסים את כל חלונות הגישה
    print("\n=== ACCESS WINDOWS ===")
    for gs in scn.ground_stations: # לולאה על כל תחנה קרקעית
        print(f"\nGround station: {gs.name}")
        for sat in scn.satellites:
            wins = access[gs.name][sat.name]
            if not wins:
                print(f"  {sat.name}: no access") # אם אין קשר בכלל → מדפיסים no access
            else:
                for (t0, t1) in wins:
                    # אם יש → מדפיסים טווחי זמנים בפורמט ISO (כולל משך)
                    print(f"  {sat.name}: {t0.isoformat()}  -->  {t1.isoformat()}  (duration {(t1-t0).total_seconds():.0f} s)")

    # Plot
    if make_plots or save_plots: # אם ביקשת להציג גרפים או לשמור קבצים:
        plot_ground_tracks(latlon_by_sat, save_dir=save_plots) # מציירים את מסלולי הקרקע של כל הלוויינים
        plot_access_timeline(access, save_dir=save_plots) # מציירים את תרשים חלונות הגישה (Gantt style)
        if make_plots:
            plt.show() # אם make_plots=True → מציגים את הגרפים על המסך

def main():
    # כאן אנחנו יוצרים אובייקט שמטפל בפרמטרים של שורת הפקודה
    # זה יאפשר להריץ את הסקריפט עם דגלים כמו --scenario או --plot
    # התיאור (description) הוא הסבר שיוצג אם נעשה python main.py --help
    parser = argparse.ArgumentParser(description="OrbitalSim: Constellation Ground-Track & Access Planner")
    # כאן מוסיפים פרמטר חובה בשם --scenario
    # כלומר, המשתמש חייב לציין את הנתיב לקובץ JSON שמכיל את פרטי התרחיש (לוויינים, תחנות וכו').
    parser.add_argument("--scenario", required=True, help="Path to scenario JSON")
    # כאן מוסיפים דגל (flag) אופציונלי בשם --plot.
    # אם הוא מופיע → args.plot == True, אם הוא לא מופיע → args.plot == False
    # הוא פשוט אומר: "האם להציג את הגרפים על המסך?"
    parser.add_argument("--plot", action="store_true", help="Show interactive plots")
    # עוד פרמטר אופציונלי, שמאפשר לציין תקיית יעד לשמירת הגרפים כקבצי PNG.
    # אם לא מציינים אותו → הוא יהיה None (ולא יישמרו קבצים), 
    # אם כן מציינים: --save-plots outputs → הגרפים יישמרו ב־outputs/
    parser.add_argument("--save-plots", default=None, help="Directory to save plots (PNG)")
    # כאן מבצעים את הפירוק של שורת הפקודה עצמה — ומקבלים אובייקט args, שבו מאוחסנים כל הערכים מהשורה.
    # דוגמה למה יש ב־args אם הרצת את זה כך: python main.py --scenario scenario_demo.json --plot
    # args.scenario = "scenario_demo.json", args.plot = True, args.save_plots = None
    args = parser.parse_args()
    # כאן אנחנו טוענים את התרחיש מקובץ ה־JSON שצויין.
    scn = load_scenario(args.scenario)
    # השורה הזו מריצה את כל הסימולציה בפועל דרך הפונקציה run_sim(...).
    run_sim(scn, make_plots=args.plot, save_plots=args.save_plots)

if __name__ == "__main__":
    main()
