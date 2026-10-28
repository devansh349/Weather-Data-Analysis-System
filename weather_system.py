import math
import random
import json
import csv
import datetime
from typing import List, Dict, Any, Tuple

random.seed(66)  # reproducible demo run


class Reading:
    """Stores one weather observation and its validation status."""
    def __init__(self, city: str, timestamp: datetime.datetime, temp_c: float,
                 humidity_pct: float, pressure_hpa: float,
                 wind_kmh: float, rain_mm: float):
        self.city = city
        self.timestamp = timestamp
        self.temp_c = temp_c
        self.humidity_pct = humidity_pct
        self.pressure_hpa = pressure_hpa
        self.wind_kmh = wind_kmh
        self.rain_mm = rain_mm
        self.valid = True
        self.flags: List[str] = []

    def to_dict(self) -> Dict[str, Any]:
        return {
            "city": self.city,
            "timestamp": self.timestamp.strftime("%Y-%m-%d %H:%M"),
            "temp_c": round(self.temp_c, 2),
            "humidity_pct": round(self.humidity_pct, 1),
            "pressure_hpa": round(self.pressure_hpa, 1),
            "wind_kmh": round(self.wind_kmh, 1),
            "rain_mm": round(self.rain_mm, 1),
            "valid": self.valid,
            "flags": ";".join(self.flags),
        }


class DataCollector:
    """Generates simulated sensor readings for five cities."""
    CITIES = [
        {"name": "Ludhiana", "base": 27.0, "hum": 55.0},
        {"name": "Chandigarh", "base": 26.0, "hum": 58.0},
        {"name": "Amritsar", "base": 27.5, "hum": 52.0},
        {"name": "Shimla", "base": 17.0, "hum": 70.0},
        {"name": "Jaipur", "base": 30.0, "hum": 45.0},
    ]

    @classmethod
    def generate(cls, days: int = 7) -> List[Reading]:
        readings = []
        start = datetime.datetime(2026, 10, 21, 0, 0)
        for d in range(days):
            for hour in range(0, 24, 3):  # 8 readings/day
                ts = start + datetime.timedelta(days=d, hours=hour)
                cycle = math.sin((hour - 9) * math.pi / 12)
                for c in cls.CITIES:
                    temp = c["base"] + 6 * cycle + random.uniform(-1.2, 1.2)
                    hum = c["hum"] - 12 * cycle + random.uniform(-4, 4)
                    hum = max(10.0, min(98.0, hum))
                    pres = 1010 + random.uniform(-4, 4)
                    wind = max(1.0, 10 + 5 * cycle + random.uniform(-4, 4))
                    rain = random.choice([0, 0, 0, 0, 0, 0, 1.5, 4.0, 9.0])
                    # Inject faulty sensor values for validation testing.
                    if d == 2 and hour == 12 and c["name"] == "Jaipur":
                        temp = 95.0
                    if d == 4 and hour == 6 and c["name"] == "Shimla":
                        hum = 140.0
                    if d == 5 and hour == 18 and c["name"] == "Amritsar":
                        pres = 700.0
                    readings.append(Reading(c["name"], ts, temp, hum, pres, wind, rain))
        return readings


class DataValidator:
    """Rejects observations with values outside configured physical limits."""
    LIMITS = {
        "temp": (-50.0, 55.0),
        "hum": (0.0, 100.0),
        "pres": (850.0, 1090.0),
        "wind": (0.0, 250.0),
        "rain": (0.0, 300.0),
    }

    @classmethod
    def validate(cls, data: List[Reading]) -> Tuple[List[Reading], Dict[str, int]]:
        clean = []
        stats = {"total": len(data), "valid": 0, "rejected": 0}
        limits = cls.LIMITS
        for r in data:
            if not limits["temp"][0] <= r.temp_c <= limits["temp"][1]:
                r.flags.append("TEMP_OUT_OF_RANGE")
            if not limits["hum"][0] <= r.humidity_pct <= limits["hum"][1]:
                r.flags.append("HUMIDITY_OUT_OF_RANGE")
            if not limits["pres"][0] <= r.pressure_hpa <= limits["pres"][1]:
                r.flags.append("PRESSURE_OUT_OF_RANGE")
            if not limits["wind"][0] <= r.wind_kmh <= limits["wind"][1]:
                r.flags.append("WIND_OUT_OF_RANGE")
            if not limits["rain"][0] <= r.rain_mm <= limits["rain"][1]:
                r.flags.append("RAIN_OUT_OF_RANGE")
            r.valid = not r.flags
            if r.valid:
                clean.append(r)
                stats["valid"] += 1
            else:
                stats["rejected"] += 1
        return clean, stats


class WeatherAnalyzer:
    @staticmethod
    def heat_index(temp_c: float, rh: float) -> float:
        if temp_c < 27.0 or rh < 40.0:
            return temp_c
        temp_f = temp_c * 9 / 5 + 32
        hi = (
            -42.379 + 2.04901523 * temp_f + 10.14333127 * rh
            - 0.22475541 * temp_f * rh - 0.00683783 * temp_f * temp_f
            - 0.05481717 * rh * rh + 0.00122874 * temp_f * temp_f * rh
            + 0.00085282 * temp_f * rh * rh
            - 0.00000199 * temp_f * temp_f * rh * rh
        )
        return (hi - 32) * 5 / 9

    @staticmethod
    def dew_point(temp_c: float, rh: float) -> float:
        a, b = 17.27, 237.7
        gamma = (a * temp_c) / (b + temp_c) + math.log(max(rh, 1.0) / 100.0)
        return (b * gamma) / (a - gamma)

    @staticmethod
    def std_dev(values: List[float]) -> float:
        mean = sum(values) / len(values)
        return math.sqrt(sum((value - mean) ** 2 for value in values) / len(values))

    @classmethod
    def city_summary(cls, data: List[Reading]) -> Dict[str, Dict[str, Any]]:
        groups: Dict[str, List[Reading]] = {}
        for reading in data:
            groups.setdefault(reading.city, []).append(reading)
        output = {}
        for city, readings in groups.items():
            temps = [r.temp_c for r in readings]
            output[city] = {
                "readings": len(readings),
                "avg_temp_c": round(sum(temps) / len(temps), 2),
                "min_temp_c": round(min(temps), 2),
                "max_temp_c": round(max(temps), 2),
                "temp_std_dev": round(cls.std_dev(temps), 2),
                "avg_humidity_pct": round(sum(r.humidity_pct for r in readings) / len(readings), 1),
                "avg_dew_point_c": round(sum(cls.dew_point(r.temp_c, r.humidity_pct) for r in readings) / len(readings), 2),
                "max_heat_index_c": round(max(cls.heat_index(r.temp_c, r.humidity_pct) for r in readings), 2),
                "total_rain_mm": round(sum(r.rain_mm for r in readings), 1),
                "max_wind_kmh": round(max(r.wind_kmh for r in readings), 1),
            }
        return output

    @staticmethod
    def forecast(series: List[float], horizon: int = 3, window: int = 4) -> List[float]:
        """Moving average plus average recent trend."""
        if len(series) < window + 1:
            return [series[-1]] * horizon
        recent = series[-window:]
        average = sum(recent) / window
        trend = (recent[-1] - recent[0]) / (window - 1)
        return [round(average + trend * (i + 1), 2) for i in range(horizon)]


class AlertEngine:
    @staticmethod
    def evaluate(data: List[Reading]) -> List[str]:
        alerts = []
        latest_ts = max(r.timestamp for r in data)
        for r in (item for item in data if item.timestamp == latest_ts):
            heat_index = WeatherAnalyzer.heat_index(r.temp_c, r.humidity_pct)
            if heat_index >= 38:
                alerts.append(f"HEAT ADVISORY - {r.city}: heat index {heat_index:.1f} C")
            if r.rain_mm >= 8:
                alerts.append(f"HEAVY RAIN     - {r.city}: {r.rain_mm:.1f} mm")
            if r.wind_kmh >= 30:
                alerts.append(f"HIGH WIND      - {r.city}: {r.wind_kmh:.1f} km/h")
            if r.temp_c <= 5:
                alerts.append(f"COLD ADVISORY - {r.city}: {r.temp_c:.1f} C")
        return alerts


class WeatherSystem:
    def run(self):
        line = "=" * 78
        print(line)
        print(">>> WEATHER DATA ANALYSIS SYSTEM <<<")
        print(line)

        print("[1/5] Collecting weather data for 5 cities (7 days)...")
        raw = DataCollector.generate(days=7)
        print(f"      [OK] Collected {len(raw)} readings")

        print("[2/5] Validating and cleaning data...")
        clean, qc = DataValidator.validate(raw)
        print(f"      [OK] Valid: {qc['valid']} | Rejected: {qc['rejected']}")
        for reading in raw:
            if not reading.valid:
                print(f"     [REJECTED] {reading.city} {reading.timestamp:%d-%b %H:%M} -> {', '.join(reading.flags)}")

        print("[3/5] Computing statistics, heat index and dew point...")
        summary = WeatherAnalyzer.city_summary(clean)

        print("[4/5] Forecasting temperature and checking alerts...")
        alerts = AlertEngine.evaluate(clean)

        print("[5/5] Generating report...\n")
        print(line)
        print(f"{'CITY':<12}| {'AVG':>6} | {'MIN':>6} | {'MAX':>6} | {'STD':>5} | {'HUM%':>5} | {'DEW':>5} | {'RAIN':>6}")
        print("-" * 78)
        for city, metrics in summary.items():
            print(f"{city:<12}| {metrics['avg_temp_c']:>6.1f} | {metrics['min_temp_c']:>6.1f} | "
                  f"{metrics['max_temp_c']:>6.1f} | {metrics['temp_std_dev']:>5.2f} | "
                  f"{metrics['avg_humidity_pct']:>5.1f} | {metrics['avg_dew_point_c']:>5.1f} | {metrics['total_rain_mm']:>5.1f}mm")
        print("-" * 78)

        series = [r.temp_c for r in clean if r.city == "Ludhiana"]
        forecast = WeatherAnalyzer.forecast(series, horizon=3)
        print("\nFORECAST (Ludhiana, next 3 readings = +3h, +6h, +9h):")
        print(f" Latest: {series[-1]:.2f} C --> {forecast[0]} C | {forecast[1]} C | {forecast[2]} C")

        print(f"\nACTIVE ALERTS ({len(alerts)}):")
        if not alerts:
            print(" [ALL CLEAR] No alerts for the latest observation.")
        for alert in alerts:
            print(" ! " + alert)
        print(line)

        self.export(clean, qc, summary, alerts)

    @staticmethod
    def export(clean, qc, summary, alerts):
        with open("weather_clean_data.csv", "w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=list(clean[0].to_dict().keys()))
            writer.writeheader()
            for reading in clean:
                writer.writerow(reading.to_dict())
        with open("weather_report.json", "w", encoding="utf-8") as file:
            json.dump({
                "generated_at": datetime.datetime.now().isoformat(),
                "validation": qc,
                "city_summary": summary,
                "alerts": alerts,
            }, file, indent=4)
        print("\n[EXPORT COMPLETE] Saved 'weather_clean_data.csv' and 'weather_report.json'.\n")


if __name__ == "__main__":
    WeatherSystem().run()
