"""Mock calendar tool for testing. This would be replaced with a Zoom Calendar API."""
# src/tools/calendar_mock.py

import json
from datetime import datetime, timedelta
from pathlib import Path

CALENDAR_FILE = Path(__file__).parent / "calendar.json"


def _load_calendar():
    if CALENDAR_FILE.exists():
        with open(CALENDAR_FILE, "r") as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                return []
    return []


def _save_calendar(events):
    with open(CALENDAR_FILE, "w") as f:
        json.dump(events, f, indent=2)


def list_events(start_date=None, end_date=None):
    """
    List events between start_date and end_date (inclusive).
    Dates: 'YYYY-MM-DD' or 'YYYY-MM-DDTHH:MM' strings. Defaults to all.
    """
    events = _load_calendar()
    if start_date:
        start = datetime.fromisoformat(start_date)
    else:
        start = datetime.min
    if end_date:
        end = datetime.fromisoformat(end_date)
        # If end_date is date-only (no time), set to end of day
        if "T" not in end_date:
            end = end.replace(hour=23, minute=59,
                              second=59, microsecond=999999)
    else:
        end = datetime.max

    filtered = [
        e for e in events
        if start <= datetime.fromisoformat(e["start"]) <= end
    ]
    return sorted(filtered, key=lambda e: e["start"])


def list_today():
    """Convenience: list all today's events."""
    today = datetime.now().date().isoformat()
    start = datetime.fromisoformat(today)
    end = start + timedelta(days=1)
    return list_events(start.isoformat(), end.isoformat())


def create_event(title: str, start_time: str, duration_minutes: int = 60):
    """
    Create an event.
      title: str
      start_time: ISO string 'YYYY-MM-DDTHH:MM'
      duration_minutes: event length
    Returns created event dict.
    """
    try:
        start_dt = datetime.fromisoformat(start_time)
    except ValueError:
        raise ValueError(
            "start_time must be ISO format, e.g. '2025-11-05T10:00'")

    end_dt = start_dt + timedelta(minutes=duration_minutes)
    event = {
        "title": title,
        "start": start_dt.isoformat(timespec="minutes"),
        "end": end_dt.isoformat(timespec="minutes"),
        "duration": duration_minutes,
    }

    events = _load_calendar()
    events.append(event)
    _save_calendar(events)
    return event


def clear_events():
    """Delete all events (useful for tests)."""
    _save_calendar([])


if __name__ == "__main__":
    print("📅 Calendar mock demo")
    clear_events()
    create_event("Agent walkthrough with Xi", "2025-11-05T10:00", 60)
    create_event("Sync with recruiter", "2025-11-06T15:30", 30)
    print("Today's events:", list_today())
    print("All events:", list_events())
