"""Appliance classes for energy calculations."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List


@dataclass(slots=True)
class Appliance:
    """Represents a single appliance."""
    name: str
    power_w: float
    usage_minutes: float = 0.0

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("'name' cannot be empty")
        if self.power_w < 0:
            raise ValueError("'power_w' must be non-negative")
        if self.usage_minutes < 0:
            raise ValueError("'usage_minutes' must be non-negative")

    def with_minutes(self, minutes: float) -> "Appliance":
        """Return copy with updated usage minutes."""
        return Appliance(self.name, self.power_w, minutes)

    def energy_kwh_single_use(self) -> float:
        """Calculate energy for single use in kWh."""
        return self.power_w * (self.usage_minutes / 60.0) / 1000.0

    def __str__(self) -> str:
        return f"{self.name} ({self.power_w}W, {self.usage_minutes}min)"

    def to_dict(self) -> dict:
        return {"name": self.name, "power_w": self.power_w, "usage_minutes": self.usage_minutes}

    @staticmethod
    def from_dict(d: dict) -> "Appliance":
        return Appliance(d["name"], float(d["power_w"]), float(d.get("usage_minutes", 0.0)))


class ApplianceList:
    """Container for multiple appliances."""
    def __init__(self, appliances: Iterable[Appliance] | None = None) -> None:
        self.appliances: List[Appliance] = list(appliances) if appliances else []

    def add(self, app: Appliance) -> None:
        if not isinstance(app, Appliance):
            raise TypeError("Only Appliance instances can be added")
        self.appliances.append(app)

    def extend(self, items: Iterable[Appliance]) -> None:
        for it in items:
            self.add(it)

    def total_energy_kwh(self) -> float:
        """Calculate total energy for all appliances."""
        return sum(a.energy_kwh_single_use() for a in self.appliances)

    def __add__(self, other: "ApplianceList") -> "ApplianceList":
        if not isinstance(other, ApplianceList):
            return NotImplemented
        return ApplianceList([*self.appliances, *other.appliances])

    def __len__(self) -> int:
        return len(self.appliances)

    def __iter__(self):
        return iter(self.appliances)
