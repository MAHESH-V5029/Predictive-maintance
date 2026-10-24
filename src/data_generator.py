"""Simulated run-to-failure sensor data for industrial machines.

Each machine starts healthy and degrades until it fails. Degradation slowly pushes the
sensors away from their healthy baseline (more heat, more vibration, lower pressure and
speed, higher torque). On top of that we add:
  * per-machine baseline offsets and sensitivities (no two machines are identical),
  * sensor noise,
  * a periodic load cycle (a nuisance signal that is unrelated to failure),
  * rare vibration spikes (outliers that can cause false alarms).
The same Machine class drives both the training data and the live dashboard demo.
"""
import numpy as np
import pandas as pd

from .config import HORIZON, SENSORS

BASE = {"temperature": 70.0, "vibration": 0.50, "pressure": 100.0,
        "rotational_speed": 1500.0, "torque": 40.0}
DEGRADE = {"temperature": 18.0, "vibration": 1.6, "pressure": -12.0,
           "rotational_speed": -120.0, "torque": 9.0}
NOISE = {"temperature": 2.0, "vibration": 0.12, "pressure": 2.5,
         "rotational_speed": 30.0, "torque": 1.8}
DEGRADATION_SHAPE = 3.0   # degradation accelerates near end of life
SPIKE_PROBABILITY = 0.01


class Machine:
    """One simulated machine that can be stepped forward in time."""

    def __init__(self, machine_id, rng, life=None):
        self.machine_id = machine_id
        self.rng = rng
        self.life = int(life if life is not None else rng.integers(150, 401))
        self.offset = {s: rng.normal(0, NOISE[s] * 1.5) for s in SENSORS}
        self.sensitivity = {s: rng.uniform(0.6, 1.3) for s in SENSORS}
        self.phase = rng.uniform(0, 2 * np.pi)
        self.age = 0

    @property
    def failed(self):
        return self.age >= self.life

    def _reading(self):
        load = 1 + 0.08 * np.sin(2 * np.pi * self.age / 60 + self.phase)
        wear = (self.age / self.life) ** DEGRADATION_SHAPE
        out = {}
        for s in SENSORS:
            out[s] = (BASE[s] + self.offset[s]
                      + self.sensitivity[s] * DEGRADE[s] * wear
                      + self.rng.normal(0, NOISE[s]))
        out["temperature"] += 40 * (load - 1)
        out["torque"] += 40 * (load - 1)
        if self.rng.random() < SPIKE_PROBABILITY:
            out["vibration"] += self.rng.uniform(0.3, 0.8)
        return out

    def step(self):
        """Advance one time step and return the sensor row."""
        row = {"machine_id": self.machine_id, "step": self.age}
        row.update(self._reading())
        row["rul"] = self.life - self.age
        self.age += 1
        return row


def generate_dataset(n_machines=150, seed=42):
    """Simulate n_machines complete run-to-failure histories."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_machines):
        machine = Machine(f"M{i + 1:03d}", rng)
        while not machine.failed:
            rows.append(machine.step())
    df = pd.DataFrame(rows)
    df["failure_soon"] = (df["rul"] <= HORIZON).astype(int)
    return df
