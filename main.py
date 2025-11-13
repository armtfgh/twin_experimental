"""
Simple digital twin of a pipette mixing three dyes plus water.

The simulation is intentionally lightweight: a single pipette moves between
reservoirs, aspirates requested volumes with a small random error, dispenses
into a vial, and reports the estimated mixture color using a Beer-Lambert style
color model. The goal is to capture the feel of a physical process without
heavy dependencies or graphics.
"""
#%%
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

Vec3 = Tuple[float, float, float]


# ---------------------------------------------------------------------------
# Color utilities
# ---------------------------------------------------------------------------

def clamp(value: float, minimum: float = 0.0, maximum: float = 1.0) -> float:
    return max(minimum, min(value, maximum))


def linear_to_srgb_channel(value: float) -> float:
    value = clamp(value)
    if value <= 0.0031308:
        return 12.92 * value
    return 1.055 * (value ** (1 / 2.4)) - 0.055


def srgb_from_linear(linear_rgb: Vec3) -> Vec3:
    return tuple(linear_to_srgb_channel(c) for c in linear_rgb)  # type: ignore[misc]


def srgb8_from_linear(linear_rgb: Vec3) -> Tuple[int, int, int]:
    srgb = srgb_from_linear(linear_rgb)
    return tuple(int(round(clamp(c) * 255)) for c in srgb)


def hex_from_rgb8(rgb: Sequence[int]) -> str:
    r, g, b = (int(clamp(v / 255, 0, 1) * 255) for v in rgb)
    return f"#{r:02X}{g:02X}{b:02X}"


def beer_lambert_linear_rgb(portions: Sequence["LiquidPortion"]) -> Vec3:
    """Return linear RGB based on simple Beer-Lambert absorption."""
    total_volume = sum(p.volume_mL for p in portions)
    if total_volume <= 0:
        return (1.0, 1.0, 1.0)

    optical_density = [0.0, 0.0, 0.0]
    for portion in portions:
        fraction = portion.volume_mL / total_volume
        for idx in range(3):
            optical_density[idx] += (
                portion.absorption[idx]
                * portion.concentration
                * fraction
            )

    return tuple(math.exp(-od) for od in optical_density)  # type: ignore[misc]


@dataclass
class ColorReading:
    linear_rgb: Vec3
    srgb: Vec3
    srgb8: Tuple[int, int, int]
    hex_value: str


def color_reading(portions: Sequence["LiquidPortion"]) -> ColorReading:
    linear = beer_lambert_linear_rgb(portions)
    srgb = srgb_from_linear(linear)
    srgb8 = srgb8_from_linear(linear)
    return ColorReading(linear, srgb, srgb8, hex_from_rgb8(srgb8))


# ---------------------------------------------------------------------------
# Core data objects
# ---------------------------------------------------------------------------


@dataclass
class LiquidPortion:
    source: str
    volume_mL: float
    absorption: Vec3
    concentration: float

    def copy_with_volume(self, volume: float) -> "LiquidPortion":
        return LiquidPortion(self.source, volume, self.absorption, self.concentration)


@dataclass
class Reservoir:
    name: str
    absorption: Vec3
    concentration: float
    volume_mL: float
    position_cm: Tuple[float, float]
    initial_volume_mL: float = field(init=False)

    def __post_init__(self) -> None:
        self.initial_volume_mL = self.volume_mL

    def withdraw(self, amount: float) -> float:
        drawn = min(amount, self.volume_mL)
        self.volume_mL -= drawn
        return drawn

    def level_fraction(self) -> float:
        if self.initial_volume_mL <= 0:
            return 0.0
        return clamp(self.volume_mL / self.initial_volume_mL)


def clone_reservoir(reservoir: Reservoir, *, use_current_volume: bool = False) -> Reservoir:
    volume = reservoir.volume_mL if use_current_volume else reservoir.initial_volume_mL
    return Reservoir(
        name=reservoir.name,
        absorption=reservoir.absorption,
        concentration=reservoir.concentration,
        volume_mL=volume,
        position_cm=reservoir.position_cm,
    )


@dataclass
class Vial:
    capacity_mL: float
    position_cm: Tuple[float, float]
    contents: List[LiquidPortion] = field(default_factory=list)

    def current_volume(self) -> float:
        return sum(portion.volume_mL for portion in self.contents)

    def remaining_capacity(self) -> float:
        return max(0.0, self.capacity_mL - self.current_volume())

    def add_portion(self, portion: LiquidPortion) -> None:
        self.contents.append(portion)

    def add_portions(self, portions: Sequence[LiquidPortion]) -> None:
        for portion in portions:
            self.add_portion(portion)

    def color(self) -> ColorReading:
        return color_reading(self.contents)

    def composition(self) -> Dict[str, float]:
        totals: Dict[str, float] = {}
        for portion in self.contents:
            totals[portion.source] = totals.get(portion.source, 0.0) + portion.volume_mL
        return totals


@dataclass
class Pipette:
    max_volume_mL: float
    carryover_fraction: float
    aspirate_rate_mL_s: float
    dispense_rate_mL_s: float
    move_speed_cm_s: float
    error_pct: float
    position_cm: Tuple[float, float]
    contents: List[LiquidPortion] = field(default_factory=list)

    def current_volume(self) -> float:
        return sum(portion.volume_mL for portion in self.contents)

    def available_capacity(self) -> float:
        return max(0.0, self.max_volume_mL - self.current_volume())

    def move_to(self, target: Tuple[float, float]) -> Tuple[float, float]:
        distance = math.dist(self.position_cm, target)
        self.position_cm = target
        time = distance / self.move_speed_cm_s if self.move_speed_cm_s > 0 else 0.0
        return time, distance

    def aspirate(self, reservoir: Reservoir, requested_volume: float) -> Tuple[float, float, str]:
        capacity = self.available_capacity()
        if capacity <= 0:
            return 0.0, 0.0, "pipette is full"

        target_volume = min(requested_volume, capacity)
        error_factor = 1 + random.uniform(-self.error_pct, self.error_pct)
        target_volume = max(0.0, target_volume * error_factor)

        actual = reservoir.withdraw(target_volume)
        if actual <= 0:
            return 0.0, 0.0, "reservoir is empty"

        self.contents.append(
            LiquidPortion(
                source=reservoir.name,
                volume_mL=actual,
                absorption=reservoir.absorption,
                concentration=reservoir.concentration,
            )
        )

        duration = actual / self.aspirate_rate_mL_s if self.aspirate_rate_mL_s > 0 else 0.0
        note = f"err {error_factor - 1:+.1%}; {reservoir.volume_mL:.1f} mL left"
        return actual, duration, note

    def dispense(self, vial: Vial) -> Tuple[float, float, str]:
        total_before = self.current_volume()
        if total_before <= 0:
            return 0.0, 0.0, "pipette empty"

        deliverable = min(total_before, vial.remaining_capacity())
        if deliverable <= 0:
            return 0.0, 0.0, "vial full"

        deliver_portions: List[LiquidPortion] = []
        new_contents: List[LiquidPortion] = []

        for portion in self.contents:
            share = portion.volume_mL / total_before if total_before > 0 else 0.0
            delivered_component = share * deliverable
            if delivered_component > 0:
                deliver_portions.append(portion.copy_with_volume(delivered_component))

            remaining_component = portion.volume_mL - delivered_component
            carryover_component = delivered_component * self.carryover_fraction
            new_volume = max(0.0, remaining_component + carryover_component)
            if new_volume > 1e-6:
                new_contents.append(portion.copy_with_volume(new_volume))

        vial.add_portions(deliver_portions)
        self.contents = new_contents

        duration = deliverable / self.dispense_rate_mL_s if self.dispense_rate_mL_s > 0 else 0.0
        note = f"carryover keeps {self.current_volume():.3f} mL in pipette"
        return deliverable, duration, note


@dataclass
class Event:
    timestamp_s: float
    action: str
    target: str
    volume_mL: float
    note: str
    duration_s: float = 0.0
    start_pos: Tuple[float, float] | None = None
    end_pos: Tuple[float, float] | None = None
    color_hex: str | None = None


class Simulation:
    def __init__(self) -> None:
        self.time_s = 0.0
        self.events: List[Event] = []

    def advance(self, dt: float) -> None:
        self.time_s += dt

    def record(
        self,
        action: str,
        target: str,
        volume: float = 0.0,
        note: str = "",
        duration: float = 0.0,
        start_pos: Tuple[float, float] | None = None,
        end_pos: Tuple[float, float] | None = None,
        color_hex: str | None = None,
    ) -> None:
        self.events.append(
            Event(
                timestamp_s=self.time_s,
                action=action,
                target=target,
                volume_mL=volume,
                note=note,
                duration_s=duration,
                start_pos=start_pos,
                end_pos=end_pos,
                color_hex=color_hex,
            )
        )


@dataclass
class RunResult:
    events: List[Event]
    vial: Vial
    final_reservoirs: Dict[str, Reservoir]
    initial_reservoirs: Dict[str, Reservoir]
    pipette_start_pos: Tuple[float, float]
    recipe: List[Tuple[str, float]]
    seed: int

    def timeline(self) -> List[Event]:
        return self.events


# ---------------------------------------------------------------------------
# Simulation runner
# ---------------------------------------------------------------------------


def run_recipe(
    sim: Simulation,
    pipette: Pipette,
    vial: Vial,
    reservoirs: Dict[str, Reservoir],
    recipe: Sequence[Tuple[str, float]],
) -> None:
    for step_index, (name, volume) in enumerate(recipe, start=1):
        reservoir = reservoirs[name]

        start_pos = pipette.position_cm
        move_time, distance = pipette.move_to(reservoir.position_cm)
        sim.advance(move_time)
        sim.record(
            "MOVE",
            reservoir.name,
            note=f"{distance:.1f} cm hop",
            duration=move_time,
            start_pos=start_pos,
            end_pos=pipette.position_cm,
        )

        actual, op_time, note = pipette.aspirate(reservoir, volume)
        sim.advance(op_time)
        sim.record(
            "ASPIRATE",
            reservoir.name,
            volume=actual,
            note=note,
            duration=op_time,
            start_pos=pipette.position_cm,
            end_pos=pipette.position_cm,
        )

        start_pos = pipette.position_cm
        move_time, distance = pipette.move_to(vial.position_cm)
        sim.advance(move_time)
        sim.record(
            "MOVE",
            "Vial",
            note=f"{distance:.1f} cm hop",
            duration=move_time,
            start_pos=start_pos,
            end_pos=pipette.position_cm,
        )

        dispensed, op_time, note = pipette.dispense(vial)
        sim.advance(op_time)
        color = vial.color()
        mix_note = f"{note}; mix {color.hex_value}"
        sim.record(
            "DISPENSE",
            "Vial",
            volume=dispensed,
            note=mix_note,
            duration=op_time,
            start_pos=pipette.position_cm,
            end_pos=pipette.position_cm,
            color_hex=color.hex_value,
        )

        sim.record(
            "STATE",
            f"Step {step_index}",
            note=f"Vial {vial.current_volume():.2f} mL / {vial.capacity_mL:.2f} mL",
        )


# ---------------------------------------------------------------------------
# Demo setup
# ---------------------------------------------------------------------------


def build_default_reservoirs() -> Dict[str, Reservoir]:
    return {
        "Crimson": Reservoir(
            name="Crimson",
            absorption=(0.3, 1.8, 1.4),
            concentration=1.0,
            volume_mL=60.0,
            position_cm=(0.0, 0.0),
        ),
        "Cobalt": Reservoir(
            name="Cobalt",
            absorption=(1.6, 1.0, 0.2),
            concentration=1.0,
            volume_mL=60.0,
            position_cm=(0.0, 3.5),
        ),
        "Saffron": Reservoir(
            name="Saffron",
            absorption=(0.4, 0.6, 2.2),
            concentration=1.0,
            volume_mL=60.0,
            position_cm=(0.0, 7.0),
        ),
        "Water": Reservoir(
            name="Water",
            absorption=(0.02, 0.02, 0.02),
            concentration=1.0,
            volume_mL=120.0,
            position_cm=(0.0, 10.5),
        ),
    }


def demo_recipe() -> List[Tuple[str, float]]:
    return [
        ("Crimson", 0.20),
        ("Water", 0.05),
        ("Cobalt", 0.15),
        ("Saffron", 0.12),
        ("Water", 0.30),
    ]


# ---------------------------------------------------------------------------
# Digital twin API
# ---------------------------------------------------------------------------


class DigitalTwin:
    def __init__(
        self,
        reservoir_templates: Dict[str, Reservoir] | None = None,
        pipette_config: Dict[str, float | Tuple[float, float]] | None = None,
        vial_capacity_mL: float = 2.0,
        vial_position_cm: Tuple[float, float] = (5.0, 3.5),
    ) -> None:
        templates = reservoir_templates or build_default_reservoirs()
        self._reservoir_templates = self._clone_map(templates)
        default_pipette = {
            "max_volume_mL": 0.25,
            "carryover_fraction": 0.02,
            "aspirate_rate_mL_s": 0.1,
            "dispense_rate_mL_s": 0.15,
            "move_speed_cm_s": 5.0,
            "error_pct": 0.03,
            "position_cm": (2.5, 3.5),
        }
        self._pipette_config = dict(default_pipette | (pipette_config or {}))
        self._vial_capacity_mL = vial_capacity_mL
        self._vial_position_cm = vial_position_cm

    def _clone_map(
        self,
        reservoirs: Dict[str, Reservoir],
        *,
        use_current_volume: bool = False,
    ) -> Dict[str, Reservoir]:
        return {
            name: clone_reservoir(res, use_current_volume=use_current_volume)
            for name, res in reservoirs.items()
        }

    def new_reservoirs(self) -> Dict[str, Reservoir]:
        """Return brand new reservoirs based on the templates."""
        return self._clone_map(self._reservoir_templates)

    def new_pipette(self) -> Pipette:
        return Pipette(**self._pipette_config)

    def new_vial(self) -> Vial:
        return Vial(capacity_mL=self._vial_capacity_mL, position_cm=self._vial_position_cm)

    def run(self, recipe: Sequence[Tuple[str, float]], seed: int = 7) -> RunResult:
        """Execute a recipe and return the timeline/event log."""
        random.seed(seed)
        reservoirs = self.new_reservoirs()
        reservoirs_initial = self._clone_map(reservoirs)
        vial = Vial(capacity_mL=self._vial_capacity_mL, position_cm=self._vial_position_cm)
        pipette = Pipette(**self._pipette_config)
        pipette_start = pipette.position_cm

        sim = Simulation()
        run_recipe(sim, pipette, vial, reservoirs, recipe)

        return RunResult(
            events=list(sim.events),
            vial=vial,
            final_reservoirs=self._clone_map(reservoirs, use_current_volume=True),
            initial_reservoirs=reservoirs_initial,
            pipette_start_pos=pipette_start,
            recipe=[(name, vol) for name, vol in recipe],
            seed=seed,
        )


class ManualSession:
    """Step-by-step controller for manual operation."""

    def __init__(self, twin: DigitalTwin, seed: int = 7) -> None:
        self.twin = twin
        self.seed = seed
        random.seed(seed)
        self.reservoirs = twin.new_reservoirs()
        self.initial_reservoirs = twin._clone_map(self.reservoirs)
        self.vial = twin.new_vial()
        self.pipette = twin.new_pipette()
        self.sim = Simulation()
        self.pipette_start_pos = self.pipette.position_cm
        self.current_target: str | None = None
        self.commands: List[str] = []
        self.default_volume = 0.10
        self._positions: Dict[str, Tuple[float, float]] = {
            name: res.position_cm for name, res in self.reservoirs.items()
        }
        self._positions["Vial"] = self.vial.position_cm

    def available_targets(self) -> List[str]:
        return list(self._positions.keys())

    def _resolve_target(self, name: str) -> str | None:
        lookup = name.strip().lower()
        for target in self._positions:
            if target.lower() == lookup:
                return target
        return None

    def move(self, target_name: str) -> Tuple[bool, str]:
        resolved = self._resolve_target(target_name)
        if not resolved:
            return False, f"Unknown target '{target_name}'. Options: {', '.join(self.available_targets())}"
        start = self.pipette.position_cm
        move_time, distance = self.pipette.move_to(self._positions[resolved])
        self.sim.advance(move_time)
        self.sim.record(
            "MOVE",
            resolved,
            note=f"{distance:.1f} cm hop",
            duration=move_time,
            start_pos=start,
            end_pos=self.pipette.position_cm,
        )
        self.current_target = resolved
        return True, f"Moved to {resolved} ({distance:.1f} cm)."

    def aspirate(self, volume_mL: float | None = None) -> Tuple[bool, str]:
        volume = self.default_volume if volume_mL is None else max(0.0, volume_mL)
        if self.current_target is None or self.current_target not in self.reservoirs:
            return False, "Move to a reservoir before aspirating."
        reservoir = self.reservoirs[self.current_target]
        actual, duration, note = self.pipette.aspirate(reservoir, volume)
        if actual <= 0:
            return False, note
        self.sim.advance(duration)
        self.sim.record(
            "ASPIRATE",
            reservoir.name,
            volume=actual,
            note=note,
            duration=duration,
            start_pos=self.pipette.position_cm,
            end_pos=self.pipette.position_cm,
        )
        return True, f"Aspirated {actual:.3f} mL from {reservoir.name} ({note})."

    def dispense(self) -> Tuple[bool, str]:
        if self.current_target != "Vial":
            return False, "Move to the Vial before dispensing."
        delivered, duration, note = self.pipette.dispense(self.vial)
        if delivered <= 0:
            return False, note
        self.sim.advance(duration)
        color = self.vial.color()
        self.sim.record(
            "DISPENSE",
            "Vial",
            volume=delivered,
            note=f"{note}; mix {color.hex_value}",
            duration=duration,
            start_pos=self.pipette.position_cm,
            end_pos=self.pipette.position_cm,
            color_hex=color.hex_value,
        )
        return True, f"Dispensed {delivered:.3f} mL into vial ({note})."

    def status_summary(self) -> str:
        pipette_vol = self.pipette.current_volume()
        capacity = self.pipette.max_volume_mL
        contents = ", ".join(
            f"{p.source}:{p.volume_mL:.2f}" for p in self.pipette.contents
        ) or "empty"
        info = [
            f"Pipette: {pipette_vol:.3f}/{capacity:.3f} mL ({contents})",
            f"Vial: {self.vial.current_volume():.3f}/{self.vial.capacity_mL:.3f} mL, color {self.vial.color().hex_value}",
            "Reservoirs: "
            + ", ".join(f"{name} {res.volume_mL:.1f} mL" for name, res in self.reservoirs.items()),
        ]
        if self.current_target:
            info.append(f"Current position: {self.current_target}")
        info.append(f"Default aspirate volume: {self.default_volume:.2f} mL")
        return "\n".join(info)

    def log_tail(self, count: int = 8) -> List[str]:
        return [
            f"{evt.action:<9} {evt.target:<8} {evt.note}"
            for evt in self.sim.events[-count:]
        ]

    def set_default_volume(self, volume: float) -> None:
        self.default_volume = clamp(volume, 0.0, self.pipette.max_volume_mL)

    def append_command(self, command: str) -> None:
        self.commands.append(command)

    def finish(self) -> RunResult:
        return RunResult(
            events=list(self.sim.events),
            vial=self.vial,
            final_reservoirs=self.twin._clone_map(self.reservoirs, use_current_volume=True),
            initial_reservoirs=self.initial_reservoirs,
            pipette_start_pos=self.pipette_start_pos,
            recipe=[(cmd, 0.0) for cmd in self.commands],
            seed=self.seed,
        )


def print_timeline(events: Sequence[Event]) -> None:
    print(" Time  | Action    | Target   | Vol (mL) | Notes")
    print("-" * 70)
    for event in events:
        print(
            f"{event.timestamp_s:6.1f}s | "
            f"{event.action:9} | "
            f"{event.target:8} | "
            f"{event.volume_mL:7.2f} | "
            f"{event.note}"
        )


def print_summary(vial: Vial) -> None:
    reading = vial.color()
    print("\nFinal mixture:")
    print(f"  Volume: {vial.current_volume():.2f} mL / {vial.capacity_mL:.2f} mL")
    print(f"  Color : {reading.hex_value}  (sRGB {reading.srgb8})")
    print("  Composition:")
    for name, volume in vial.composition().items():
        print(f"    - {name:<7} {volume:.2f} mL")


def manual_text_loop(session: ManualSession) -> RunResult:
    print(
        "\nManual mode commands:\n"
        "  move <target>|m   - Move pipette to a reservoir or 'vial'\n"
        "  aspirate [vol]|a  - Aspirate volume in mL (defaults to current preset)\n"
        "  dispense|d        - Dispense into the vial\n"
        "  volume <val>|v    - Set default aspirate volume in mL\n"
        "  status|s          - Show pipette, vial, and reservoir state\n"
        "  log|l             - Show recent event log\n"
        "  help|h            - Show this help again\n"
        "  quit|exit|q       - Finish manual session\n"
    )
    print(f"Targets: {', '.join(session.available_targets())}")
    while True:
        try:
            raw = input("manual> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[info] Manual session interrupted by user.")
            break
        if not raw:
            continue
        lowered = raw.lower()
        if lowered in {"quit", "exit", "q"}:
            break
        parts = raw.split()
        cmd = parts[0].lower()
        args = parts[1:]

        def respond(success: bool, message: str) -> None:
            icon = "✅" if success else "⚠️"
            print(f"{icon} {message}")

        if cmd in {"move", "m"}:
            if not args:
                respond(False, "Specify a target, e.g., 'move Crimson'")
                continue
            session.append_command(raw)
            ok, msg = session.move(" ".join(args))
            respond(ok, msg)
        elif cmd in {"aspirate", "a"}:
            volume = None
            if args:
                try:
                    volume = float(args[0])
                except ValueError:
                    respond(False, f"Invalid volume '{args[0]}'.")
                    continue
            session.append_command(raw)
            ok, msg = session.aspirate(volume)
            respond(ok, msg)
        elif cmd in {"dispense", "d"}:
            session.append_command(raw)
            ok, msg = session.dispense()
            respond(ok, msg)
        elif cmd in {"volume", "v"}:
            if not args:
                print(f"Current default volume: {session.default_volume:.2f} mL")
                continue
            try:
                value = float(args[0])
            except ValueError:
                respond(False, f"Invalid number '{args[0]}'.")
                continue
            session.set_default_volume(value)
            print(f"Default aspirate volume set to {session.default_volume:.2f} mL.")
        elif cmd in {"status", "s"}:
            print(session.status_summary())
        elif cmd in {"log", "l"}:
            tail = session.log_tail()
            print("Recent events:")
            for line in tail or ["<none>"]:
                print("  ", line)
        elif cmd in {"help", "h"}:
            print(
                "Commands: move/m, aspirate/a, dispense/d, volume/v, status/s, log/l, help/h, quit/q"
            )
        else:
            respond(False, f"Unknown command '{cmd}'. Type 'help' for options.")

    print("[info] Manual session finished.")
    return session.finish()


def manual_pygame_loop(session: ManualSession) -> RunResult:
    try:
        import pygame
    except ImportError:
        print("PyGame is not installed. Falling back to text-based manual mode.")
        return manual_text_loop(session)

    try:
        pygame.init()
        screen = pygame.display.set_mode((1000, 640))
        pygame.display.set_caption("Manual Control — Digital Twin")
    except pygame.error as exc:
        print(
            "PyGame could not open a window (missing display?). "
            f"Falling back to manual text mode. Details: {exc}"
        )
        return manual_text_loop(session)

    font = pygame.font.SysFont("Arial", 18)
    small_font = pygame.font.SysFont("Arial", 14)
    clock = pygame.time.Clock()
    scale = 55.0
    origin = (180.0, 70.0)

    def world_to_screen(pos: Tuple[float, float]) -> Tuple[int, int]:
        return (
            int(origin[0] + pos[0] * scale),
            int(origin[1] + pos[1] * scale),
        )

    target_order = list(session.reservoirs.keys()) + ["Vial"]
    key_codes = [
        pygame.K_1,
        pygame.K_2,
        pygame.K_3,
        pygame.K_4,
        pygame.K_5,
        pygame.K_6,
        pygame.K_7,
        pygame.K_8,
        pygame.K_9,
    ]
    target_key_map = {
        key_codes[idx]: target
        for idx, target in enumerate(target_order)
        if idx < len(key_codes)
    }

    messages: List[Tuple[bool, str]] = []

    def push_message(success: bool, text: str) -> None:
        messages.append((success, text))
        if len(messages) > 6:
            messages.pop(0)

    def draw_reservoirs() -> None:
        for name, reservoir in session.reservoirs.items():
            center = world_to_screen(reservoir.position_cm)
            rect = pygame.Rect(center[0] - 35, center[1] - 50, 70, 100)
            pygame.draw.rect(screen, (50, 52, 60), rect, 2, border_radius=6)
            level_frac = (
                reservoir.volume_mL / reservoir.initial_volume_mL
                if reservoir.initial_volume_mL > 0
                else 0.0
            )
            level_frac = clamp(level_frac, 0.0, 1.0)
            fill_height = int((rect.height - 6) * level_frac)
            fill_rect = pygame.Rect(
                rect.left + 3,
                rect.bottom - 3 - fill_height,
                rect.width - 6,
                fill_height,
            )
            pygame.draw.rect(screen, approx_reservoir_color(reservoir.absorption), fill_rect, border_radius=4)
            label = font.render(name, True, (220, 220, 230))
            screen.blit(label, (rect.centerx - label.get_width() // 2, rect.top - 24))
            screen.blit(
                small_font.render(f"{reservoir.volume_mL:.1f} mL", True, (200, 200, 210)),
                (rect.centerx - 34, rect.bottom + 6),
            )

    def draw_vial() -> None:
        center = world_to_screen(session.vial.position_cm)
        rect = pygame.Rect(center[0] - 40, center[1] - 65, 80, 130)
        pygame.draw.rect(screen, (70, 70, 80), rect, 2, border_radius=8)
        fill_frac = clamp(
            session.vial.current_volume() / session.vial.capacity_mL
            if session.vial.capacity_mL
            else 0.0,
            0.0,
            1.0,
        )
        fill_height = int((rect.height - 6) * fill_frac)
        fill_rect = pygame.Rect(
            rect.left + 3,
            rect.bottom - 3 - fill_height,
            rect.width - 6,
            fill_height,
        )
        pygame.draw.rect(
            screen,
            hex_to_rgb(session.vial.color().hex_value),
            fill_rect,
            border_radius=6,
        )
        label = font.render("Vial", True, (220, 220, 230))
        screen.blit(label, (rect.centerx - label.get_width() // 2, rect.top - 28))
        screen.blit(
            small_font.render(
                f"{session.vial.current_volume():.2f}/{session.vial.capacity_mL:.2f} mL",
                True,
                (200, 200, 210),
            ),
            (rect.left, rect.bottom + 6),
        )

    def draw_pipette() -> None:
        pos = world_to_screen(session.pipette.position_cm)
        pygame.draw.circle(screen, (240, 240, 240), pos, 12)
        pygame.draw.circle(screen, (100, 180, 255), pos, 12, 2)

    def draw_messages() -> None:
        area = pygame.Rect(620, 40, 340, 200)
        pygame.draw.rect(screen, (20, 22, 30), area, border_radius=8)
        pygame.draw.rect(screen, (60, 62, 70), area, 2, border_radius=8)
        title = font.render("Messages", True, (230, 230, 230))
        screen.blit(title, (area.left + 12, area.top + 8))
        for idx, (success, text) in enumerate(messages[-6:]):
            color = (140, 220, 160) if success else (240, 190, 100)
            screen.blit(
                small_font.render(text, True, color),
                (area.left + 12, area.top + 36 + idx * 20),
            )

    def draw_log_tail() -> None:
        area = pygame.Rect(620, 260, 340, 170)
        pygame.draw.rect(screen, (20, 22, 30), area, border_radius=8)
        pygame.draw.rect(screen, (60, 62, 70), area, 2, border_radius=8)
        title = font.render("Event Log", True, (230, 230, 230))
        screen.blit(title, (area.left + 12, area.top + 8))
        tail = session.log_tail()
        for idx, line in enumerate(reversed(tail[-7:])):
            screen.blit(
                small_font.render(line, True, (200, 200, 210)),
                (area.left + 12, area.top + 36 + idx * 18),
            )

    def draw_status_bar() -> None:
        area = pygame.Rect(20, 560, 940, 60)
        pygame.draw.rect(screen, (20, 22, 30), area, border_radius=8)
        pygame.draw.rect(screen, (60, 62, 70), area, 2, border_radius=8)
        info = (
            f"Pipette {session.pipette.current_volume():.2f}/{session.pipette.max_volume_mL:.2f} mL  |  "
            f"Default vol {session.default_volume:.2f} mL  |  "
            f"Current target: {session.current_target or 'None'}"
        )
        screen.blit(small_font.render(info, True, (220, 220, 230)), (area.left + 12, area.top + 10))
        instructions = (
            "Controls: 1-n move, A aspirate, D/Space dispense, +/- adjust vol, L log, TAB status, ESC quit"
        )
        screen.blit(small_font.render(instructions, True, (200, 200, 210)), (area.left + 12, area.top + 32))

    def push_status_summary() -> None:
        for line in session.status_summary().splitlines():
            push_message(True, line)

    def push_log_tail() -> None:
        tail = session.log_tail()
        if not tail:
            push_message(True, "<no events yet>")
        else:
            for line in tail[-4:]:
                push_message(True, line)

    push_message(True, "Manual PyGame control ready. Use number keys to move.")

    running = True
    while running:
        dt = clock.tick(60) / 1000.0  # noqa: F841
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key in target_key_map:
                    target = target_key_map[event.key]
                    session.append_command(f"move {target} [key]")
                    ok, msg = session.move(target)
                    push_message(ok, msg)
                elif event.key == pygame.K_a:
                    session.append_command("aspirate [key]")
                    ok, msg = session.aspirate()
                    push_message(ok, msg)
                elif event.key in (pygame.K_d, pygame.K_SPACE):
                    session.append_command("dispense [key]")
                    ok, msg = session.dispense()
                    push_message(ok, msg)
                elif event.key in (pygame.K_MINUS, pygame.K_UNDERSCORE):
                    session.set_default_volume(session.default_volume - 0.01)
                    push_message(True, f"Default volume {session.default_volume:.2f} mL")
                elif event.key in (pygame.K_EQUALS, pygame.K_PLUS):
                    session.set_default_volume(session.default_volume + 0.01)
                    push_message(True, f"Default volume {session.default_volume:.2f} mL")
                elif event.key == pygame.K_TAB:
                    push_status_summary()
                elif event.key == pygame.K_l:
                    push_log_tail()

        screen.fill((12, 13, 18))
        draw_reservoirs()
        draw_vial()
        draw_pipette()
        draw_messages()
        draw_log_tail()
        draw_status_bar()
        pygame.display.flip()

    pygame.quit()
    print("[info] Manual PyGame session finished.")
    return session.finish()


# ---------------------------------------------------------------------------
# PyGame visualization
# ---------------------------------------------------------------------------


def hex_to_rgb(hex_value: str) -> Tuple[int, int, int]:
    sanitized = hex_value.lstrip("#")
    if len(sanitized) != 6:
        return (255, 255, 255)
    return tuple(int(sanitized[i : i + 2], 16) for i in (0, 2, 4))


def approx_reservoir_color(absorption: Vec3) -> Tuple[int, int, int]:
    linear = tuple(math.exp(-a * 0.7) for a in absorption)  # gentle attenuation
    return srgb8_from_linear(linear)


def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def visualize_with_pygame(
    events: Sequence[Event],
    reservoirs: Dict[str, Reservoir],
    vial_capacity: float,
    pipette_start_pos: Tuple[float, float],
    target_hex: str | None = None,
    wait_for_continue: bool = False,
) -> None:
    try:
        import pygame
    except ImportError:  # pragma: no cover - convenience path
        print("PyGame is not installed. Install it with `pip install pygame` to enable visualization.")
        return

    try:
        pygame.init()
        screen = pygame.display.set_mode((1000, 640))
        pygame.display.set_caption("Mini Dye Mixing Digital Twin")
    except pygame.error as exc:  # pragma: no cover - environment specific
        print(
            "PyGame could not open a window. This usually happens when no GUI/display "
            f"is available (e.g., running headless). Details: {exc}"
        )
        return

    font = pygame.font.SysFont("Arial", 18)
    small_font = pygame.font.SysFont("Arial", 14)
    clock = pygame.time.Clock()

    scale = 55.0
    origin = (180.0, 70.0)

    def world_to_screen(pos: Tuple[float, float]) -> Tuple[int, int]:
        return (
            int(origin[0] + pos[0] * scale),
            int(origin[1] + pos[1] * scale),
        )

    def draw_text(surface, text: str, pos: Tuple[int, int], color=(230, 230, 230)) -> None:
        surface.blit(small_font.render(text, True, color), pos)

    reservoir_levels = {
        name: {
            "current": res.volume_mL,
            "initial": res.initial_volume_mL,
            "color": approx_reservoir_color(res.absorption),
            "pos": world_to_screen(res.position_cm),
        }
        for name, res in reservoirs.items()
    }
    vial_volume = 0.0
    vial_color_hex = "#FFFFFF"
    pipette_pos = world_to_screen(pipette_start_pos)
    current_event_index = 0
    event_progress = 0.0
    log_lines: List[str] = []
    timeline_done = False
    target_hex_display = target_hex.upper() if target_hex else None

    def draw_reservoirs() -> None:
        for name, data in reservoir_levels.items():
            center = data["pos"]
            rect = pygame.Rect(center[0] - 35, center[1] - 50, 70, 100)
            pygame.draw.rect(screen, (50, 52, 60), rect, 2, border_radius=6)
            level_frac = data["current"] / data["initial"] if data["initial"] > 0 else 0.0
            level_frac = clamp(level_frac, 0.0, 1.0)
            fill_height = int((rect.height - 6) * level_frac)
            fill_rect = pygame.Rect(
                rect.left + 3,
                rect.bottom - 3 - fill_height,
                rect.width - 6,
                fill_height,
            )
            pygame.draw.rect(screen, data["color"], fill_rect, border_radius=4)
            label = font.render(name, True, (220, 220, 230))
            screen.blit(label, (rect.centerx - label.get_width() // 2, rect.top - 24))
            draw_text(
                screen,
                f"{data['current']:.1f} mL",
                (rect.centerx - 32, rect.bottom + 6),
            )

    def draw_vial() -> None:
        center = world_to_screen((5.0, 3.5))
        rect = pygame.Rect(center[0] - 40, center[1] - 65, 80, 130)
        pygame.draw.rect(screen, (70, 70, 80), rect, 2, border_radius=8)
        fill_frac = clamp(vial_volume / vial_capacity if vial_capacity else 0.0, 0.0, 1.0)
        fill_height = int((rect.height - 6) * fill_frac)
        fill_rect = pygame.Rect(
            rect.left + 3,
            rect.bottom - 3 - fill_height,
            rect.width - 6,
            fill_height,
        )
        pygame.draw.rect(screen, hex_to_rgb(vial_color_hex), fill_rect, border_radius=6)
        label = font.render("Vial", True, (220, 220, 230))
        screen.blit(label, (rect.centerx - label.get_width() // 2, rect.top - 28))
        draw_text(screen, f"{vial_volume:.2f} / {vial_capacity:.2f} mL", (rect.left, rect.bottom + 6))

    def draw_pipette() -> None:
        pygame.draw.circle(screen, (240, 240, 240), pipette_pos, 12)
        pygame.draw.circle(screen, (100, 180, 255), pipette_pos, 12, 2)

    def draw_log() -> None:
        area = pygame.Rect(620, 40, 340, 240)
        pygame.draw.rect(screen, (24, 26, 34), area, border_radius=8)
        pygame.draw.rect(screen, (60, 62, 70), area, 2, border_radius=8)
        title = font.render("Event Log", True, (230, 230, 230))
        screen.blit(title, (area.left + 12, area.top + 8))
        for idx, line in enumerate(reversed(log_lines[-8:])):
            draw_text(screen, line, (area.left + 12, area.top + 36 + idx * 20))

    def draw_color_panel() -> None:
        panel_height = 220 if target_hex_display else 150
        panel_rect = pygame.Rect(620, 320, 340, panel_height)
        pygame.draw.rect(screen, (24, 26, 34), panel_rect, border_radius=8)
        pygame.draw.rect(screen, (60, 62, 70), panel_rect, 2, border_radius=8)

        current_rect = pygame.Rect(panel_rect.left + 16, panel_rect.top + 40, panel_rect.width - 32, 70)
        pygame.draw.rect(screen, hex_to_rgb(vial_color_hex), current_rect, border_radius=12)
        draw_text(screen, "Current mix", (panel_rect.left + 16, panel_rect.top + 14))
        draw_text(
            screen,
            f"{vial_volume:.2f} mL  {vial_color_hex}",
            (panel_rect.left + 16, current_rect.bottom + 6),
        )

        if target_hex_display:
            target_rect = pygame.Rect(panel_rect.left + 16, current_rect.bottom + 36, panel_rect.width - 32, 60)
            pygame.draw.rect(screen, hex_to_rgb(target_hex_display), target_rect, border_radius=12)
            draw_text(screen, "Target mix", (panel_rect.left + 16, target_rect.top - 24))
            draw_text(
                screen,
                target_hex_display,
                (panel_rect.left + 16, target_rect.bottom + 4),
            )

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return
            if event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    pygame.quit()
                    return
                if wait_for_continue and timeline_done and event.key in (pygame.K_RETURN, pygame.K_SPACE):
                    pygame.quit()
                    return

        dt = clock.tick(60) / 1000.0
        screen.fill((12, 13, 18))

        if not timeline_done and current_event_index < len(events):
            evt = events[current_event_index]
            duration = evt.duration_s if evt.duration_s > 0 else 0.4
            event_progress += dt
            t = clamp(event_progress / duration if duration > 0 else 1.0)
            if evt.action == "MOVE" and evt.start_pos and evt.end_pos:
                start = world_to_screen(evt.start_pos)
                end = world_to_screen(evt.end_pos)
                pipette_pos = (
                    int(lerp(start[0], end[0], t)),
                    int(lerp(start[1], end[1], t)),
                )
            elif evt.end_pos:
                pipette_pos = world_to_screen(evt.end_pos)

            if event_progress >= duration - 1e-6:
                event_progress = 0.0
                if evt.action == "ASPIRATE":
                    data = reservoir_levels.get(evt.target)
                    if data:
                        data["current"] = clamp(data["current"] - evt.volume_mL, 0.0, data["initial"])
                elif evt.action == "DISPENSE":
                    vial_volume = clamp(vial_volume + evt.volume_mL, 0.0, vial_capacity)
                    if evt.color_hex:
                        vial_color_hex = evt.color_hex
                log_lines.append(f"{evt.action:<9} {evt.target:<8} {evt.note}")
                current_event_index += 1
                if current_event_index >= len(events):
                    timeline_done = True
        else:
            timeline_done = True

        draw_reservoirs()
        draw_vial()
        draw_pipette()
        draw_log()
        draw_color_panel()

        status_text = (
            f"Event {min(current_event_index + 1, len(events))}/{len(events)}"
            if events
            else "No events"
        )
        draw_text(screen, status_text, (20, 20))
        if wait_for_continue and timeline_done:
            draw_text(screen, "Timeline done. Press ENTER to continue or ESC to exit.", (20, 40))
        else:
            draw_text(screen, "Press ESC to exit", (20, 40))

        pygame.display.flip()


SIMULATION_MODE = "manual-live"  # options: "text", "pygame", "both", "manual", "manual-pygame", "manual-live"
SIMULATION_SEED = 7


def main() -> None:
    mode = (SIMULATION_MODE or "text").strip().lower()
    twin = DigitalTwin()

    manual_live_modes = {"manual-live"}
    if mode in manual_live_modes:
        session = ManualSession(twin, seed=SIMULATION_SEED)
        run = manual_pygame_loop(session)
        print_timeline(run.events)
        print_summary(run.vial)
        return

    manual_modes = {"manual", "manual-pygame", "manual-both"}
    if mode in manual_modes:
        session = ManualSession(twin, seed=SIMULATION_SEED)
        run = manual_text_loop(session)
        print_timeline(run.events)
        print_summary(run.vial)
        if mode in {"manual-pygame", "manual-both"}:
            print("\n[info] Launching PyGame visualizer (close the window to finish)...")
            visualize_with_pygame(
                run.events,
                run.initial_reservoirs,
                run.vial.capacity_mL,
                run.pipette_start_pos,
            )
        return

    if mode not in {"text", "pygame", "both"}:
        print(f"[info] Unknown SIMULATION_MODE '{SIMULATION_MODE}', defaulting to 'text'.")
        mode = "text"

    run = twin.run(recipe=demo_recipe(), seed=SIMULATION_SEED)

    if mode in ("text", "both"):
        print_timeline(run.events)
        print_summary(run.vial)

    if mode in ("pygame", "both"):
        print("\n[info] Launching PyGame visualizer (close the window to finish)...")
        visualize_with_pygame(
            run.events,
            run.initial_reservoirs,
            run.vial.capacity_mL,
            run.pipette_start_pos,
        )


if __name__ == "__main__":
    main()

# %%
