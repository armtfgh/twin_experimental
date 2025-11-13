"""
Minimal 3D visualization for the dye-mixing digital twin.

This viewer replays a run (list of Events) inside a simple OpenGL scene using
pyglet. The pipette moves above the reservoirs and vial, giving a sense of depth
and timing. It is not meant to be photorealistic—just a quick “lab bench” view.

Usage:
    pip install pyglet
    python viewer3d.py --mode demo --seed 7

Or reuse the helper function `playback_3d(run_result, target_hex=None)` to
visualize agent runs.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

try:
    import pyglet
    from pyglet.gl import gl as gl_core, lib as gl_lib
except ImportError as exc:  # pragma: no cover - requires pyglet runtime
    raise SystemExit(
        "pyglet is required for the 3D viewer. Install it with `pip install pyglet`."
    ) from exc

gl = gl_core
glMatrixMode = gl_lib.GL.glMatrixMode
glLoadIdentity = gl_lib.GL.glLoadIdentity


def load_gl_matrix(values: Sequence[float]) -> None:
    array_type = gl.GLfloat * 16
    gl.glMultMatrixf(array_type(*values))


def perspective(fov_deg: float, aspect: float, z_near: float, z_far: float) -> None:
    f = 1.0 / math.tan(math.radians(fov_deg) / 2)
    nf = 1.0 / (z_near - z_far)
    matrix = (
        f / aspect,
        0.0,
        0.0,
        0.0,
        0.0,
        f,
        0.0,
        0.0,
        0.0,
        0.0,
        (z_far + z_near) * nf,
        -1.0,
        0.0,
        0.0,
        (2 * z_far * z_near) * nf,
        0.0,
    )
    load_gl_matrix(matrix)


def normalize(vec: Vec3) -> Vec3:
    length = math.sqrt(vec[0] ** 2 + vec[1] ** 2 + vec[2] ** 2)
    if length == 0:
        return (0.0, 0.0, 0.0)
    return (vec[0] / length, vec[1] / length, vec[2] / length)


def cross(a: Vec3, b: Vec3) -> Vec3:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def look_at(eye: Vec3, center: Vec3, up: Vec3) -> None:
    f = normalize((center[0] - eye[0], center[1] - eye[1], center[2] - eye[2]))
    up_norm = normalize(up)
    s = normalize(cross(f, up_norm))
    u = cross(s, f)
    matrix = (
        s[0],
        u[0],
        -f[0],
        0.0,
        s[1],
        u[1],
        -f[1],
        0.0,
        s[2],
        u[2],
        -f[2],
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
    )
    load_gl_matrix(matrix)
    gl.glTranslatef(-eye[0], -eye[1], -eye[2])

from main import (
    DigitalTwin,
    Event,
    RunResult,
    Reservoir,
    clone_reservoir,
    demo_recipe,
    hex_to_rgb,
    lerp,
)

Vec3 = Tuple[float, float, float]


@dataclass
class PlaybackState:
    events: Sequence[Event]
    reservoirs: Dict[str, Reservoir]
    vial_capacity: float
    pipette_pos: Vec3
    current_event_index: int = 0
    event_progress: float = 0.0
    done: bool = False
    vial_volume: float = 0.0
    vial_color_hex: str = "#FFFFFF"


class Twin3DViewer(pyglet.window.Window):
    def __init__(self, run: RunResult, target_hex: str | None = None) -> None:
        super().__init__(width=1100, height=720, caption="Digital Twin 3D Viewer", resizable=False)
        gl.glEnable(gl.GL_DEPTH_TEST)
        self.run = run
        self.state = PlaybackState(
            events=run.events,
            reservoirs={name: clone_reservoir(res, use_current_volume=True) for name, res in run.initial_reservoirs.items()},
            vial_capacity=run.vial.capacity_mL,
            pipette_pos=(*run.pipette_start_pos, 6.0),
        )
        self.world_scale = 0.2  # world scale (cm -> OpenGL units)
        self.camera_distance = 22.0
        self.camera_yaw = 40.0
        self.camera_pitch = 35.0
        self.target_hex = target_hex.upper() if target_hex else None
        self.batch = pyglet.graphics.Batch()
        self.info_label = pyglet.text.Label(
            "",
            font_size=12,
            x=10,
            y=self.height - 20,
            anchor_x="left",
            anchor_y="center",
            color=(230, 230, 230, 255),
            batch=self.batch,
        )
        self.instruction_label = pyglet.text.Label(
            "Controls: ←→ yaw, ↑↓ pitch, +/- zoom, SPACE pause/play, ESC exit",
            font_size=11,
            x=10,
            y=20,
            anchor_x="left",
            anchor_y="bottom",
            color=(220, 220, 220, 255),
            batch=self.batch,
        )
        self.paused = False
        self.dt_accumulator = 0.0
        pyglet.clock.schedule_interval(self.update, 1 / 60.0)

    # ------------------------------------------------------------------ camera
    def on_resize(self, width: int, height: int) -> None:  # pragma: no cover
        super().on_resize(width, height)
        gl.glViewport(0, 0, width, height)
        glMatrixMode(gl.GL_PROJECTION)
        glLoadIdentity()
        perspective(45.0, width / float(height), 0.1, 200.0)
        glMatrixMode(gl.GL_MODELVIEW)

    def set_camera(self) -> None:
        glMatrixMode(gl.GL_MODELVIEW)
        glLoadIdentity()
        eye_x = self.camera_distance * math.cos(math.radians(self.camera_yaw)) * math.cos(
            math.radians(self.camera_pitch)
        )
        eye_y = self.camera_distance * math.sin(math.radians(self.camera_pitch))
        eye_z = self.camera_distance * math.sin(math.radians(self.camera_yaw)) * math.cos(
            math.radians(self.camera_pitch)
        )
        look_at(
            (eye_x, eye_y + 5.0, eye_z),
            (5.0, 0.5, 3.0),
            (0.0, 1.0, 0.0),
        )

    # ----------------------------------------------------------------- drawing
    def on_draw(self) -> None:  # pragma: no cover - requires OpenGL
        self.clear()
        self.set_camera()
        self.draw_floor()
        self.draw_reservoirs()
        self.draw_vial()
        self.draw_pipette()
        self.draw_overlay()

    def draw_floor(self) -> None:
        gl.glColor3f(0.15, 0.17, 0.2)
        gl.glBegin(gl.GL_QUADS)
        gl.glVertex3f(-5, 0, -5)
        gl.glVertex3f(15, 0, -5)
        gl.glVertex3f(15, 0, 15)
        gl.glVertex3f(-5, 0, 15)
        gl.glEnd()

        gl.glColor3f(0.25, 0.28, 0.32)
        gl.glBegin(gl.GL_LINES)
        for i in range(-5, 16):
            gl.glVertex3f(i, 0.001, -5)
            gl.glVertex3f(i, 0.001, 15)
            gl.glVertex3f(-5, 0.001, i)
            gl.glVertex3f(15, 0.001, i)
        gl.glEnd()

    def draw_box(self, center: Tuple[float, float, float], size: Tuple[float, float, float], color: Tuple[float, float, float]) -> None:
        cx, cy, cz = center
        sx, sy, sz = size
        gl.glColor3f(*color)
        hw, hh, hd = sx / 2, sy / 2, sz / 2
        vertices = [
            (cx - hw, cy - hh, cz - hd),
            (cx + hw, cy - hh, cz - hd),
            (cx + hw, cy + hh, cz - hd),
            (cx - hw, cy + hh, cz - hd),
            (cx - hw, cy - hh, cz + hd),
            (cx + hw, cy - hh, cz + hd),
            (cx + hw, cy + hh, cz + hd),
            (cx - hw, cy + hh, cz + hd),
        ]
        faces = [
            (0, 1, 2, 3),
            (4, 5, 6, 7),
            (0, 1, 5, 4),
            (2, 3, 7, 6),
            (1, 2, 6, 5),
            (0, 3, 7, 4),
        ]
        gl.glBegin(gl.GL_QUADS)
        for a, b, c, d in faces:
            gl.glVertex3f(*vertices[a])
            gl.glVertex3f(*vertices[b])
            gl.glVertex3f(*vertices[c])
            gl.glVertex3f(*vertices[d])
        gl.glEnd()

    def draw_reservoirs(self) -> None:
        for idx, (name, res) in enumerate(self.state.reservoirs.items()):
            x, y = res.position_cm
            center = (x * self.world_scale, 1.5, y * self.world_scale)
            color = [c / 255 for c in hex_to_rgb(f"#{idx*4+100:02X}80{200-idx*20:02X}")]
            self.draw_box(center, (1.5, 3.0, 1.5), tuple(color))

    def draw_vial(self) -> None:
        center = (
            self.run.vial.position_cm[0] * self.world_scale,
            1.3,
            self.run.vial.position_cm[1] * self.world_scale,
        )
        fill_level = (self.state.vial_volume / self.state.vial_capacity) if self.state.vial_capacity else 0.0
        fill_height = max(0.1, 2.2 * fill_level)
        self.draw_box(center, (1.4, 2.6, 1.4), (0.4, 0.42, 0.45))
        gl.glColor3f(*[c / 255 for c in hex_to_rgb(self.state.vial_color_hex)])
        self.draw_box((center[0], center[1] - 1.2 + fill_height / 2, center[2]), (1.0, fill_height, 1.0), (0.6, 0.2, 0.3))

    def draw_pipette(self) -> None:
        x, y, z = self.state.pipette_pos
        gl.glColor3f(0.85, 0.9, 0.95)
        self.draw_box((x, z + 0.2, y), (0.4, 0.4, 2.5), (0.85, 0.9, 0.95))

    def draw_overlay(self) -> None:
        current_event = (
            self.state.events[self.state.current_event_index]
            if self.state.current_event_index < len(self.state.events)
            else None
        )
        label_text = (
            f"Event {self.state.current_event_index + 1}/{len(self.state.events)} — {current_event.action if current_event else 'Done'}"
            if current_event
            else "Playback complete"
        )
        self.info_label.text = label_text
        gl.glDisable(gl.GL_DEPTH_TEST)
        self.batch.draw()
        gl.glEnable(gl.GL_DEPTH_TEST)
        if self.target_hex:
            gl.glDisable(gl.GL_DEPTH_TEST)
            self.draw_target_swatch()
            gl.glEnable(gl.GL_DEPTH_TEST)

    def draw_target_swatch(self) -> None:
        color = [c / 255 for c in hex_to_rgb(self.target_hex)]
        gl.glBegin(gl.GL_QUADS)
        gl.glColor3f(*color)
        gl.glVertex3f(-4, 5, -4)
        gl.glVertex3f(-1, 5, -4)
        gl.glVertex3f(-1, 7, -4)
        gl.glVertex3f(-4, 7, -4)
        gl.glEnd()

    # -------------------------------------------------------------- interaction
    def on_key_press(self, symbol: int, modifiers: int) -> None:  # pragma: no cover
        if symbol == pyglet.window.key.LEFT:
            self.camera_yaw -= 5
        elif symbol == pyglet.window.key.RIGHT:
            self.camera_yaw += 5
        elif symbol == pyglet.window.key.UP:
            self.camera_pitch = clamp(self.camera_pitch + 3, -5, 80)
        elif symbol == pyglet.window.key.DOWN:
            self.camera_pitch = clamp(self.camera_pitch - 3, -5, 80)
        elif symbol in (pyglet.window.key.PLUS, pyglet.window.key.EQUAL):
            self.camera_distance = max(10.0, self.camera_distance - 1.0)
        elif symbol == pyglet.window.key.MINUS:
            self.camera_distance = min(40.0, self.camera_distance + 1.0)
        elif symbol == pyglet.window.key.SPACE:
            self.paused = not self.paused
        elif symbol == pyglet.window.key.ESCAPE:
            pyglet.app.exit()

    # ------------------------------------------------------------------ update
    def update(self, dt: float) -> None:
        if self.paused or self.state.done:
            return
        self.dt_accumulator += dt
        self.advance_events(self.dt_accumulator)
        self.dt_accumulator = 0.0

    def advance_events(self, dt: float) -> None:
        if self.state.current_event_index >= len(self.state.events):
            self.state.done = True
            return

        event = self.state.events[self.state.current_event_index]
        duration = event.duration_s if event.duration_s > 0 else 0.4
        self.state.event_progress += dt
        t = min(1.0, self.state.event_progress / duration) if duration > 0 else 1.0
        self.interpolate_pipette(event, t)

        if self.state.event_progress >= duration - 1e-6:
            self.apply_event_effects(event)
            self.state.event_progress = 0.0
            self.state.current_event_index += 1
            if self.state.current_event_index >= len(self.state.events):
                self.state.done = True

    def interpolate_pipette(self, event: Event, t: float) -> None:
        base_height = 6.0
        active_height = 2.5
        if event.start_pos and event.end_pos:
            sx, sy = event.start_pos
            ex, ey = event.end_pos
            px = lerp(sx, ex, t) * self.world_scale
            py = lerp(sy, ey, t) * self.world_scale
        elif event.end_pos:
            px = event.end_pos[0] * self.world_scale
            py = event.end_pos[1] * self.world_scale
        else:
            px, py = self.state.pipette_pos[0], self.state.pipette_pos[2]

        if event.action in {"ASPIRATE", "DISPENSE"}:
            z = base_height - (base_height - active_height) * math.sin(math.pi * min(t, 1.0))
        else:
            z = base_height

        self.state.pipette_pos = (px, py, z)

    def apply_event_effects(self, event: Event) -> None:
        if event.action == "ASPIRATE":
            res = self.state.reservoirs.get(event.target)
            if res:
                res.volume_mL = clamp(res.volume_mL - event.volume_mL, 0.0, res.initial_volume_mL)
        elif event.action == "DISPENSE":
            self.state.vial_volume = clamp(
                self.state.vial_volume + event.volume_mL, 0.0, self.state.vial_capacity
            )
            if event.color_hex:
                self.state.vial_color_hex = event.color_hex


def playback_3d(run: RunResult, target_hex: str | None = None) -> None:
    viewer = Twin3DViewer(run, target_hex=target_hex)
    pyglet.app.run()


def clamp(value: float, minimum: float = 0.0, maximum: float = 1.0) -> float:
    return max(minimum, min(value, maximum))


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="3D viewer for the dye-mixing digital twin.")
    parser.add_argument(
        "--seed",
        type=int,
        default=7,
        help="Random seed for the demo recipe.",
    )
    parser.add_argument(
        "--target",
        type=str,
        default=None,
        help="Optional target color hex to show in the scene (#RRGGBB).",
    )
    parser.add_argument(
        "--mode",
        choices=("demo",),
        default="demo",
        help="Currently only demo mode is supported.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    twin = DigitalTwin()
    run = twin.run(recipe=demo_recipe(), seed=args.seed)
    playback_3d(run, target_hex=args.target)


if __name__ == "__main__":
    main()
