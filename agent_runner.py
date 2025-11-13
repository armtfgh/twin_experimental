"""
LLM-guided recipe search for the dye-mixing digital twin.

This module drives the `DigitalTwin` defined in `main.py` using an OpenAI model.
Given a target color and a budget of trials, the agent repeatedly proposes
recipes, observes the simulated outcome, and iterates until it finds a mixture
that meets the desired color tolerance or runs out of attempts.

Usage:
    export OPENAI_API_KEY=sk-...
    python agent_runner.py --target "#C08090" --total-volume 1.0 --max-trials 10

Requirements:
    pip install openai
    (Optional) adjust MODEL_NAME for different OpenAI models.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import textwrap
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple
import httpx

from openai import OpenAI

from main import DigitalTwin, Reservoir, visualize_with_pygame

Vec3 = Tuple[float, float, float]


def clamp(value: float, minimum: float = 0.0, maximum: float = 1.0) -> float:
    return max(minimum, min(value, maximum))


def parse_hex_color(hex_value: str) -> Tuple[int, int, int]:
    stripped = hex_value.strip().lstrip("#")
    if len(stripped) != 6:
        raise ValueError(f"Invalid hex color '{hex_value}'. Expected #RRGGBB.")
    return tuple(int(stripped[i : i + 2], 16) for i in (0, 2, 4))


def srgb_distance(rgb_a: Tuple[int, int, int], rgb_b: Tuple[int, int, int]) -> float:
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(rgb_a, rgb_b)))


def describe_reservoirs(reservoirs: Dict[str, Reservoir]) -> str:
    lines = [
        f"- {name}: absorption={res.absorption}, capacity≈{res.volume_mL:.1f} mL"
        for name, res in reservoirs.items()
    ]
    return "\n".join(lines)


def summarize_history(history: Sequence["TrialRecord"]) -> str:
    if not history:
        return "None yet."
    lines: List[str] = []
    for record in history:
        recipe_str = ", ".join(f"{step['source']} {step['volume_mL']:.2f} mL" for step in record.recipe)
        lines.append(
            f"Trial {record.trial_index}: {recipe_str} => {record.color_hex} "
            f"(distance {record.distance:.2f})"
        )
        lines.append(f"    Agent rationale: {record.rationale}")
    return "\n".join(lines)


def summarize_rationales(history: Sequence["TrialRecord"]) -> str:
    if not history:
        return "None yet."
    lines = []
    for record in history:
        rationale = record.rationale.strip() or "<empty>"
        lines.append(f"Trial {record.trial_index}: {rationale}")
    return "\n".join(lines)


def extract_json_block(text: str) -> str:
    """Return the first JSON object found inside text."""
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("No JSON object found in LLM response.")
    return text[start : end + 1]


@dataclass
class TrialRecord:
    trial_index: int
    recipe: List[Dict[str, float]]
    color_hex: str
    color_rgb: Tuple[int, int, int]
    distance: float
    rationale: str


class LLMRecipeAgent:
    def __init__(
        self,
        target_hex: str,
        total_volume_mL: float,
        max_trials: int,
        tolerance: float,
        model_name: str,
        temperature: float,
        visualize: bool,
    ) -> None:
        self.target_hex = target_hex.upper()
        self.target_rgb = parse_hex_color(target_hex)
        self.total_volume_mL = total_volume_mL
        self.max_trials = max_trials
        self.tolerance = tolerance
        self.client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"),http_client=httpx.Client(verify=False))
        self.model_name = model_name
        self.temperature = temperature
        self.twin = DigitalTwin()
        self.show_visualization = visualize
        self.base_reservoirs = self.twin.new_reservoirs()

    def run(self) -> List[TrialRecord]:
        history: List[TrialRecord] = []
        for trial in range(1, self.max_trials + 1):
            proposal = self._request_recipe(history, trial)
            recipe_steps = self._sanitize_recipe(proposal["recipe"])
            recipe_for_twin = [(step["source"], step["volume_mL"]) for step in recipe_steps]

            run = self.twin.run(recipe=recipe_for_twin, seed=42 + trial)
            reading = run.vial.color()
            distance = srgb_distance(reading.srgb8, self.target_rgb)

            record = TrialRecord(
                trial_index=trial,
                recipe=recipe_steps,
                color_hex=reading.hex_value,
                color_rgb=reading.srgb8,
                distance=distance,
                rationale=proposal.get("rationale", ""),
            )
            history.append(record)
            self._print_trial_summary(record)
            if self.show_visualization:
                self._visualize_run(run, trial, distance)

            if distance <= self.tolerance:
                print(f"[success] Target reached within tolerance after {trial} trials.")
                break
        else:
            print("[info] Trial budget exhausted without reaching target tolerance.")

        return history

    def _print_trial_summary(self, record: TrialRecord) -> None:
        recipe_str = ", ".join(f"{step['source']} {step['volume_mL']:.2f} mL" for step in record.recipe)
        print(
            textwrap.dedent(
                f"""
                Trial {record.trial_index}:
                    Recipe: {recipe_str}
                    Outcome color: {record.color_hex} (sRGB {record.color_rgb})
                    Distance to target: {record.distance:.2f}
                    Agent rationale: {record.rationale}
                """
            ).strip()
        )

    def _request_recipe(self, history: List[TrialRecord], trial_index: int) -> Dict:
        system_prompt = textwrap.dedent(
            """
            You are an autonomous lab planner for dye mixing. Available sources are dyes with
            absorption vectors and water. You must design a pipetting recipe (sequence of
            aspirate/dispense steps) whose total dispensed volume equals the requested target volume.

            Respond ONLY with JSON of the form:
            {
              "rationale": "...",
              "recipe": [
                {"source": "Crimson", "volume_mL": 0.25},
                ...
              ]
            }
            Constraints:
            - Sources must be chosen from the provided list.
            - Sum of volume_mL must equal the specified total volume (tolerance 0.01 mL).
            - No step may have volume <= 0.
            - Prefer at most 6 steps.
            """
        ).strip()

        history_summary = summarize_history(history)
        rationale_summary = summarize_rationales(history)
        reservoirs_description = describe_reservoirs(self.base_reservoirs)

        user_prompt = textwrap.dedent(
            f"""
            Target color (sRGB): {self.target_hex} / {self.target_rgb}
            Desired total volume: {self.total_volume_mL:.2f} mL
            Tolerance goal: color distance ≤ {self.tolerance:.2f}
            Available reservoirs:
            {reservoirs_description}

            Previous trials:
            {history_summary}

            Previous rationales:
            {rationale_summary}

            Provide the next recipe for trial #{trial_index}.
            """
        ).strip()

        response = self.client.chat.completions.create(
            model=self.model_name,
            temperature=self.temperature,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        content = response.choices[0].message.content or ""
        try:
            data = json.loads(extract_json_block(content))
        except ValueError as exc:
            raise RuntimeError(f"Could not parse agent response:\n{content}") from exc

        if "recipe" not in data or not isinstance(data["recipe"], list):
            raise RuntimeError(f"Agent response missing 'recipe' list:\n{data}")
        return data

    def _sanitize_recipe(self, recipe: List[Dict]) -> List[Dict[str, float]]:
        cleaned: List[Dict[str, float]] = []
        total = 0.0
        valid_sources = {name.lower(): name for name in self.base_reservoirs.keys()}
        valid_sources["vial"] = "Vial"  # allow referencing target by mistake; will be filtered

        for step in recipe:
            source = step.get("source")
            volume = step.get("volume_mL")
            if not isinstance(source, str) or not isinstance(volume, (int, float)):
                continue
            source_lookup = source.lower()
            if source_lookup == "vial":
                continue
            if source_lookup not in valid_sources:
                continue
            volume = max(0.0, float(volume))
            if volume <= 0:
                continue
            cleaned.append({"source": valid_sources[source_lookup], "volume_mL": volume})
            total += volume

        if not cleaned:
            raise RuntimeError("Agent produced empty/invalid recipe.")

        if total <= 0:
            raise RuntimeError("Recipe total volume is zero.")

        scale = self.total_volume_mL / total
        scaled = [
            {"source": step["source"], "volume_mL": step["volume_mL"] * scale}
            for step in cleaned
        ]
        return scaled

    def _visualize_run(self, run_result, trial_index: int, distance: float) -> None:
        print(
            f"\n[info] Trial {trial_index} visual playback. "
            "Press Enter inside the window to proceed when done."
        )
        visualize_with_pygame(
            run_result.events,
            run_result.initial_reservoirs,
            run_result.vial.capacity_mL,
            run_result.pipette_start_pos,
            target_hex=self.target_hex,
            wait_for_continue=True,
        )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="LLM-driven recipe planner for the dye digital twin.")
    parser.add_argument("--target", type=str, default="#CFB5B7", help="Target color hex (#RRGGBB).")
    parser.add_argument("--total-volume", type=float, default=1.0, help="Total mixture volume in mL.")
    parser.add_argument("--max-trials", type=int, default=10, help="Trial budget for the agent.")
    parser.add_argument("--tolerance", type=float, default=15.0, help="sRGB distance tolerance for success.")
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o-mini",
        help="OpenAI chat model to use.",
    )
    parser.add_argument("--temperature", type=float, default=0.4, help="Sampling temperature.")
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="After each trial, replay the run in PyGame (close window to continue).",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY environment variable is not set.")

    agent = LLMRecipeAgent(
        target_hex=args.target,
        total_volume_mL=args.total_volume,
        max_trials=args.max_trials,
        tolerance=args.tolerance,
        model_name=args.model,
        temperature=args.temperature,
        visualize=args.visualize,
    )
    history = agent.run()

    print("\n=== Final Summary ===")
    for record in history:
        recipe_str = ", ".join(f"{step['source']} {step['volume_mL']:.2f} mL" for step in record.recipe)
        print(
            f"Trial {record.trial_index}: {recipe_str} -> {record.color_hex} "
            f"(distance {record.distance:.2f})"
        )


if __name__ == "__main__":
    main()
