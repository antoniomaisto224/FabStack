"""Local GGUF-backed process-step planner for FabStack."""

import json
import os
import re
from collections import Counter
from typing import Dict, Iterable, List, Tuple

from fabstack.assistant.normalization import (
    _coerce_bool,
    _coerce_float,
    _measure_was_requested,
    _normalize_user_request,
    _unique,
    _user_facing_detail,
)
from fabstack.assistant.measurements import (
    _NUMBER_PATTERN,
    _UNIT_PATTERN,
    _extract_all_measure_um,
    _extract_measure,
    _extract_measure_mentions_um,
    _extract_measure_um,
    _extract_substrate_like_unitless_um,
    _extract_unitless_measure_um,
    _first_explicit_unit,
    _last_explicit_unit,
    _number_span_is_identifier,
    _unit_value_to_um,
)
from fabstack.assistant.prompts import (
    COMMAND_SYSTEM_PROMPT,
    MODEL_FILENAME,
    SYSTEM_PROMPT,
)
from fabstack.operations.contracts import OPERATION_KEYS
from fabstack.resource_paths import resource_path
from fabstack.terminology import operation_label


ALLOWED_OPERATIONS = OPERATION_KEYS


class LocalStepPlanner:
    """Lazy loader for a local GGUF model with a deterministic fallback."""

    def __init__(
        self,
        model_path: str = None,
        *,
        n_ctx: int = 6144,
        max_tokens: int = 1000,
        temperature: float = 0.05,
        top_p: float = 0.85,
    ):
        self.model_path = model_path or resource_path(MODEL_FILENAME)
        self.n_ctx = int(n_ctx)
        self.max_tokens = int(max_tokens)
        self.temperature = float(temperature)
        self.top_p = float(top_p)
        self._model = None
        self._load_error = ""
        self._last_backend = "not loaded"

    def model_exists(self) -> bool:
        return os.path.exists(self.model_path)

    def runtime_available(self) -> bool:
        try:
            import llama_cpp  # noqa: F401
        except Exception:
            return False
        return True

    def status_text(self) -> str:
        """Return task-level UI status without exposing backend diagnostics."""
        return "Ready"

    def generate_step_plan(self, user_request: str, context: Dict = None) -> str:
        user_request = str(user_request or "").strip()
        if not user_request:
            return "Before I continue:\n1. Describe the fabrication process you want to build."

        context = context if isinstance(context, dict) else {}
        try:
            return self._generate_with_model(user_request, context)
        except Exception as exc:
            self._load_error = str(exc)
            self._last_backend = "fallback"
            return self._fallback_plan(user_request, context)

    def generate_command_plan(self, user_request: str, context: Dict = None) -> Dict:
        user_request = _normalize_user_request(user_request)
        context = context if isinstance(context, dict) else {}
        if not user_request:
            return {
                "status": "needs_clarification",
                "message": "More details needed.",
                "missing": ["Describe the fabrication process you want to build."],
                "commands": [],
            }

        resolved_request = _resolve_conversation_request(user_request, context)
        unresolved_fragment = (
            resolved_request == user_request
            and _request_is_clarification_fragment(user_request, context)
            and not _request_contains_operation_intent(user_request, context)
        )

        model_plan = {}
        model_error = None
        try:
            raw = self._generate_command_json_with_model(resolved_request, context)
            model_plan = self._parse_command_json(raw)
            model_plan = self._sanitize_command_plan(model_plan, context)
            model_plan = self._repair_command_details_from_request(model_plan, resolved_request, context)
            model_plan = self._verify_command_numbers_against_request(model_plan, resolved_request, context)
            model_plan = self._order_plan_commands_for_context(model_plan, context)
            model_plan = self._verify_plan_covers_request(model_plan, resolved_request, context)
        except Exception as exc:
            self._load_error = str(exc)
            self._last_backend = "fallback"
            model_error = exc

        if unresolved_fragment and not self._assistant_plan_is_usable(model_plan):
            return {
                "status": "needs_clarification",
                "message": "More details needed.",
                "missing": ["Tell me which operation and material these details belong to."],
                "commands": [],
            }

        repair_plan = self._fallback_command_plan(resolved_request, context)
        if _request_is_clarification_fragment(user_request, context) and not self._assistant_plan_is_usable(repair_plan):
            recent_plan = self._plan_from_recent_detailed_request(user_request, context)
            if self._assistant_plan_is_usable(recent_plan):
                repair_plan = recent_plan

        chosen = self._choose_command_plan(model_plan, repair_plan)
        if chosen:
            return self._repair_command_details_from_request(chosen, resolved_request, context)
        if model_error is not None:
            return self._repair_command_details_from_request(repair_plan, resolved_request, context)
        return self._repair_command_details_from_request(model_plan or repair_plan, resolved_request, context)

    def _assistant_plan_is_usable(self, plan: Dict) -> bool:
        return (
            isinstance(plan, dict)
            and plan.get("status") == "ready"
            and self._has_operation_command(plan)
        )

    def _choose_command_plan(self, model_plan: Dict, repair_plan: Dict) -> Dict:
        model_ok = self._assistant_plan_is_usable(model_plan)
        repair_ok = self._assistant_plan_is_usable(repair_plan)
        if model_ok and not repair_ok:
            return model_plan
        if repair_ok and not model_ok:
            return repair_plan
        if model_ok and repair_ok:
            if _model_adds_unrequested_physical_layers(model_plan, repair_plan):
                return repair_plan
            return repair_plan if _plan_score(repair_plan) >= _plan_score(model_plan) else model_plan
        if isinstance(repair_plan, dict) and repair_plan.get("missing"):
            return repair_plan
        if isinstance(model_plan, dict) and model_plan.get("missing"):
            return model_plan
        return repair_plan if isinstance(repair_plan, dict) else model_plan

    def _plan_from_recent_detailed_request(self, user_request: str, context: Dict) -> Dict:
        if not _request_is_sparse_followup(user_request, context):
            return {}

        recent_requests = context.get("recent_user_requests", []) if isinstance(context, dict) else []
        if not isinstance(recent_requests, list):
            return {}

        active = [str(x) for x in context.get("active_materials", []) if str(x).strip()]
        library = [str(x) for x in context.get("library_materials", []) if str(x).strip()]
        known = _unique(active + library)
        current = _normalize_user_request(user_request).lower()
        for previous in reversed(recent_requests):
            previous = _normalize_user_request(previous)
            if not previous or previous.lower() == current:
                continue
            if not _request_has_material_and_geometry(previous, known):
                continue
            plan = self._fallback_command_plan(previous, context)
            if plan.get("status") == "ready" and self._has_operation_command(plan):
                return plan
        return {}

    def _has_operation_command(self, plan: Dict) -> bool:
        commands = plan.get("commands", []) if isinstance(plan, dict) else []
        return any(
            isinstance(cmd, dict) and cmd.get("action") in {"operation", "set_substrate", "add_material"}
            for cmd in commands
        )

    def _load_model(self):
        if self._model is not None:
            return self._model
        if not self.model_exists():
            raise RuntimeError(f"missing model file: {os.path.basename(self.model_path)}")

        # Metal can fail in headless/offscreen macOS contexts. Use CPU by
        # default for reliability; advanced users can override this env var.
        os.environ.setdefault("GGML_METAL_DEVICES", "-1")

        try:
            from llama_cpp import Llama
        except Exception as exc:
            raise RuntimeError("install llama-cpp-python to enable the local GGUF model") from exc

        thread_count = max(2, min(8, (os.cpu_count() or 4) - 1))
        kwargs = {
            "model_path": self.model_path,
            "n_ctx": self.n_ctx,
            "n_threads": thread_count,
            "n_batch": 512,
            "verbose": False,
        }
        self._model = Llama(**kwargs)
        self._load_error = ""
        return self._model

    def _mistral_prompt(self, system_prompt: str, user_prompt: str) -> str:
        return f"<s>[INST] {system_prompt.strip()}\n\n{user_prompt.strip()} [/INST]"

    def _generate_with_model(self, user_request: str, context: Dict) -> str:
        model = self._load_model()
        prompt = self._compose_user_prompt(user_request, context)

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]

        if hasattr(model, "create_chat_completion"):
            response = model.create_chat_completion(
                messages=messages,
                temperature=self.temperature,
                top_p=self.top_p,
                max_tokens=self.max_tokens,
                stop=["</s>", "<|endoftext|>"],
            )
            text = (
                response.get("choices", [{}])[0]
                .get("message", {})
                .get("content", "")
            )
        else:
            raw_prompt = self._mistral_prompt(SYSTEM_PROMPT, prompt)
            response = model(
                raw_prompt,
                temperature=self.temperature,
                top_p=self.top_p,
                max_tokens=self.max_tokens,
                stop=["</s>", "<|endoftext|>"],
            )
            text = response.get("choices", [{}])[0].get("text", "")

        self._last_backend = "local"
        return self._clean_model_output(text)

    def _generate_command_json_with_model(self, user_request: str, context: Dict) -> str:
        model = self._load_model()
        prompt = self._compose_command_prompt(user_request, context)
        messages = [
            {"role": "system", "content": COMMAND_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]

        if hasattr(model, "create_chat_completion"):
            response = model.create_chat_completion(
                messages=messages,
                temperature=0.0,
                top_p=0.9,
                max_tokens=max(self.max_tokens, 1100),
                stop=["</s>", "<|endoftext|>"],
            )
            text = (
                response.get("choices", [{}])[0]
                .get("message", {})
                .get("content", "")
            )
        else:
            raw_prompt = self._mistral_prompt(COMMAND_SYSTEM_PROMPT, prompt)
            response = model(
                raw_prompt,
                temperature=0.0,
                top_p=0.9,
                max_tokens=max(self.max_tokens, 1100),
                stop=["</s>", "<|endoftext|>"],
            )
            text = response.get("choices", [{}])[0].get("text", "")

        self._last_backend = "local"
        return self._clean_model_output(text)

    def _compose_user_prompt(self, user_request: str, context: Dict) -> str:
        lines = [
            "Translate this request into FabStack process steps.",
            "",
            "Current FabStack context:",
            f"- Substrate: {context.get('substrate', 'unknown')}",
            f"- Active materials: {', '.join(context.get('active_materials', [])) or 'none'}",
            f"- Allowed operations: {', '.join(ALLOWED_OPERATIONS)}",
        ]

        history = context.get("process_flow", [])
        if history:
            lines.append("- Existing process flow:")
            for i, title in enumerate(history[-12:], 1):
                lines.append(f"  {i}. {title}")

        restrictions = context.get("operation_restrictions", [])
        if restrictions:
            lines.append("- Restricted operation/material pairs:")
            for note in restrictions[:16]:
                lines.append(f"  - {note}")

        library = context.get("library_summary", [])
        if library:
            lines.append("- Material library summary:")
            for note in library[:20]:
                lines.append(f"  - {note}")

        lines.extend(["", "User request:", user_request])
        return "\n".join(lines)

    def _compose_command_prompt(self, user_request: str, context: Dict) -> str:
        active = context.get("active_materials", [])
        library = context.get("library_materials", [])
        photoresists = context.get("photoresist_materials", [])
        lines = [
            "Compile this request into executable FabStack JSON commands.",
            "",
            "Current FabStack context:",
            f"- Substrate: {context.get('substrate', 'unknown')}",
            f"- Active materials: {', '.join(active) or 'none'}",
            f"- Allowed operations: {', '.join(ALLOWED_OPERATIONS)}",
            f"- Photoresist materials: {', '.join(photoresists) or 'none'}",
            "",
            "Canonical material library names:",
        ]
        for material in library[:160]:
            lines.append(f"- {material}")

        history = context.get("process_flow", [])
        if history:
            lines.append("")
            lines.append("Existing process flow:")
            for i, title in enumerate(history[-12:], 1):
                lines.append(f"{i}. {title}")

        thicknesses = context.get("material_thickness_um", {})
        if isinstance(thicknesses, dict) and thicknesses:
            lines.append("")
            lines.append("Current material thickness estimates:")
            for material, value in list(thicknesses.items())[:40]:
                try:
                    lines.append(f"- {material}: {float(value):g} µm")
                except (TypeError, ValueError):
                    continue

        order = context.get("top_material_order", [])
        if isinstance(order, list) and order:
            lines.append("")
            lines.append(f"Current stack order, top to bottom: {', '.join(str(x) for x in order[:40])}")

        recent_operation = str(context.get("recent_operation", "") or "").strip()
        if recent_operation:
            lines.append(f"Recent requested operation: {recent_operation}")

        restrictions = context.get("operation_restrictions", [])
        if restrictions:
            lines.append("")
            lines.append("Restricted operation/material pairs:")
            for note in restrictions[:20]:
                lines.append(f"- {note}")

        lines.extend(["", "User request:", user_request])
        return "\n".join(lines)

    def _clean_model_output(self, text: str) -> str:
        text = str(text or "").strip()
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.IGNORECASE | re.DOTALL)
        text = re.sub(r"</?think>", "", text, flags=re.IGNORECASE)
        text = re.sub(r"<\|[^>]+?\|>", "", text)
        text = text.strip()
        if not text:
            return "Before I continue:\n1. I couldn’t create a plan from that request. Add the process steps and try again."
        return text

    def _parse_command_json(self, text: str) -> Dict:
        text = str(text or "").strip()
        try:
            return json.loads(text)
        except Exception:
            pass

        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            return json.loads(text[start:end + 1])
        raise ValueError("No JSON command object found")

    def _sanitize_command_plan(self, plan: Dict, context: Dict) -> Dict:
        if not isinstance(plan, dict):
            raise ValueError("Invalid command plan")

        active = [str(x) for x in context.get("active_materials", []) if str(x).strip()]
        library = [str(x) for x in context.get("library_materials", []) if str(x).strip()]
        photoresists = [str(x) for x in context.get("photoresist_materials", []) if str(x).strip()]
        known = _unique(active + library)
        missing = [
            _user_facing_detail(x)
            for x in plan.get("missing", [])
            if str(x).strip()
        ]
        raw_commands = plan.get("commands", [])
        commands = []
        used_materials = []

        if not isinstance(raw_commands, list):
            raw_commands = []

        for raw in raw_commands:
            if not isinstance(raw, dict):
                continue
            action = str(raw.get("action", "")).strip()

            if action == "add_material":
                material = _canonical_material(str(raw.get("material", "")), known)
                if not material:
                    missing.append("Which material should I add?")
                    continue
                commands.append({"action": "add_material", "material": material})
                continue

            if action == "set_substrate":
                material = _canonical_material(str(raw.get("material", "")), known)
                if not material:
                    missing.append("Which substrate material should I use?")
                    continue
                cmd = {"action": "set_substrate", "material": material}
                thickness = _coerce_float(raw.get("thickness_um"))
                if thickness is not None:
                    cmd["thickness_um"] = thickness
                commands.append(cmd)
                continue

            if action != "operation":
                continue

            operation = _canonical_operation(str(raw.get("operation", "")))
            if not operation:
                missing.append("Which supported operation should I apply?")
                continue

            cmd = {"action": "operation", "operation": operation}
            material = str(raw.get("material", "") or "").strip()
            if operation != "Flip":
                material = _canonical_material(material, known)
                if not material:
                    missing.append(
                        f"Which material should I use for the {operation_label(operation).lower()} step?"
                    )
                    continue
                if operation == "Deposit" and material in photoresists:
                    operation = "Spincoating"
                    cmd["operation"] = operation
                cmd["material"] = material
                used_materials.append(material)
            else:
                cmd["material"] = ""

            if operation in {"Deposit", "Spincoating", "Bond"}:
                thickness = _coerce_float(raw.get("thickness_um"))
                if thickness is None or thickness <= 0:
                    missing.append(
                        f"What thickness should I use for the {operation_label(operation).lower()} step? "
                        "Include a unit."
                    )
                    continue
                cmd["thickness_um"] = thickness

            if operation == "Etch":
                depth = _coerce_float(raw.get("depth_um"))
                if depth is None or depth <= 0:
                    missing.append("What etch depth should I use? Include a unit.")
                    continue
                cmd["depth_um"] = depth
                mode = str(raw.get("etch_mode", "") or "").strip().lower()
                cmd["etch_mode"] = mode if mode in {"anisotropic", "isotropic"} else "anisotropic"

            if operation == "Lithography":
                if photoresists and cmd.get("material") not in photoresists:
                    missing.append("Which photoresist should I use for lithography?")
                    continue
                manual_windowing = (
                    str(raw.get("windowing_mode", "") or "").strip().lower() == "manual"
                    or _extract_lithography_manual_windowing(str(raw.get("method", "") or ""))
                )
                windows = _sanitize_windows(raw.get("windows"))
                if not windows and not manual_windowing:
                    missing.append("Which lithography windows should I use? Example: 0.35–0.65.")
                    continue
                cmd["windows"] = windows
                if manual_windowing:
                    cmd["windowing_mode"] = "manual"
                invert = _coerce_bool(raw.get("invert_lithography"))
                if invert is None:
                    invert = _extract_lithography_invert(str(raw.get("method", "") or ""))
                cmd["invert_lithography"] = bool(invert) if invert is not None else False

            method = str(raw.get("method", "") or "").strip()
            if method:
                cmd["method"] = method
            temperature = _coerce_float(raw.get("temperature_c"))
            if temperature is not None:
                cmd["temperature_c"] = temperature
            time_s = _coerce_float(raw.get("time_s"))
            if time_s is not None:
                cmd["time_s"] = time_s

            commands.append(cmd)

        active_set = {x.lower(): x for x in active}
        add_materials = []
        for material in used_materials:
            if material.lower() in active_set:
                continue
            if material not in add_materials:
                add_materials.append(material)

        existing_adds = {
            str(cmd.get("material", "")).lower()
            for cmd in commands
            if cmd.get("action") == "add_material"
        }
        prefix = [
            {"action": "add_material", "material": material}
            for material in add_materials
            if material.lower() not in existing_adds
        ]

        commands = prefix + commands
        missing = _unique(missing)
        status = "needs_clarification" if missing else "ready"
        if status != "ready":
            commands = []

        message = "Process plan ready." if status == "ready" else "More details needed."

        commands = self._order_commands_for_context(commands, context)

        return {
            "status": status,
            "message": message,
            "missing": missing,
            "commands": commands,
        }

    def _fallback_command_plan(self, user_request: str, context: Dict) -> Dict:
        user_request = _normalize_user_request(user_request)
        active = [str(x) for x in context.get("active_materials", []) if str(x).strip()]
        library = [str(x) for x in context.get("library_materials", []) if str(x).strip()]
        photoresists = [str(x) for x in context.get("photoresist_materials", []) if str(x).strip()]
        known = _unique(active + library)
        segments = _extract_command_segments(user_request, context, known)
        commands = []
        missing = []
        used_materials = []
        local_context = _copy_context_for_request(context)

        for operation_hint, clause in segments:
            low = clause.lower()

            if operation_hint == "Set substrate":
                material = _find_material(clause, known) or _default_substrate_material(clause, known)
                if not material:
                    missing.append("Which substrate material should I use?")
                    continue
                command = {"action": "set_substrate", "material": material}
                thickness = _extract_measure_um(clause)
                if thickness is None:
                    thickness = _extract_substrate_like_unitless_um(clause)
                if thickness is not None and thickness > 0.0:
                    command["thickness_um"] = float(thickness)
                commands.append(command)
                used_materials.append(material)
                local_context = _record_substrate_set(local_context, material, command.get("thickness_um"))
                continue

            if operation_hint == "Etch":
                etch_commands, etch_used, etch_missing = _parse_etch_commands(clause, local_context, known)
                if etch_commands:
                    commands.extend(etch_commands)
                    used_materials.extend(etch_used)
                    for command in etch_commands:
                        local_context = _record_layer_etch(
                            local_context,
                            command.get("material", ""),
                            command.get("depth_um", 0.0),
                        )
                missing.extend(etch_missing)
                continue

            if operation_hint in {"Deposit", "Spincoating", "Bond"}:
                layer_commands, layer_used, layer_missing = _parse_layer_commands(
                    clause,
                    operation_hint,
                    local_context,
                    known,
                    photoresists,
                )
                commands.extend(layer_commands)
                used_materials.extend(layer_used)
                missing.extend(layer_missing)
                for command in layer_commands:
                    local_context = _record_layer_addition(
                        local_context,
                        command.get("material", ""),
                        command.get("thickness_um", 0.0),
                        command.get("operation", ""),
                    )
                continue

            if operation_hint == "Strip material":
                materials = _materials_for_segment(clause, known)
                if _segment_mentions_generic_photoresist(clause):
                    preferred = _latest_used_photoresist(used_materials, photoresists) or _default_photoresist_material(local_context, known, photoresists)
                    if preferred:
                        materials = [preferred]
                if not materials and _segment_mentions_previous_layer(clause):
                    preferred = (
                        _latest_used_photoresist(used_materials, photoresists)
                        or (used_materials[-1] if used_materials else "")
                    )
                    if preferred:
                        materials = [preferred]
                if not materials:
                    missing.append("Which material should I strip?")
                    continue
                for material in materials:
                    command = {"action": "operation", "operation": "Strip material", "material": material}
                    commands.append(command)
                    used_materials.append(material)
                    local_context = _record_layer_strip(local_context, material)
                continue

            if operation_hint == "Lift-off":
                material = (
                    _find_material(clause, known)
                    or _latest_used_photoresist(used_materials, photoresists)
                    or _default_photoresist_material(local_context, known, photoresists)
                )
                if not material:
                    missing.append("Which photoresist should I use for lift-off?")
                    continue
                command = {"action": "operation", "operation": "Lift-off", "material": material}
                commands.append(command)
                used_materials.append(material)
                local_context = _record_layer_strip(local_context, material)
                continue

            if operation_hint == "Clean / Surface prep":
                target = _find_material(clause, known) or (active[0] if active else "")
                if not target:
                    missing.append("Which material or wafer should I clean?")
                    continue
                command = {"action": "operation", "operation": "Clean / Surface prep", "material": target}
                commands.append(command)
                used_materials.append(target)
                continue

            if operation_hint == "Bake/Annealing":
                materials = _materials_for_segment(clause, known)
                if not materials:
                    missing.append("Which material should I bake or anneal?")
                    continue
                for material in materials:
                    command = {"action": "operation", "operation": "Bake/Annealing", "material": material}
                    commands.append(command)
                    used_materials.append(material)
                continue

            if operation_hint == "Lithography":
                material = (
                    _find_material(clause, known)
                    or _latest_used_photoresist(used_materials, photoresists)
                    or _default_photoresist_material(local_context, known, photoresists)
                )
                if not material:
                    missing.append("Which photoresist should I use for lithography?")
                    continue
                if photoresists and material not in photoresists:
                    missing.append("Which photoresist should I use for lithography?")
                    continue
                windows = _extract_lithography_windows(clause)
                manual_windowing = _extract_lithography_manual_windowing(clause)
                if not windows and not manual_windowing:
                    missing.append("Which lithography windows should I use? Example: 0.35–0.65.")
                    continue
                command = {
                    "action": "operation",
                    "operation": "Lithography",
                    "material": material,
                    "windows": windows,
                    "invert_lithography": bool(_extract_lithography_invert(clause) or False),
                }
                if manual_windowing:
                    command["windowing_mode"] = "manual"
                commands.append(command)
                used_materials.append(material)
                continue

            if operation_hint == "Flip":
                commands.append({"action": "operation", "operation": "Flip", "material": ""})
                continue

            missing.append(f"I couldn’t match this text to a supported operation: “{clause}”.")

        if missing or not commands:
            return {
                "status": "needs_clarification",
                "message": "More details needed.",
                "missing": _unique(missing or ["List the process steps in order."]),
                "commands": [],
            }

        active_lower = {x.lower() for x in active}
        prefix = []
        for material in used_materials:
            if material.lower() not in active_lower and not any(c.get("material") == material for c in prefix):
                prefix.append({"action": "add_material", "material": material})

        ordered_commands = prefix + commands
        return {
            "status": "ready",
            "message": "Process plan ready.",
            "missing": [],
            "commands": ordered_commands,
        }

    def _verify_command_numbers_against_request(self, plan: Dict, user_request: str, context: Dict = None) -> Dict:
        if not isinstance(plan, dict) or plan.get("status") != "ready":
            return plan

        context = context if isinstance(context, dict) else {}
        measures = _extract_all_measure_um(user_request)
        missing = list(plan.get("missing", []))
        for command in plan.get("commands", []):
            if not isinstance(command, dict) or command.get("action") != "operation":
                continue
            op = command.get("operation")
            if op in {"Deposit", "Spincoating", "Bond"}:
                value = _coerce_float(command.get("thickness_um"))
                if value is None or not _command_measure_was_requested(command, value, user_request, measures, context):
                    missing.append(
                        f"Give the thickness for the {operation_label(op).lower()} step, including a unit."
                    )
            elif op == "Etch":
                value = _coerce_float(command.get("depth_um"))
                if value is None or not _command_measure_was_requested(command, value, user_request, measures, context):
                    missing.append("Give the etch depth with a unit.")

        missing = _unique(missing)
        if missing:
            return {
                "status": "needs_clarification",
                "message": "More details needed.",
                "missing": missing,
                "commands": [],
            }
        return plan

    def _verify_plan_covers_request(self, plan: Dict, user_request: str, context: Dict = None) -> Dict:
        if not isinstance(plan, dict) or plan.get("status") != "ready":
            return plan

        context = context if isinstance(context, dict) else {}
        known = _unique(list(_iter_materials(context)))
        requirements = _request_command_requirements(user_request, context, known)
        if not requirements:
            return plan

        missing = []
        for requirement in requirements:
            if not _plan_satisfies_requirement(plan, requirement, known):
                missing.append(
                    f"I couldn’t include this requested step: “{_requirement_label(requirement)}”. "
                    "Rephrase that step."
                )

        missing = _unique(missing)
        if missing:
            return {
                "status": "needs_clarification",
                "message": "More details needed.",
                "missing": missing,
                "commands": [],
            }
        return plan

    def _order_plan_commands_for_context(self, plan: Dict, context: Dict) -> Dict:
        if isinstance(plan, dict) and isinstance(plan.get("commands"), list):
            plan = dict(plan)
            plan["commands"] = self._order_commands_for_context(plan.get("commands", []), context)
        return plan

    def _order_commands_for_context(self, commands: List[Dict], context: Dict) -> List[Dict]:
        return _order_commands_for_context(commands, context)

    def _repair_command_details_from_request(self, plan: Dict, user_request: str, context: Dict) -> Dict:
        if not isinstance(plan, dict) or not isinstance(plan.get("commands"), list):
            return plan

        known = _unique(list(_iter_materials(context if isinstance(context, dict) else {})))
        segments = _extract_command_segments(user_request, context if isinstance(context, dict) else {}, known)
        clauses_by_operation = {}
        for operation, clause in segments:
            clauses_by_operation.setdefault(operation, []).append(clause)
        cursors = {operation: 0 for operation in clauses_by_operation}

        repaired = dict(plan)
        repaired_commands = []
        for command in plan.get("commands", []):
            if not isinstance(command, dict):
                repaired_commands.append(command)
                continue

            updated = dict(command)
            if updated.get("action") == "operation":
                operation = str(updated.get("operation", "") or "")
                operation_clauses = clauses_by_operation.get(operation, [])
                cursor = cursors.get(operation, 0)
                clause = operation_clauses[cursor] if cursor < len(operation_clauses) else ""
                if cursor < len(operation_clauses):
                    cursors[operation] = cursor + 1

                if operation == "Etch":
                    mode = _extract_etch_mode(clause)
                    if mode:
                        updated["etch_mode"] = mode
                    if not str(updated.get("method", "") or "").strip():
                        etch_method = _extract_etch_method(clause)
                        if etch_method:
                            updated["method"] = etch_method

                if operation == "Lithography":
                    windows = _extract_lithography_windows(clause)
                    if windows:
                        updated["windows"] = windows
                    if _extract_lithography_manual_windowing(clause):
                        updated["windowing_mode"] = "manual"
                        updated.setdefault("windows", [])
                    invert = _extract_lithography_invert(clause)
                    updated["invert_lithography"] = bool(invert) if invert is not None else False

            repaired_commands.append(updated)

        repaired["commands"] = repaired_commands
        return repaired

    def _fallback_plan(self, user_request: str, context: Dict) -> str:
        materials = list(_iter_materials(context))
        clauses = [
            c.strip(" .")
            for c in re.split(r"\bthen\b|\bafter that\b|;|\n", user_request, flags=re.IGNORECASE)
            if c.strip(" .")
        ]
        steps = []
        clarifications = []

        for clause in clauses:
            step, needs = self._fallback_step_for_clause(clause, materials)
            if step:
                steps.append(step)
            clarifications.extend(needs)

        if not steps:
            return (
                "Before I continue:\n"
                "1. List the process steps in order.\n"
                "2. Include each material and required dimension."
            )

        output = []
        for idx, step in enumerate(steps, 1):
            output.append(f"{idx}. {step}")

        unique_clarifications = []
        for item in clarifications:
            if item not in unique_clarifications:
                unique_clarifications.append(item)
        if unique_clarifications:
            output.append("")
            output.append("Before I continue:")
            for idx, item in enumerate(unique_clarifications, 1):
                output.append(f"{idx}. {item}")
        return "\n".join(output)

    def _fallback_step_for_clause(self, clause: str, materials: List[str]) -> Tuple[str, List[str]]:
        low = clause.lower()
        material = _find_material(clause, materials)
        measure = _extract_measure(clause)
        needs = []

        if any(k in low for k in ("spin coat", "spin-coat", "spincoat", "spincoating")):
            if not material:
                material = "[photoresist/polymer]"
                needs.append("Which material should I spin coat?")
            if not measure:
                measure = "[thickness]"
                needs.append("What spin-coat thickness should I use? Include a unit.")
            return f"Spin coat {material} ({measure}).", needs

        if any(k in low for k in ("deposit", "deposition", "evaporate", "sputter", "ald", "cvd")):
            if not material:
                material = "[material]"
                needs.append("Which material should I deposit?")
            if not measure:
                measure = "[thickness]"
                needs.append("What deposition thickness should I use? Include a unit.")
            return f"Deposit {material} ({measure}).", needs

        if "bond" in low:
            if not material:
                material = "[bonding material]"
                needs.append("Which bonding material should I use?")
            if not measure:
                measure = "[thickness]"
                needs.append("What bond-layer thickness should I use? Include a unit.")
            return f"Bond {material} ({measure}); select the stack to bond if needed.", needs

        if "etch" in low or "rie" in low or "drie" in low:
            if not material:
                material = "[material]"
                needs.append("Which material should I etch?")
            if not measure:
                measure = "[depth]"
                needs.append("What etch depth should I use? Include a unit.")
            return f"Etch {material} ({measure}, anisotropic).", needs

        if "strip" in low or "remove resist" in low:
            if not material:
                material = "[material]"
                needs.append("Which material should I strip?")
            return f"Strip material {material}.", needs

        if "lift" in low and "off" in low:
            if not material:
                material = "[photoresist]"
                needs.append("Which photoresist should I use for lift-off?")
            return f"Lift-off {material}.", needs

        if "clean" in low or "surface prep" in low or "surface preparation" in low:
            target = material or "wafer"
            return f"Clean / Surface prep {target}.", needs

        if "bake" in low or "anneal" in low:
            if not material:
                material = "[material]"
                needs.append("Which material should I bake or anneal?")
            return f"Bake / anneal {material}.", needs

        if "lith" in low or "pattern" in low or "develop" in low:
            if not material:
                material = "[photoresist]"
                needs.append("Which photoresist should I use for lithography?")
            return f"Lithography {material}; add the lithography window positions.", needs

        if "flip" in low:
            return "Flip stack.", needs

        needs.append(f"I couldn’t match this text to a supported operation: “{clause}”.")
        return "", needs


def _iter_materials(context: Dict) -> Iterable[str]:
    seen = set()
    for key in ("active_materials", "library_materials"):
        values = context.get(key, [])
        if not isinstance(values, list):
            continue
        for value in values:
            txt = str(value or "").strip()
            if txt and txt not in seen:
                seen.add(txt)
                yield txt


_OPERATION_ALIASES = (
    ("Set substrate", "set substrate"),
    ("Clean / Surface prep", "surface preparation"),
    ("Clean / Surface prep", "surface prep"),
    ("Spincoating", "spin coating"),
    ("Spincoating", "spin-coating"),
    ("Spincoating", "spin coat"),
    ("Spincoating", "spin-coat"),
    ("Spincoating", "spincoat"),
    ("Spincoating", "spincoating"),
    ("Strip material", "remove photoresist"),
    ("Strip material", "remove resist"),
    ("Strip material", "remove pr"),
    ("Bake/Annealing", "annealing"),
    ("Bake/Annealing", "anneal"),
    ("Bake/Annealing", "bake"),
    ("Bake/Annealing", "cure"),
    ("Lithography", "lithography"),
    ("Lithography", "litho"),
    ("Lithography", "windows"),
    ("Lithography", "window"),
    ("Lithography", "openings"),
    ("Lithography", "opening"),
    ("Lithography", "pattern"),
    ("Lithography", "develop"),
    ("Lift-off", "lift-off"),
    ("Lift-off", "lift off"),
    ("Lift-off", "liftoff"),
    ("Clean / Surface prep", "clean"),
    ("Deposit", "evaporate"),
    ("Deposit", "evaporation"),
    ("Deposit", "sputter"),
    ("Deposit", "sputtering"),
    ("Deposit", "deposition"),
    ("Deposit", "depositing"),
    ("Deposit", "deposited"),
    ("Deposit", "deposion"),
    ("Deposit", "depostion"),
    ("Deposit", "deposite"),
    ("Deposit", "depositon"),
    ("Deposit", "backfilled with"),
    ("Deposit", "backfill with"),
    ("Deposit", "filled with"),
    ("Deposit", "fill with"),
    ("Deposit", "backfilled"),
    ("Deposit", "backfill"),
    ("Deposit", "filled"),
    ("Deposit", "fill"),
    ("Deposit", "deposit"),
    ("Deposit", "coat"),
    ("Deposit", "apply"),
    ("Deposit", "grow"),
    ("Deposit", "add"),
    ("Bond", "bonding"),
    ("Bond", "bonded"),
    ("Bond", "bond"),
    ("Etch", "etching"),
    ("Etch", "etched"),
    ("Etch", "etch"),
    ("Etch", "drie"),
    ("Etch", "rie"),
    ("Strip material", "stripped"),
    ("Strip material", "strip"),
    ("Flip", "flip"),
)


def _resolve_conversation_request(user_request: str, context: Dict) -> str:
    user_request = _normalize_user_request(user_request)
    if not _request_is_clarification_fragment(user_request, context):
        return user_request

    previous = _last_detailed_user_request(context, exclude=user_request)
    if not previous:
        return user_request

    if _request_is_measure_only(user_request):
        return f"{previous} {user_request}"
    if _request_is_sparse_followup(user_request, context):
        return previous
    return user_request


def _request_is_clarification_fragment(text: str, context: Dict) -> bool:
    text = _normalize_user_request(text)
    if not text:
        return False
    if _request_is_measure_only(text):
        return True
    if _request_is_sparse_followup(text, context):
        return True
    recent = str((context or {}).get("recent_operation", "") or "").strip()
    if recent == "Lithography" and _extract_window_value(text):
        return True
    return False


def _request_is_measure_only(text: str) -> bool:
    text = _normalize_user_request(text)
    if not text:
        return False
    stripped = re.sub(rf"\b{_NUMBER_PATTERN}\s*(?:{_UNIT_PATTERN})\b", "", text, flags=re.IGNORECASE)
    stripped = re.sub(r"\b(of|for|the|a|an|is|range|window|thickness|depth)\b", "", stripped, flags=re.IGNORECASE)
    stripped = re.sub(r"[\s.,;:]+", "", stripped)
    return stripped == "" and bool(_extract_all_measure_um(text))


def _last_detailed_user_request(context: Dict, exclude: str = "") -> str:
    context = context if isinstance(context, dict) else {}
    exclude = _normalize_user_request(exclude).lower()
    recent = context.get("recent_user_requests", [])
    if not isinstance(recent, list):
        return ""
    known = _unique(list(_iter_materials(context)))
    for item in reversed(recent):
        request = _normalize_user_request(item)
        if not request or request.lower() == exclude:
            continue
        if _request_has_material_and_geometry(request, known) or (_operation_matches(request) and _find_material_mentions(request, known)):
            return request
    return ""


def _request_contains_operation_intent(text: str, context: Dict) -> bool:
    text = _normalize_user_request(text)
    if _operation_matches(text):
        return True
    if _clause_requests_etch(text, context):
        return True
    recent = str((context or {}).get("recent_operation", "") or "").strip()
    if recent == "Lithography" and _extract_window_value(text):
        return True
    materials = list(_iter_materials(context or {}))
    return bool(_find_material_mentions(text, materials) and _extract_all_measure_um(text))


def _request_is_sparse_followup(text: str, context: Dict) -> bool:
    text = _normalize_user_request(text)
    if _extract_all_measure_um(text) or _extract_window_value(text):
        return False
    materials = list(_iter_materials(context or {}))
    if _find_material_mentions(text, materials):
        return False
    return bool(_operation_matches(text))


def _request_has_material_and_geometry(text: str, known: List[str]) -> bool:
    text = _normalize_user_request(text)
    return bool(_find_material_mentions(text, known) and (_extract_all_measure_um(text) or _extract_window_value(text)))


def _plan_score(plan: Dict) -> int:
    if not isinstance(plan, dict):
        return -1000
    score = 0
    if plan.get("status") == "ready":
        score += 100
    score -= 20 * len(plan.get("missing", []) or [])
    for command in plan.get("commands", []) or []:
        if not isinstance(command, dict):
            continue
        action = command.get("action")
        if action == "operation":
            score += 10
            if command.get("operation") in {"Deposit", "Spincoating", "Bond"} and command.get("thickness_um"):
                score += 3
            if command.get("operation") == "Etch" and command.get("depth_um"):
                score += 3
            if command.get("operation") == "Lithography" and command.get("windows"):
                score += 3 + len(command.get("windows", []))
            if command.get("operation") == "Lithography" and str(command.get("windowing_mode", "")).lower() == "manual":
                score += 3
        elif action in {"set_substrate", "add_material"}:
            score += 2
    return score


def _model_adds_unrequested_physical_layers(model_plan: Dict, repair_plan: Dict) -> bool:
    model_layers = _plan_physical_layer_counts(model_plan)
    repair_layers = _plan_physical_layer_counts(repair_plan)
    if not repair_layers:
        return False
    return bool(model_layers - repair_layers)


def _plan_physical_layer_counts(plan: Dict) -> Counter:
    counts = Counter()
    if not isinstance(plan, dict):
        return counts
    for command in plan.get("commands", []) or []:
        if not isinstance(command, dict):
            continue
        action = command.get("action")
        if action == "set_substrate":
            counts[(
                "Substrate",
                _plan_layer_material_key(command.get("material")),
                _plan_layer_measure_key(command.get("thickness_um")),
            )] += 1
            continue
        if action != "operation" or command.get("operation") not in {"Deposit", "Spincoating", "Bond"}:
            continue
        counts[(
            str(command.get("operation") or "").strip(),
            _plan_layer_material_key(command.get("material")),
            _plan_layer_measure_key(command.get("thickness_um")),
        )] += 1
    return counts


def _plan_layer_material_key(material: str) -> str:
    return re.sub(r"\s+", " ", str(material or "").strip().lower())


def _plan_layer_measure_key(value):
    value = _coerce_float(value)
    if value is None:
        return None
    return round(float(value), 9)


def _copy_context_for_request(context: Dict) -> Dict:
    context = context if isinstance(context, dict) else {}
    local = dict(context)
    thicknesses = context.get("material_thickness_um", {})
    local["material_thickness_um"] = dict(thicknesses) if isinstance(thicknesses, dict) else {}
    order = context.get("top_material_order", [])
    local["top_material_order"] = list(order) if isinstance(order, list) else []
    latest_etch = context.get("latest_etch", {})
    if isinstance(latest_etch, dict):
        local["latest_etch"] = dict(latest_etch)
    return local


def _record_layer_addition(context: Dict, material: str, thickness_um, operation: str = "") -> Dict:
    material = str(material or "").strip()
    thickness = _coerce_float(thickness_um)
    if not material or thickness is None or thickness <= 0.0:
        return context
    thicknesses = context.setdefault("material_thickness_um", {})
    previous = _coerce_float(thicknesses.get(material)) or 0.0
    thicknesses[material] = previous + float(thickness)
    order = context.setdefault("top_material_order", [])
    if material in order:
        order.remove(material)
    order.insert(0, material)
    if str(operation or "").strip() == "Deposit":
        _consume_open_cavity_depth(context, float(thickness))
    return context


def _record_layer_etch(context: Dict, material: str, depth_um) -> Dict:
    material = str(material or "").strip()
    depth = _coerce_float(depth_um)
    if not material or depth is None or depth <= 0.0:
        return context
    thicknesses = context.setdefault("material_thickness_um", {})
    current = _coerce_float(thicknesses.get(material)) or 0.0
    remaining = max(0.0, current - float(depth))
    if remaining > 1e-12:
        thicknesses[material] = remaining
    else:
        thicknesses.pop(material, None)
        order = context.setdefault("top_material_order", [])
        while material in order:
            order.remove(material)
    context["latest_etch"] = {
        "material": material,
        "depth_um": float(depth),
    }
    context["open_cavity_remaining_um"] = float(depth)
    return context


def _consume_open_cavity_depth(context: Dict, thickness_um: float) -> None:
    remaining = _coerce_float((context or {}).get("open_cavity_remaining_um"))
    thickness = _coerce_float(thickness_um)
    if remaining is None or remaining <= 0.0 or thickness is None or thickness <= 0.0:
        return
    context["open_cavity_remaining_um"] = max(0.0, float(remaining) - float(thickness))


def _remaining_fill_depth_um(context: Dict):
    remaining = _coerce_float((context or {}).get("open_cavity_remaining_um"))
    if remaining is not None and remaining > 1e-12:
        return float(remaining)
    latest = (context or {}).get("latest_etch", {})
    if isinstance(latest, dict):
        depth = _coerce_float(latest.get("depth_um"))
        if depth is not None and depth > 1e-12:
            return float(depth)
    return None


def _record_substrate_set(context: Dict, material: str, thickness_um) -> Dict:
    material = str(material or "").strip()
    if not material:
        return context
    thickness = _coerce_float(thickness_um)
    if thickness is not None and thickness > 0.0:
        thicknesses = context.setdefault("material_thickness_um", {})
        thicknesses[material] = float(thickness)
    order = context.setdefault("top_material_order", [])
    while material in order:
        order.remove(material)
    order.append(material)
    return context


def _record_layer_strip(context: Dict, material: str) -> Dict:
    material = str(material or "").strip()
    if not material:
        return context
    thicknesses = context.setdefault("material_thickness_um", {})
    thicknesses.pop(material, None)
    order = context.setdefault("top_material_order", [])
    while material in order:
        order.remove(material)
    return context


def _operation_matches(text: str) -> List[Tuple[int, int, str]]:
    matches = []
    for operation, alias in sorted(_OPERATION_ALIASES, key=lambda item: len(item[1]), reverse=True):
        alias_pattern = re.escape(alias).replace(r"\ ", r"\s+")
        pattern = re.compile(rf"(?<![a-z0-9_]){alias_pattern}(?![a-z0-9_])", re.IGNORECASE)
        for match in pattern.finditer(text or ""):
            matches.append((match.start(), match.end(), operation))
    matches.sort(key=lambda item: (item[0], -(item[1] - item[0])))

    filtered = []
    occupied = []
    for start, end, operation in matches:
        if any(start < used_end and end > used_start for used_start, used_end in occupied):
            continue
        filtered.append((start, end, operation))
        occupied.append((start, end))
    filtered.sort(key=lambda item: item[0])
    return filtered


def _extract_command_segments(text: str, context: Dict, known: List[str]) -> List[Tuple[str, str]]:
    text = _normalize_user_request(text)
    matches = _operation_matches(text)
    records = []
    substrate_segment = _extract_initial_substrate_segment(text, matches, known)
    if substrate_segment:
        records.append({
            "operation": "Set substrate",
            "clause": substrate_segment,
            "after_next": False,
        })
    for idx, (start, end, operation) in enumerate(matches):
        next_start = matches[idx + 1][0] if idx + 1 < len(matches) else len(text)
        next_operation = matches[idx + 1][2] if idx + 1 < len(matches) else ""
        clause_end = next_start
        clause_start = start
        if operation == "Lithography":
            if idx == 0:
                previous_end = 0
            elif matches[idx - 1][2] == "Lithography":
                previous_end = start
            else:
                previous_end = matches[idx - 1][1]
            clause_start = _lithography_clause_start(text, start, previous_end)
        elif operation == "Deposit" and _is_fill_alias(text[start:end]):
            clause_start = _clause_boundary_start(text, start)
        after_next = False
        if idx + 1 < len(matches) and re.search(r"\bafter\s*$", text[end:next_start], re.IGNORECASE):
            after_match = list(re.finditer(r"\bafter\b", text[end:next_start], re.IGNORECASE))[-1]
            clause_end = end + after_match.start()
            after_next = True
        clause = text[clause_start:clause_end].strip(" .,\n\t")
        if operation == "Etch" and next_operation != "Etch" and _is_etch_reference_only_clause(clause, known):
            continue
        if operation == "Deposit" and not _segment_has_material_or_measure(clause, known):
            continue
        if clause:
            records.append({
                "operation": operation,
                "clause": clause,
                "after_next": after_next,
            })

    records = _merge_adjacent_operation_records(records, known)

    if records:
        segments = []
        idx = 0
        while idx < len(records):
            if records[idx].get("after_next") and idx + 1 < len(records):
                segments.append((records[idx + 1]["operation"], records[idx + 1]["clause"]))
                segments.append((records[idx]["operation"], records[idx]["clause"]))
                idx += 2
                continue
            segments.append((records[idx]["operation"], records[idx]["clause"]))
            idx += 1
        return segments

    recent = str((context or {}).get("recent_operation", "") or "").strip()
    if recent == "Lithography" and _extract_window_value(text):
        return [("Lithography", text)]

    if _extract_lithography_windows(text):
        return [("Lithography", text)]

    if _clause_requests_etch(text, context):
        return [("Etch", text)]

    if _find_material_mentions(text, known) and _extract_all_measure_um(text):
        return [("Deposit", text)]

    return []


def _merge_adjacent_operation_records(records: List[Dict], known: List[str] = None) -> List[Dict]:
    merged = []
    for record in records:
        if (
            merged
            and not merged[-1].get("after_next")
            and not record.get("after_next")
            and merged[-1].get("operation") == record.get("operation")
        ):
            operation = record.get("operation")
            if operation == "Lithography" or (
                operation == "Etch"
                and (
                    _is_etch_method_only_clause(merged[-1].get("clause", ""), known or [])
                    or _is_etch_method_only_clause(record.get("clause", ""), known or [])
                )
            ):
                merged[-1]["clause"] = f"{merged[-1].get('clause', '')} {record.get('clause', '')}".strip()
                continue
        merged.append(dict(record))
    return merged


def _clause_boundary_start(text: str, start: int) -> int:
    start = max(0, min(int(start or 0), len(text)))
    candidate = 0
    prefix = text[:start]
    for match in re.finditer(r"\b(?:then|after that)\b|;|\n|(?<!\d)\.(?!\d)", prefix, re.IGNORECASE):
        candidate = max(candidate, match.end())
    return candidate


def _is_fill_alias(text: str) -> bool:
    return bool(re.search(r"\b(?:backfill(?:ed)?|fill(?:ed)?)\b", str(text or ""), re.IGNORECASE))


def _clause_requests_remaining_fill(text: str) -> bool:
    low = str(text or "").lower()
    return bool(
        re.search(r"\b(?:backfill(?:ed)?|fill(?:ed)?)\b", low)
        and re.search(r"\b(?:remaining|rest|leftover|unfilled|cavity|trench|gap|depth)\b", low)
    )


def _etch_target_mentions(clause: str, known: List[str]) -> List[Tuple[int, int, str]]:
    return _filter_etch_target_mentions(clause, _find_material_mentions(clause, known or []))


def _is_etch_reference_only_clause(clause: str, known: List[str]) -> bool:
    low = str(clause or "").lower().strip()
    if not low:
        return False
    if _extract_measure_um(low) is not None or _has_full_etch_term(low) or _has_half_etch_term(low):
        return False
    if _etch_target_mentions(low, known):
        return False
    return bool(re.search(r"\b(?:etch|etched|etching)\b", low))


def _is_etch_method_only_clause(clause: str, known: List[str]) -> bool:
    low = str(clause or "").lower().strip()
    if not low:
        return False
    if _extract_measure_um(low) is not None:
        return False
    if _etch_target_mentions(low, known):
        return False
    return bool(_extract_etch_method(low) or re.search(r"\b(?:drie|icp[-\s]?rie|rie)\b", low))


def _lithography_clause_start(text: str, start: int, minimum_start: int) -> int:
    minimum_start = max(0, min(int(minimum_start or 0), start))
    prefix = text[minimum_start:start]
    candidate = minimum_start

    for match in re.finditer(r"\b(?:then|after that)\b|;|\n|(?<!\d)\.(?!\d)", prefix, re.IGNORECASE):
        candidate = max(candidate, minimum_start + match.end())

    for match in re.finditer(
        r"\band(?:\s+we)?\s+(?=(?:make|create|open|define|pattern|a|an|the|central|center|centre|middle|manual|custom|customizable|customisable|interactive|freehand|user[-\s]?defined|pause|inverse|inversed|inverted|invert|negative|positive|\d))",
        prefix,
        re.IGNORECASE,
    ):
        candidate = max(candidate, minimum_start + match.end())

    leading = text[candidate:start]
    if (
        _extract_window_value(leading) is not None
        or re.search(
            r"\b(make|create|open|define|pattern|central|center|centre|middle|manual|custom|customizable|customisable|interactive|freehand|user[-\s]?defined|pause|inverse|inversed|inverted|invert|negative|positive)\b",
            leading,
            re.IGNORECASE,
        )
    ):
        return candidate
    return start


def _segment_has_material_or_measure(text: str, known: List[str]) -> bool:
    return bool(_find_material_mentions(text, known) or _extract_all_measure_um(text))


def _filter_surface_material_mentions(text: str, mentions: List[Tuple[int, int, str]]) -> List[Tuple[int, int, str]]:
    if len(mentions) <= 1:
        return mentions
    text = _normalize_user_request(text)
    filtered = []
    for index, (start, end, material) in enumerate(mentions):
        prefix = text[max(0, start - 24):start].lower()
        segment = _material_measure_segment(text, mentions, index)
        is_surface_target = bool(re.search(r"\b(on|onto|over|upon)\s+(?:the\s+)?$", prefix))
        has_local_measure = _extract_measure_um(segment) is not None
        if is_surface_target and not has_local_measure:
            continue
        filtered.append((start, end, material))
    return filtered


def _extract_initial_substrate_segment(text: str, matches: List[Tuple[int, int, str]], known: List[str]) -> str:
    text = _normalize_user_request(text)
    first_start = matches[0][0] if matches else len(text)
    preamble = text[:first_start].strip(" .,\n\t")
    if not preamble:
        return ""
    low = preamble.lower()
    if not re.search(r"\b(wafer|substrate)\b", low):
        return ""
    if _extract_measure_um(preamble) is not None or _extract_substrate_like_unitless_um(preamble) is not None:
        return preamble
    if _find_material(preamble, known) or _default_substrate_material(preamble, known):
        return preamble
    return ""


def _parse_layer_commands(
    clause: str,
    operation: str,
    context: Dict,
    known: List[str],
    photoresists: List[str],
) -> Tuple[List[Dict], List[str], List[str]]:
    mentions = _find_material_mentions(clause, known)
    if operation in {"Deposit", "Spincoating"}:
        mentions = _filter_surface_material_mentions(clause, mentions)
        mentions = _prefer_specific_photoresist_mentions(mentions, photoresists)
    if not mentions and operation == "Spincoating":
        material = _default_photoresist_material(context, known, photoresists)
        if material:
            mentions = [(0, 0, material)]

    if not mentions:
        question = {
            "Deposit": "Which material should I deposit?",
            "Spincoating": "Which material should I spin coat?",
            "Bond": "Which bonding material should I use?",
        }.get(operation, "Which material should I use?")
        return [], [], [question]

    command_entries = []
    missing = []
    ordered_values = _ordered_measure_values_for_mentions(clause, mentions)
    for index, (_start, _end, material) in enumerate(mentions):
        thickness = ordered_values[index] if index < len(ordered_values) else None
        if thickness is None:
            segment = _material_measure_segment(clause, mentions, index)
            default_unit = _last_explicit_unit(clause[:mentions[index][0]]) or _first_explicit_unit(clause)
            thickness = _extract_measure_um(segment)
            if thickness is None and default_unit:
                thickness = _extract_unitless_measure_um(segment, default_unit)
            if thickness is None and operation == "Bond":
                thickness = (
                    _extract_substrate_like_unitless_um(segment)
                    or _extract_substrate_like_unitless_um(clause)
                )
            if thickness is None and operation == "Deposit" and _clause_requests_remaining_fill(clause):
                thickness = _remaining_fill_depth_um(context)
        if thickness is None or thickness <= 0.0:
            if operation == "Deposit" and _clause_requests_remaining_fill(clause):
                missing.append(
                    f"Give the fill thickness for {material}; I couldn’t determine the remaining cavity depth."
                )
            else:
                missing.append(f"Give the {material} thickness with a unit.")
            continue

        op_name = operation
        if op_name == "Deposit" and _is_photoresist_material(material, photoresists):
            op_name = "Spincoating"
        command = {
            "action": "operation",
            "operation": op_name,
            "material": material,
            "thickness_um": float(thickness),
        }
        priority = 0 if re.search(r"\bfirst\b", _material_measure_segment(clause, mentions, index), re.IGNORECASE) else 1
        command_entries.append((priority, index, command, material))

    command_entries.sort(key=lambda item: (item[0], item[1]))
    commands = [entry[2] for entry in command_entries]
    used_materials = [entry[3] for entry in command_entries]
    return commands, _unique(used_materials), _unique(missing)


def _ordered_measure_values_for_mentions(text: str, mentions: List[Tuple[int, int, str]]) -> List[float]:
    values = [None for _item in mentions]
    measures = _extract_measure_mentions_um(text, infer_unitless=False)
    if len(measures) >= len(mentions):
        for index in range(len(mentions)):
            values[index] = measures[index][0]
        return values

    material_spans = [(start, end) for start, end, _material in mentions]
    measures = [
        item for item in _extract_measure_mentions_um(text, infer_unitless=True)
        if not any(item[1] >= span_start and item[2] <= span_end for span_start, span_end in material_spans)
    ]
    if len(measures) < len(mentions):
        return values
    for index in range(len(mentions)):
        values[index] = measures[index][0]
    return values


def _prefer_specific_photoresist_mentions(
    mentions: List[Tuple[int, int, str]],
    photoresists: List[str],
) -> List[Tuple[int, int, str]]:
    specific = [
        item
        for item in mentions
        if _is_photoresist_material(item[2], photoresists)
        and str(item[2]).strip().lower() != "photoresist"
    ]
    if not specific:
        return mentions
    return [
        item
        for item in mentions
        if not (
            _is_photoresist_material(item[2], photoresists)
            and str(item[2]).strip().lower() == "photoresist"
        )
    ]


def _material_measure_segment(text: str, mentions: List[Tuple[int, int, str]], index: int) -> str:
    text = _normalize_user_request(text)
    start, end, _material = mentions[index]
    left = 0
    right = len(text)

    if index > 0:
        left = mentions[index - 1][1]
    if index + 1 < len(mentions):
        right = mentions[index + 1][0]

    boundary = re.compile(r"\b(?:and|then|after that)\b|[,;\n]", re.IGNORECASE)
    for match in boundary.finditer(text):
        if match.end() <= start:
            left = max(left, match.end())
        elif match.start() >= end:
            right = min(right, match.start())
            break

    return text[left:right].strip()


def _materials_for_segment(text: str, known: List[str]) -> List[str]:
    return _unique(material for _start, _end, material in _find_material_mentions(text, known))


def _latest_used_photoresist(used_materials: List[str], photoresists: List[str]) -> str:
    for material in reversed(used_materials or []):
        if _is_photoresist_material(material, photoresists):
            return material
    return ""


def _segment_mentions_generic_photoresist(text: str) -> bool:
    low = str(text or "").lower()
    return bool(re.search(r"\b(photo\s*resist|photoresist|resist|pr)\b", low)) and not bool(re.search(r"\bsu[-\s]?8\b|\bpmma\b", low))


def _segment_mentions_previous_layer(text: str) -> bool:
    return bool(re.search(r"\b(it|that|this|same layer|previous layer)\b", str(text or ""), re.IGNORECASE))


def _is_photoresist_material(material: str, photoresists: List[str]) -> bool:
    material = str(material or "").strip()
    if not material:
        return False
    if material in photoresists:
        return True
    low = material.lower()
    return "photoresist" in low or "photo resist" in low or low.startswith("pr_") or "pmma" in low


def _default_photoresist_material(context: Dict, known: List[str], photoresists: List[str]) -> str:
    context = context if isinstance(context, dict) else {}
    candidate_groups = [
        context.get("top_material_order", []),
        context.get("active_materials", []),
        photoresists,
        known,
    ]
    for group in candidate_groups:
        if not isinstance(group, list):
            continue
        for candidate in group:
            material = _canonical_material(str(candidate), known)
            if material and _is_photoresist_material(material, photoresists):
                return material
    return ""


def _default_substrate_material(text: str, known: List[str]) -> str:
    low = str(text or "").lower()
    if not re.search(r"\b(wafer|substrate)\b", low):
        return ""
    return _canonical_material("Silicon (Si)", known) or _canonical_material("Silicon", known)


def _find_material(text: str, materials: List[str]) -> str:
    mentions = _find_material_mentions(text, materials)
    if mentions:
        return mentions[0][2]
    return _alias_material(text, materials)


def _find_material_mentions(text: str, materials: List[str]) -> List[Tuple[int, int, str]]:
    low = _normalize_user_request(text).lower()
    candidates = []
    for material in materials:
        best = None
        for term in _material_search_terms(material):
            term_low = term.lower().strip()
            if not term_low:
                continue
            term_pattern = re.escape(term_low).replace(r"\ ", r"\s+")
            pattern = re.compile(rf"(?<![a-z0-9_]){term_pattern}(?![a-z0-9_])", re.IGNORECASE)
            match = pattern.search(low)
            if not match:
                continue
            priority = _material_match_priority(material, term_low, low)
            candidate = (match.start(), match.end(), material, priority)
            if best is None or candidate[0] < best[0] or (candidate[0] == best[0] and candidate[3] > best[3]):
                best = candidate
        if best is not None:
            candidates.append(best)

    candidates.sort(key=lambda item: (item[0], -(item[1] - item[0]), -item[3]))
    found = []
    occupied = []
    seen_materials = set()
    for start, end, material, _priority in candidates:
        if material.lower() in seen_materials:
            continue
        if _material_mention_is_parenthetical_alias(low, start, end, material, found):
            continue
        if any(start < used_end and end > used_start for used_start, used_end in occupied):
            continue
        seen_materials.add(material.lower())
        occupied.append((start, end))
        found.append((start, end, material))
    found.sort(key=lambda item: item[0])
    return found


def _material_mention_is_parenthetical_alias(
    text_low: str,
    start: int,
    end: int,
    material: str,
    accepted: List[Tuple[int, int, str]],
) -> bool:
    if not accepted:
        return False
    previous_start, previous_end, previous_material = accepted[-1]
    if previous_material.lower() == str(material or "").lower():
        return False
    between = text_low[previous_end:start]
    if not re.fullmatch(r"\s*\(\s*", between or ""):
        return False
    close_index = text_low.find(")", end)
    if close_index < 0 or close_index - start > 24:
        return False
    alias = text_low[start:end].strip()
    previous_terms = {term.lower().strip() for term in _material_search_terms(previous_material)}
    return alias in previous_terms


def _material_match_priority(material: str, term: str, text_low: str) -> int:
    material_low = str(material or "").lower()
    base_low = _material_base_name(material).lower()
    priority = len(term)
    if material_low and material_low in text_low:
        priority += 1000
    elif base_low and base_low in text_low:
        priority += 1000
    if len(_material_term_words(term)) > 1:
        priority += 500
    if term in {"su8", "su-8", "su 8"}:
        if "pr_su-8" in material_low:
            priority += 700
        elif "su-8" in material_low:
            priority += 400
    if material_low == "photoresist":
        priority -= 50
    return priority


def _material_search_terms(material: str) -> List[str]:
    txt = str(material or "").strip()
    low = txt.lower()
    terms = [txt]
    base = _material_base_name(txt)
    if base and base.lower() != low:
        terms.append(base)
    for variant in _material_spelling_variants(base or txt):
        terms.append(variant)
    short = _material_short_name(txt)
    if short:
        terms.append(short)
    is_compound = len(_material_term_words(base or txt)) > 1
    head = re.split(r"\s|\(", txt, maxsplit=1)[0].strip()
    if head and len(head) > 1 and not is_compound:
        terms.append(head)

    if _material_base_is(txt, "chromium"):
        terms.extend(["chrome", "chromium", "cr"])
    if _material_base_is(txt, "gold"):
        terms.extend(["gold", "au"])
    if _material_base_is(txt, "aluminum", "aluminium"):
        terms.extend(["aluminum", "aluminium", "al"])
    if _material_base_is(txt, "titanium"):
        terms.extend(["titanium", "ti"])
    if _material_base_is(txt, "nickel"):
        terms.extend(["nickel", "ni"])
    if _material_base_is(txt, "copper"):
        terms.extend(["copper", "cu"])
    if _material_base_is(txt, "platinum"):
        terms.extend(["platinum", "pt"])
    if _material_base_is(txt, "tungsten"):
        terms.extend(["tungsten", "w"])
    short_low = short.lower()
    if _material_base_is(txt, "silicon dioxide") or short_low == "sio2":
        terms.extend(["silicon dioxide", "silicon oxide", "sio2", "silica", "oxide"])
    if _material_base_is(txt, "silicon nitride") or short_low == "si3n4":
        terms.extend(["silicon nitride", "silicon nitide", "si3n4", "nitride", "nitide"])
    if _material_base_is(txt, "silicon carbide") or short_low == "sic":
        terms.extend(["silicon carbide", "sic"])
    if _material_base_is(txt, "aluminum nitride", "aluminium nitride") or short_low == "aln":
        terms.extend(["aluminum nitride", "aluminium nitride", "aluminum nitide", "aluminium nitide", "aln"])
    if _material_base_is(txt, "titanium nitride") or short_low == "tin":
        terms.extend(["titanium nitride", "titanium nitide", "tin"])
    if low == "photoresist" or txt == "Photoresist":
        terms.extend(["photoresist", "photo resist", "resist", "pr material", "photoresist material"])
    if "pmma" in low:
        terms.append("pmma")
    if "su-8" in low or "su8" in low:
        terms.extend(["su-8", "su8", "su 8"])
    if "mempex" in low:
        terms.extend(["mempex", "mempax"])

    unique = []
    seen = set()
    for term in sorted(terms, key=len, reverse=True):
        clean = str(term or "").strip()
        key = clean.lower()
        if clean and key not in seen:
            seen.add(key)
            unique.append(clean)
    return unique


def _material_base_name(material: str) -> str:
    txt = str(material or "").strip()
    if not txt:
        return ""
    return re.sub(r"\s*\([^)]*\)\s*$", "", txt).strip() or txt


def _material_base_is(material: str, *names: str) -> bool:
    base = _material_base_name(material).lower()
    return any(base == str(name or "").lower() for name in names)


def _material_term_words(term: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", str(term or "").lower())


def _material_spelling_variants(term: str) -> List[str]:
    txt = str(term or "").strip()
    variants = []
    if not txt:
        return variants
    if re.search("aluminum", txt, re.IGNORECASE):
        variants.append(re.sub("aluminum", "aluminium", txt, flags=re.IGNORECASE))
    if re.search("aluminium", txt, re.IGNORECASE):
        variants.append(re.sub("aluminium", "aluminum", txt, flags=re.IGNORECASE))
    if re.search("nitride", txt, re.IGNORECASE):
        variants.append(re.sub("nitride", "nitide", txt, flags=re.IGNORECASE))
    return variants


def _clause_has_explicit_operation(text: str) -> bool:
    low = str(text or "").lower()
    operation_terms = (
        "deposit", "deposition", "deposion", "depostion", "deposite", "depositon",
        "evaporate", "sputter", "ald", "cvd",
        "spin coat", "spin-coat", "spincoat", "spincoating",
        "bond", "etch", "rie", "drie", "strip", "remove resist",
        "lift-off", "lift off", "liftoff", "clean", "surface prep",
        "surface preparation", "bake", "anneal", "lith", "pattern",
        "develop", "flip",
    )
    return any(term in low for term in operation_terms)


def _clause_requests_etch(text: str, context: Dict) -> bool:
    low = str(text or "").lower()
    if any(term in low for term in ("etch", "rie", "drie")):
        return True
    recent = str((context or {}).get("recent_operation", "") or "").strip().lower()
    if recent != "etch" or _clause_has_explicit_operation(text):
        return False
    return bool(
        _extract_measure_um(text) is not None
        or _has_full_etch_term(text)
        or _has_half_etch_term(text)
    )


def _request_command_requirements(user_request: str, context: Dict, known: List[str]) -> List[Dict]:
    requirements = []
    context = context if isinstance(context, dict) else {}
    photoresists = [str(x) for x in context.get("photoresist_materials", []) if str(x).strip()]
    for operation, clause in _extract_command_segments(user_request, context, known):
        if operation == "Set substrate":
            material = _find_material(clause, known) or _default_substrate_material(clause, known)
            requirements.append({"action": "set_substrate", "material": material})
            continue

        if operation in {"Deposit", "Bond", "Etch", "Strip material", "Bake/Annealing", "Clean / Surface prep"}:
            if operation == "Etch":
                materials = _unique(
                    material
                    for _start, _end, material in _filter_etch_target_mentions(
                        clause,
                        _find_material_mentions(clause, known),
                    )
                )
            else:
                materials = _materials_for_segment(clause, known)
            if operation == "Strip material" and _segment_mentions_generic_photoresist(clause):
                material = _default_photoresist_material(context, known, photoresists)
                materials = [material] if material else materials
            if materials:
                for material in materials:
                    req = {"action": "operation", "operation": operation, "material": material}
                    requirements.append(req)
            else:
                req = {"action": "operation", "operation": operation, "material": ""}
                requirements.append(req)
            continue

        if operation == "Spincoating":
            materials = _materials_for_segment(clause, known)
            if not materials:
                material = _default_photoresist_material(context, known, photoresists)
                materials = [material] if material else []
            if materials:
                for material in materials:
                    req = {"action": "operation", "operation": "Spincoating", "material": material}
                    requirements.append(req)
            else:
                req = {"action": "operation", "operation": "Spincoating", "material": ""}
                requirements.append(req)
            continue

        if operation in {"Lithography", "Lift-off"}:
            material = (
                _find_material(clause, known)
                or _default_photoresist_material(context, known, photoresists)
            )
            req = {"action": "operation", "operation": operation, "material": material}
            requirements.append(req)
            continue

        if operation == "Flip":
            requirements.append({"action": "operation", "operation": "Flip", "material": ""})

    return _unique_requirements(requirements)


def _unique_requirements(requirements: List[Dict]) -> List[Dict]:
    out = []
    seen = set()
    for req in requirements:
        key = (
            str(req.get("action", "")),
            str(req.get("operation", "")),
            str(req.get("material", "")).lower(),
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(req)
    return out


def _plan_satisfies_requirement(plan: Dict, requirement: Dict, known: List[str]) -> bool:
    commands = plan.get("commands", []) if isinstance(plan, dict) else []
    action = str(requirement.get("action", "") or "")
    operation = str(requirement.get("operation", "") or "")
    material = str(requirement.get("material", "") or "").strip()
    material = _canonical_material(material, known) or material
    for command in commands:
        if not isinstance(command, dict) or command.get("action") != action:
            continue
        if action == "set_substrate":
            if not material:
                return True
            cmd_material = _canonical_material(str(command.get("material", "")), known) or str(command.get("material", ""))
            if cmd_material.lower() == material.lower():
                return True
            continue
        if action == "operation":
            if str(command.get("operation", "")) != operation:
                continue
            if not material or operation == "Flip":
                return True
            cmd_material = _canonical_material(str(command.get("material", "")), known) or str(command.get("material", ""))
            if cmd_material.lower() == material.lower():
                return True
    return False


def _requirement_label(requirement: Dict) -> str:
    action = str(requirement.get("action", "") or "")
    material = str(requirement.get("material", "") or "").strip()
    if action == "set_substrate":
        return f"Set substrate {material}".strip()
    operation = str(requirement.get("operation", "") or "").strip()
    parts = [operation, material]
    return " ".join(part for part in parts if part).strip()


def _parse_etch_commands(clause: str, context: Dict, known: List[str]) -> Tuple[List[Dict], List[str], List[str]]:
    mentions = _find_material_mentions(clause, known)
    mentions = _filter_etch_target_mentions(clause, mentions)
    if not mentions:
        return [], [], ["Which material should I etch?"]

    thicknesses = _context_material_thickness_um(context, known)
    commands = []
    used_materials = []
    missing = []

    for start, end, material in mentions:
        segment = _segment_for_material(clause, start, end)
        default_unit = _last_explicit_unit(clause[:start]) or _first_explicit_unit(clause)
        depth = _extract_measure_um(segment)
        if depth is None and default_unit:
            depth = _extract_unitless_measure_um(segment, default_unit)

        if depth is None:
            fraction = _etch_fraction_for_segment(segment, clause)
            if fraction is not None:
                layer_thickness = thicknesses.get(material)
                if layer_thickness is None or layer_thickness <= 0.0:
                    missing.append(
                        f"Give the current deposited thickness of {material}, or enter an explicit etch depth."
                    )
                    continue
                depth = layer_thickness * fraction

        if depth is None or depth <= 0.0:
            missing.append(
                f"Give the etch depth for {material}, including a unit, or say ‘half’ or ‘complete’."
            )
            continue

        command = {
            "action": "operation",
            "operation": "Etch",
            "material": material,
            "depth_um": float(depth),
            "etch_mode": _extract_etch_mode(segment) or _extract_etch_mode(clause) or "anisotropic",
        }
        etch_method = _extract_etch_method(segment) or _extract_etch_method(clause)
        if etch_method:
            command["method"] = etch_method
        commands.append(command)
        used_materials.append(material)

    return _order_commands_for_context(commands, context), _unique(used_materials), _unique(missing)


def _filter_etch_target_mentions(clause: str, mentions: List[Tuple[int, int, str]]) -> List[Tuple[int, int, str]]:
    filtered = []
    for start, end, material in mentions:
        if _is_process_method_material(material):
            continue
        prefix = clause[max(0, start - 32):start].lower()
        if re.search(r"\b(?:with|using|in|by)\s+(?:an?\s+|the\s+)?$", prefix):
            continue
        filtered.append((start, end, material))
    return filtered


def _is_process_method_material(material: str) -> bool:
    low = str(material or "").lower()
    return bool(
        "etchant" in low
        or "developer" in low
        or "solvent" in low
        or "drie" in low
        or re.search(r"\brie\b|\bicp[-\s]?rie\b", low)
    )


def _segment_for_material(text: str, start: int, end: int) -> str:
    text = _normalize_user_request(text)
    boundary = re.compile(r"\b(?:and|then|after that)\b|[,;\n]", re.IGNORECASE)
    left = 0
    right = len(text)
    for match in boundary.finditer(text):
        if match.end() <= start:
            left = match.end()
        elif match.start() >= end:
            right = match.start()
            break
    return text[left:right].strip()


def _context_material_thickness_um(context: Dict, known: List[str]) -> Dict[str, float]:
    raw = (context or {}).get("material_thickness_um", {})
    if not isinstance(raw, dict):
        return {}
    out = {}
    for key, value in raw.items():
        material = _canonical_material(str(key), known) or str(key)
        thickness = _coerce_float(value)
        if material and thickness is not None and thickness > 0.0:
            out[material] = float(thickness)
    return out


def _has_half_etch_term(text: str) -> bool:
    return bool(re.search(r"\b(half|halve|halved)\b|\b1\s*/\s*2\b|\b50\s*%", str(text or ""), re.IGNORECASE))


def _has_full_etch_term(text: str) -> bool:
    return bool(re.search(
        r"\b(complete|completely|full|fully|entire|entirely|all)\b|\bremove\s+all\b|\bclear\s+all\b",
        str(text or ""),
        re.IGNORECASE,
    ))


def _etch_fraction_for_segment(segment: str, full_text: str):
    if _has_half_etch_term(segment):
        return 0.5
    if _has_full_etch_term(segment):
        return 1.0
    if _has_full_etch_term(full_text) and not _has_half_etch_term(full_text):
        return 1.0
    return None


def _extract_etch_mode(text: str) -> str:
    low = str(text or "").lower()
    if re.search(r"\bisotropic\b", low):
        return "isotropic"
    if re.search(r"\banisotropic\b", low):
        return "anisotropic"
    return ""


def _extract_etch_method(text: str) -> str:
    txt = str(text or "")
    if re.search(r"\bDRIE\b", txt, re.IGNORECASE):
        return "DRIE (Bosch process)"
    if re.search(r"\bICP[-\s]?RIE\b", txt, re.IGNORECASE):
        return "ICP-RIE (SF6 / O2)"
    if re.search(r"\bHF\b", txt, re.IGNORECASE):
        return "HF"
    method_match = re.search(
        r"\bwith\s+([A-Za-z][A-Za-z0-9+\-/ ]{0,24}?)(?=\s+(?:for|to|through|until|and|then)\b|[().,;]|$)",
        txt,
        re.IGNORECASE,
    )
    if not method_match:
        return ""
    method = method_match.group(1).strip()
    if not method or method.lower() in {"a", "an", "the"}:
        return ""
    return method


def _extract_lithography_invert(text: str):
    low = str(text or "").lower()
    if not low:
        return None
    if re.search(r"\b(?:not|no)\s+(?:inverse|inversed|invert|inverted)\b", low):
        return False
    if re.search(r"\bpositive(?:\s+tone)?\s+(?:lithography|litho|pattern|exposure|development)\b", low):
        return False
    if re.search(r"\bnormal\s+(?:lithography|litho|pattern|exposure|development)\b", low):
        return False
    if re.search(r"\b(?:inverse|inversed|invert|inverted|reverse)\b", low):
        return True
    if re.search(r"\bnegative(?:\s+tone)?\s+(?:lithography|litho|pattern|exposure|development)\b", low):
        return True
    return None


def _extract_lithography_manual_windowing(text: str) -> bool:
    low = str(text or "").lower()
    if not low:
        return False
    litho_terms = r"(?:litho(?:graphy)?|window(?:s)?|opening(?:s)?|pattern)"
    if re.search(rf"\b(?:manual|custom|customi[sz]able|interactive|freehand|user[-\s]?defined)\b.*\b{litho_terms}\b", low):
        return True
    if re.search(rf"\b{litho_terms}\b.*\b(?:manual|custom|customi[sz]able|interactive|freehand|user[-\s]?defined)\b", low):
        return True
    if re.search(rf"\bpause\b.*\b{litho_terms}\b|\b{litho_terms}\b.*\bpause\b", low):
        return True
    if re.search(
        rf"\b(?:let\s+me|i\s+will|i'll|i\s+want\s+to|user\s+will)\b.*\b(?:draw|define|choose|select|set|make)\b.*\b{litho_terms}\b",
        low,
    ):
        return True
    if re.search(r"\bspecific\s+distance\s+from\s+(?:the\s+)?edges?\b", low) and not _extract_lithography_windows(low):
        return True
    return False


def _order_commands_for_context(commands: List[Dict], context: Dict) -> List[Dict]:
    if not isinstance(commands, list):
        return []
    order = (context or {}).get("top_material_order", [])
    if not isinstance(order, list) or not order:
        return list(commands)
    rank = {str(material).lower(): idx for idx, material in enumerate(order)}

    def etch_rank(command):
        material = str(command.get("material", "") or "").lower()
        return rank.get(material, len(rank) + 1000)

    ordered = []
    group = []

    def flush_group():
        if group:
            ordered.extend(sorted(group, key=etch_rank))
            group.clear()

    for command in commands:
        if (
            isinstance(command, dict)
            and command.get("action") == "operation"
            and command.get("operation") == "Etch"
        ):
            group.append(command)
            continue
        flush_group()
        ordered.append(command)
    flush_group()
    return ordered


def _material_short_name(material: str) -> str:
    txt = str(material or "").strip()
    if "(" in txt and ")" in txt:
        start = txt.rfind("(")
        end = txt.rfind(")")
        if 0 <= start < end:
            return txt[start + 1:end].strip()
    return ""


def _canonical_material(value: str, materials: List[str]) -> str:
    txt = str(value or "").strip()
    if not txt:
        return ""
    for material in materials:
        if material.lower() == txt.lower():
            return material
    found = _find_material(txt, materials)
    if found:
        return found
    return _alias_material(txt, materials)


def _alias_material(value: str, materials: List[str]) -> str:
    txt = str(value or "").strip().lower()
    if not txt:
        return ""

    def pick(*targets):
        target_lows = [str(t).lower() for t in targets if str(t).strip()]
        for target in target_lows:
            for material in materials:
                if material.lower() == target:
                    return material
        for target in target_lows:
            for material in materials:
                if target in material.lower():
                    return material
        return ""

    if re.search(r"\b(sic|silicon carbide)\b", txt):
        return pick("Silicon Carbide (SiC)")
    if re.search(r"\b(aln|aluminum nitride|aluminium nitride|aluminum nitide|aluminium nitide)\b", txt):
        return pick("Aluminum Nitride (AlN)")
    if re.search(r"\b(tin|titanium nitride|titanium nitide)\b", txt):
        return pick("Titanium Nitride (TiN)")
    if re.search(r"\b(sio2|silica|silicon oxide|silicon dioxide|oxide)\b", txt):
        return pick("Silicon Dioxide (SiO2)", "Glass", "Quartz")
    if re.search(r"\b(si3n4|silicon nitride|silicon nitide|nitride|nitide)\b", txt):
        return pick("Silicon Nitride (Si3N4)")
    if re.search(r"\b(chrome|chromium|cr)\b", txt):
        return pick("Chromium")
    if re.search(r"\b(gold|au)\b", txt):
        return pick("Gold")
    if re.search(r"\b(aluminum|aluminium|al)\b", txt):
        return pick("Aluminum")
    if re.search(r"\b(titanium|ti)\b", txt):
        return pick("Titanium")
    if re.search(r"\b(nickel|ni)\b", txt):
        return pick("Nickel")
    if re.search(r"\b(copper|cu)\b", txt):
        return pick("Copper")
    if re.search(r"\b(platinum|pt)\b", txt):
        return pick("Platinum")
    if re.search(r"\b(tungsten|w)\b", txt):
        return pick("Tungsten")
    if re.search(r"\b(silicon|si)\b", txt) and "carbide" not in txt and "nitride" not in txt and "nitide" not in txt and "oxide" not in txt:
        return pick("Silicon (Si)")
    if re.search(r"\b(pdms)\b", txt):
        return pick("Polydimethylsiloxane (PDMS)")
    if re.search(r"\b(mempex|mempax)\b", txt):
        return pick("MEMpax", "Glass")
    if re.search(r"\b(pmma)\b", txt):
        return pick("PMMA (Poly(methyl methacrylate))")
    if re.search(r"\b(su[-\s]?8)\b", txt):
        return pick("PR_SU-8 Photoresist", "PR_Negative Photoresist (SU-8, HSQ, AZ nLOF series)", "Photoresist")
    if re.search(r"\b(photo\s*resist|photoresist|resist|pr material|photoresist material)\b", txt):
        return pick("Photoresist", "PR_SU-8 Photoresist")
    return ""


def _canonical_operation(value: str) -> str:
    txt = str(value or "").strip()
    if not txt:
        return ""
    aliases = {
        "clean": "Clean / Surface prep",
        "surface prep": "Clean / Surface prep",
        "surface preparation": "Clean / Surface prep",
        "strip": "Strip material",
        "lift off": "Lift-off",
        "liftoff": "Lift-off",
        "bake": "Bake/Annealing",
        "anneal": "Bake/Annealing",
        "annealing": "Bake/Annealing",
        "spin coat": "Spincoating",
        "spin-coat": "Spincoating",
        "spincoat": "Spincoating",
        "spin coating": "Spincoating",
        "sputter": "Deposit",
        "sputtering": "Deposit",
        "evaporate": "Deposit",
        "evaporation": "Deposit",
        "thermal evaporation": "Deposit",
        "thermal deposition": "Deposit",
        "rie": "Etch",
        "drie": "Etch",
        "wet etch": "Etch",
        "wet etching": "Etch",
        "dry etch": "Etch",
        "dry etching": "Etch",
        "litho": "Lithography",
    }
    low = txt.lower()
    if low in aliases:
        return aliases[low]
    for op in ALLOWED_OPERATIONS:
        if op.lower() == low:
            return op
    return ""


def _sanitize_windows(value) -> List[Tuple[float, float]]:
    windows = []
    if not isinstance(value, list):
        return windows
    for item in value:
        if not isinstance(item, (list, tuple)) or len(item) < 2:
            continue
        start = _coerce_float(item[0])
        end = _coerce_float(item[1])
        if start is None or end is None:
            continue
        start = max(0.0, min(1.0, start))
        end = max(0.0, min(1.0, end))
        if end > start:
            windows.append((start, end))
    return windows


def _extract_windows(text: str) -> List[Tuple[float, float]]:
    windows = []
    text = _normalize_user_request(text)
    pattern = re.compile(
        r"(?:window|range|from)?\s*(0(?:\.\d+)?|1(?:\.0+)?)\s*(?:-|to|until|through)\s*(0(?:\.\d+)?|1(?:\.0+)?)",
        flags=re.IGNORECASE,
    )
    for match in pattern.finditer(text or ""):
        start = _coerce_float(match.group(1))
        end = _coerce_float(match.group(2))
        if start is None or end is None:
            continue
        if end > start:
            windows.append((start, end))
    return windows


def _extract_lithography_windows(text: str) -> List[Tuple[float, float]]:
    text = _normalize_user_request(text)
    windows = _extract_windows(text)
    windows.extend(_extract_edge_windows(text))
    if not windows or re.search(r"\b(center|central|centre|middle)\b", text, re.IGNORECASE):
        windows.extend(_extract_center_window(text))
    windows.extend(_extract_side_windows(text))
    return _unique_windows(windows)


def _extract_center_window(text: str) -> List[Tuple[float, float]]:
    low = _normalize_user_request(text).lower()
    center_match = re.search(r"\b(center|central|centre|middle)\b", low)
    if not center_match:
        if any(k in low for k in ("left", "right", "edge", "edges")):
            return []
        if not any(k in low for k in ("window", "range", "opening", "litho", "lithography")):
            return []
        width = _extract_window_value(low)
    else:
        before = low[:center_match.start()]
        after = low[center_match.end():]
        start = 0
        for side in (" left ", " right "):
            index = before.rfind(side)
            if index >= 0:
                start = max(start, index + len(side))
        end = len(low)
        after_side_indexes = [
            after.find(side)
            for side in (" left ", " right ")
            if after.find(side) >= 0
        ]
        if after_side_indexes:
            end = center_match.end() + min(after_side_indexes)
        width = _extract_window_value(low[start:end])
    if width is None:
        return []
    width = max(0.0, min(1.0, float(width)))
    start = max(0.0, 0.5 - width / 2.0)
    end = min(1.0, 0.5 + width / 2.0)
    if end > start:
        return [(start, end)]
    return []


def _extract_edge_windows(text: str) -> List[Tuple[float, float]]:
    low = _normalize_user_request(text).lower()
    if not re.search(r"\b(edge|edges)\b", low):
        return []
    if re.search(r"\b(left|right)\b", low):
        return []
    if not any(k in low for k in ("window", "windows", "opening", "openings", "litho", "lithography", "pattern")):
        return []
    if not re.search(r"\b(two|both|each|edges)\b", low):
        return []
    numbers = [
        _coerce_float(match.group(1))
        for match in re.finditer(r"\b(0(?:\.\d+)?|1(?:\.0+)?)\b", low)
    ]
    numbers = [value for value in numbers if value is not None and 0.0 < value <= 1.0]
    if not numbers:
        return []
    width = max(0.0, min(1.0, float(numbers[-1])))
    if width <= 0.0 or width >= 0.5:
        return []
    return [(0.0, width), (1.0 - width, 1.0)]


def _extract_side_windows(text: str) -> List[Tuple[float, float]]:
    low = _normalize_user_request(text).lower()
    windows = []
    side_pattern = re.compile(
        rf"\b(left|right)\b(?P<body>.{{0,90}}?)(?=(?:\b(?:left|right|center|central|centre|middle)\b)|$)",
        re.IGNORECASE,
    )
    for match in side_pattern.finditer(low):
        side = match.group(1).lower()
        body = match.group("body")
        numbers = [
            _coerce_float(num.group(1))
            for num in re.finditer(r"\b(0(?:\.\d+)?|1(?:\.0+)?)\b", body)
        ]
        numbers = [n for n in numbers if n is not None and 0.0 < n <= 1.0]
        if not numbers:
            continue
        width = numbers[0]
        margin = numbers[1] if len(numbers) > 1 else 0.0
        if side == "right":
            start = max(0.0, 1.0 - margin - width)
            end = min(1.0, 1.0 - margin)
        else:
            start = max(0.0, margin)
            end = min(1.0, margin + width)
        if end > start:
            windows.append((start, end))
    return windows


def _unique_windows(windows: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    out = []
    seen = set()
    for start, end in windows:
        start = max(0.0, min(1.0, float(start)))
        end = max(0.0, min(1.0, float(end)))
        if end <= start:
            continue
        key = (round(start, 9), round(end, 9))
        if key in seen:
            continue
        seen.add(key)
        out.append((start, end))
    return out


def _extract_window_value(text: str):
    low = _normalize_user_request(text).lower()
    numbers = []
    for match in re.finditer(r"\b(0(?:\.\d+)?|1(?:\.0+)?)\b", low):
        value = _coerce_float(match.group(1))
        if value is not None and 0.0 < value <= 1.0:
            numbers.append(value)
    if numbers:
        return numbers[-1]
    return None


def _command_measure_was_requested(command: Dict, value: float, user_request: str, all_measures: List[float], context: Dict = None) -> bool:
    if not _measure_was_requested(value, all_measures):
        if str(command.get("operation", "") or "") == "Etch":
            return _semantic_etch_depth_was_requested(command, value, user_request, context or {})
        if str(command.get("operation", "") or "") == "Bond":
            return _semantic_substrate_thickness_was_requested(command, value, user_request, context or {})
        return False

    clauses = [
        c.strip()
        for c in re.split(r"\bthen\b|\bafter that\b|;|\n", user_request or "", flags=re.IGNORECASE)
        if c.strip()
    ]
    if len(clauses) <= 1:
        return True

    op = str(command.get("operation", "") or "").lower()
    material = str(command.get("material", "") or "").lower()
    material_head = material.split("(", 1)[0].strip()
    op_terms = {
        "deposit": ("deposit", "deposition", "evaporate", "sputter", "ald", "cvd"),
        "spincoating": ("spin", "spincoat", "spincoating"),
        "bond": ("bond",),
        "etch": ("etch", "rie", "drie"),
    }
    terms = op_terms.get(op, (op,))

    relevant = []
    for clause in clauses:
        low = clause.lower()
        if any(term and term in low for term in terms):
            relevant.append(clause)
            continue
        if material and material in low:
            relevant.append(clause)
            continue
        if material_head and material_head in low:
            relevant.append(clause)

    if not relevant:
        return True
    return any(
        _measure_was_requested(value, _extract_all_measure_um(clause))
        or _measure_was_requested(value, [_extract_substrate_like_unitless_um(clause)])
        for clause in relevant
    )


def _semantic_etch_depth_was_requested(command: Dict, value: float, user_request: str, context: Dict) -> bool:
    material = str(command.get("material", "") or "").strip()
    if not material:
        return False
    known = _unique([material] + list(_iter_materials(context)))
    thicknesses = _context_material_thickness_um(context, known)
    layer_thickness = thicknesses.get(material)
    if layer_thickness is None or layer_thickness <= 0.0:
        return False

    mentions = _find_material_mentions(user_request, [material])
    if not mentions:
        return False
    for start, end, _mat in mentions:
        segment = _segment_for_material(user_request, start, end)
        fraction = _etch_fraction_for_segment(segment, user_request)
        if fraction is None:
            continue
        if _measure_was_requested(value, [layer_thickness * fraction]):
            return True
    return False


def _semantic_substrate_thickness_was_requested(command: Dict, value: float, user_request: str, context: Dict) -> bool:
    material = str(command.get("material", "") or "").strip()
    known = _unique([material] + list(_iter_materials(context)))
    mentions = _find_material_mentions(user_request, [material]) if material else []
    if not mentions:
        candidate = _extract_substrate_like_unitless_um(user_request)
        return _measure_was_requested(value, [candidate])
    for start, end, _mat in mentions:
        segment = _segment_for_material(user_request, start, end)
        if _measure_was_requested(value, [_extract_substrate_like_unitless_um(segment)]):
            return True
        if _measure_was_requested(value, [_extract_substrate_like_unitless_um(user_request)]):
            return True
    return False
