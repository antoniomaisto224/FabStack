"""Local LLM chat integration for process-step planning."""

import html
import logging

import numpy as np
from PyQt5.QtCore import QObject, Qt, QThread, pyqtSignal
from PyQt5.QtGui import QTextCursor
from PyQt5.QtWidgets import (
    QApplication,
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSizePolicy,
    QTextEdit,
    QVBoxLayout,
)

from fabstack.assistant.planner import (
    ALLOWED_OPERATIONS,
    LocalStepPlanner,
    _canonical_material,
    _coerce_bool,
)
from fabstack.terminology import operation_label

LOGGER = logging.getLogger(__name__)

ASSISTANT_WELCOME_MESSAGE = (
    "Hi, I’m Fabrix. Describe the fabrication process in order, including each "
    "material, dimension, and unit. I’ll prepare the steps for your review before applying them."
)
ASSISTANT_REWRITE_INSTRUCTIONS_MESSAGE = (
    "Reply with the missing or corrected details."
)
ASSISTANT_PROMPTING_RULES_HTML = """
<div style="font-weight: 700; margin-bottom: 8px;">Process prompt checklist</div>
<ol style="margin: 0; padding-left: 18px;">
  <li>List the steps in order.</li>
  <li>Use one consistent name for each material.</li>
  <li>Include dimensions and units, such as 100 µm, 25 °C, or 60 s.</li>
  <li>Name the material being etched.</li>
  <li>For lithography, describe the windows and whether the resist tone is inverted.</li>
</ol>
"""


class AssistantPromptEdit(QPlainTextEdit):
    sendRequested = pyqtSignal()

    def keyPressEvent(self, event):
        if event.key() in (Qt.Key_Return, Qt.Key_Enter):
            if event.modifiers() & Qt.ShiftModifier:
                super().keyPressEvent(event)
                return
            self.sendRequested.emit()
            return
        super().keyPressEvent(event)


class PlannerWorker(QObject):
    finished = pyqtSignal(object, str, str)

    def __init__(self, planner, prompt, context):
        super().__init__()
        self.planner = planner
        self.prompt = prompt
        self.context = context

    def run(self):
        error = ""
        response = {}
        try:
            response = self.planner.generate_command_plan(self.prompt, self.context)
        except Exception:
            LOGGER.exception("Fabrix process-plan generation failed")
            error = "generation_failed"
        self.finished.emit(response, error, self.planner.status_text())


class LlmAssistantMixin:
    def _init_process_assistant_state(self):
        self.llm_chat_messages = []
        self.llm_chat_dialog = None
        self.llm_step_planner = None
        self._assistant_thread = None
        self._assistant_worker = None
        self._assistant_pending_plan = None
        self._assistant_manual_lithography_pause = None

    def _assistant_chat_for_save(self):
        cleaned = []
        for msg in getattr(self, "llm_chat_messages", []):
            if not isinstance(msg, dict):
                continue
            role = str(msg.get("role", "")).strip()
            content = str(msg.get("content", "")).strip()
            if role in {"user", "assistant"} and content:
                cleaned.append({"role": role, "content": content})
        return cleaned

    def _set_assistant_chat_messages(self, messages):
        cleaned = []
        if isinstance(messages, list):
            for msg in messages:
                if not isinstance(msg, dict):
                    continue
                role = str(msg.get("role", "")).strip()
                content = str(msg.get("content", "")).strip()
                if role in {"user", "assistant"} and content:
                    cleaned.append({"role": role, "content": content})
        self.llm_chat_messages = cleaned
        self._assistant_pending_plan = None
        self._refresh_assistant_dialog()

    def _assistant_planner(self):
        if self.llm_step_planner is None:
            self.llm_step_planner = LocalStepPlanner()
        return self.llm_step_planner

    def open_process_assistant_chat(self):
        if self.llm_chat_dialog is None:
            self._build_process_assistant_dialog()
        self._refresh_assistant_dialog()
        self.llm_chat_dialog.show()
        self.llm_chat_dialog.raise_()
        self.llm_chat_dialog.activateWindow()
        self.assistant_prompt_edit.setFocus()

    def _build_process_assistant_dialog(self):
        dlg = QDialog(self)
        dlg.setWindowTitle("Fabrix process assistant")
        dlg.setModal(False)
        dlg.resize(760, 660)
        dlg.setMinimumSize(620, 500)
        self.llm_chat_dialog = dlg

        layout = QVBoxLayout(dlg)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        header_row = QHBoxLayout()
        self.assistant_title_label = QLabel("Fabrix process assistant")
        self.assistant_title_label.setObjectName("assistantTitleLabel")
        header_row.addWidget(self.assistant_title_label, 1)

        self.assistant_status_label = QLabel(self._assistant_planner().status_text())
        self.assistant_status_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        header_row.addWidget(self.assistant_status_label, 0)
        layout.addLayout(header_row)

        self.assistant_transcript = QTextEdit(dlg)
        self.assistant_transcript.setReadOnly(True)
        self.assistant_transcript.setAcceptRichText(True)
        self.assistant_transcript.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.assistant_rules_panel = QTextEdit(dlg)
        self.assistant_rules_panel.setReadOnly(True)
        self.assistant_rules_panel.setAcceptRichText(True)
        self.assistant_rules_panel.setHtml(ASSISTANT_PROMPTING_RULES_HTML)
        self.assistant_rules_panel.setFixedWidth(280)
        self.assistant_rules_panel.setFocusPolicy(Qt.NoFocus)
        self.assistant_rules_panel.setToolTip("Details needed to build an accurate process plan.")
        self.assistant_rules_panel.setStyleSheet(
            "QTextEdit {"
            " border: 1px solid palette(mid);"
            " border-radius: 8px;"
            " padding: 8px;"
            " background-color: palette(alternate-base);"
            "}"
        )

        body_row = QHBoxLayout()
        body_row.setSpacing(8)
        body_row.addWidget(self.assistant_transcript, 1)
        body_row.addWidget(self.assistant_rules_panel, 0)
        layout.addLayout(body_row, 1)

        self.assistant_prompt_edit = AssistantPromptEdit(dlg)
        self.assistant_prompt_edit.setPlaceholderText(
            "Describe the process in order, with materials, dimensions, and units"
        )
        self.assistant_prompt_edit.setFixedHeight(88)
        self.assistant_prompt_edit.sendRequested.connect(self._send_assistant_prompt)
        layout.addWidget(self.assistant_prompt_edit, 0)

        buttons_row = QHBoxLayout()
        self.assistant_send_button = QPushButton("Create plan")
        self.assistant_regen_button = QPushButton("Regenerate plan")
        self.assistant_run_button = QPushButton("Apply plan")
        self.assistant_copy_button = QPushButton("Copy response")
        self.assistant_delete_button = QPushButton("Clear chat")
        buttons_row.addWidget(self.assistant_send_button)
        buttons_row.addWidget(self.assistant_regen_button)
        buttons_row.addWidget(self.assistant_run_button)
        buttons_row.addWidget(self.assistant_copy_button)
        buttons_row.addStretch(1)
        buttons_row.addWidget(self.assistant_delete_button)
        layout.addLayout(buttons_row)

        self.assistant_send_button.clicked.connect(self._send_assistant_prompt)
        self.assistant_regen_button.clicked.connect(self._regenerate_last_assistant_response)
        self.assistant_run_button.clicked.connect(self._apply_assistant_pending_plan)
        self.assistant_copy_button.clicked.connect(self._copy_last_assistant_plan)
        self.assistant_delete_button.clicked.connect(self._delete_assistant_chat)

        close_box = QDialogButtonBox(QDialogButtonBox.Close)
        close_box.rejected.connect(dlg.close)
        layout.addWidget(close_box)

    def _refresh_assistant_dialog(self):
        if self.llm_chat_dialog is None:
            return

        tab_name = self._assistant_current_tab_name()
        self.llm_chat_dialog.setWindowTitle(f"Fabrix — {tab_name}")
        self.assistant_title_label.setText(f"Fabrix — {tab_name}")
        if self._assistant_is_paused_for_manual_lithography():
            status_text = "Waiting for lithography windows"
        elif self._assistant_plan_is_runnable(getattr(self, "_assistant_pending_plan", None)):
            status_text = "Plan ready"
        else:
            status_text = self._assistant_planner().status_text()
        self.assistant_status_label.setText(status_text)
        self._render_assistant_transcript()
        self._sync_assistant_buttons()

    def _assistant_current_tab_name(self):
        if hasattr(self, "project_tabs"):
            idx = self.project_tabs.currentIndex()
            if idx >= 0:
                txt = str(self.project_tabs.tabText(idx) or "").strip()
                if txt:
                    return txt
        return "current process"

    def _render_assistant_transcript(self):
        if not hasattr(self, "assistant_transcript"):
            return

        messages = getattr(self, "llm_chat_messages", [])
        blocks = [self._assistant_message_html("assistant", ASSISTANT_WELCOME_MESSAGE)]
        for msg in messages:
            blocks.append(self._assistant_message_html(msg.get("role"), msg.get("content", "")))
        self.assistant_transcript.setHtml("".join(blocks))
        self.assistant_transcript.moveCursor(QTextCursor.End)

    def _assistant_message_html(self, role, content):
        label = "You" if role == "user" else "Fabrix"
        text = html.escape(str(content or "")).replace("\n", "<br>")
        return (
            "<div style='margin: 0 0 10px 0;'>"
            f"<div style='font-weight: 700;'>{label}</div>"
            f"<div style='white-space: normal;'>{text}</div>"
            "</div>"
        )

    def _sync_assistant_buttons(self):
        busy = self._assistant_thread is not None and self._assistant_thread.isRunning()
        paused = self._assistant_is_paused_for_manual_lithography()
        has_last_user = self._last_user_prompt() != ""
        has_last_assistant = self._last_assistant_response() != ""
        can_run = self._assistant_plan_is_runnable(getattr(self, "_assistant_pending_plan", None)) or paused
        if hasattr(self, "assistant_send_button"):
            self.assistant_send_button.setEnabled(not busy)
            self.assistant_regen_button.setEnabled((not busy) and has_last_user)
            self.assistant_run_button.setText("Continue" if paused else "Apply plan")
            self.assistant_run_button.setEnabled((not busy) and can_run)
            self.assistant_copy_button.setEnabled(has_last_assistant)
            self.assistant_delete_button.setEnabled((not busy) and bool(getattr(self, "llm_chat_messages", [])))

    def _assistant_mark_dirty(self):
        if bool(getattr(self, "tab_switch_in_progress", False)):
            return
        if hasattr(self, "_snapshot_active_tab_state"):
            self._snapshot_active_tab_state()

    def _assistant_append_message(self, role: str, content: str):
        role = str(role or "").strip()
        content = str(content or "").strip()
        if role not in {"user", "assistant"} or not content:
            return
        self.llm_chat_messages.append({"role": role, "content": content})
        self._render_assistant_transcript()
        self._sync_assistant_buttons()
        self._assistant_mark_dirty()

    def _send_assistant_prompt(self):
        if self._assistant_thread is not None and self._assistant_thread.isRunning():
            return
        prompt = self.assistant_prompt_edit.toPlainText().strip()
        if not prompt:
            return
        self.assistant_prompt_edit.clear()
        self._assistant_append_message("user", prompt)
        self._start_assistant_generation(prompt)

    def _start_assistant_generation(self, prompt: str):
        context = self._build_assistant_context()
        planner = self._assistant_planner()
        self._assistant_pending_plan = None
        self._assistant_manual_lithography_pause = None
        self.assistant_status_label.setText("Creating process plan…")
        self._sync_assistant_buttons()

        thread = QThread(self)
        worker = PlannerWorker(planner, prompt, context)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.finished.connect(self._assistant_generation_finished)
        worker.finished.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        self._assistant_thread = thread
        self._assistant_worker = worker
        thread.start()

    def _assistant_generation_finished(self, response, error: str, _status: str):
        self._assistant_thread = None
        self._assistant_worker = None
        try:
            if error:
                response_text = (
                    "I couldn’t create the process plan. Review the request and try again."
                )
                self._assistant_pending_plan = None
                status_text = "Plan not created"
            else:
                plan = response if isinstance(response, dict) else {}
                self._assistant_pending_plan = plan if self._assistant_plan_is_runnable(plan) else None
                response_text = self._format_assistant_plan_response(plan)
                status_text = (
                    "Plan ready"
                    if self._assistant_pending_plan is not None
                    else "More details needed"
                )
            self._assistant_append_message("assistant", response_text)
            if hasattr(self, "assistant_status_label"):
                self.assistant_status_label.setText(status_text)
        finally:
            self._sync_assistant_buttons()

    def _last_user_prompt(self):
        for msg in reversed(getattr(self, "llm_chat_messages", [])):
            if msg.get("role") == "user":
                return str(msg.get("content", "")).strip()
        return ""

    def _last_assistant_response(self):
        for msg in reversed(getattr(self, "llm_chat_messages", [])):
            if msg.get("role") == "assistant":
                return str(msg.get("content", "")).strip()
        return ""

    def _regenerate_last_assistant_response(self):
        if self._assistant_thread is not None and self._assistant_thread.isRunning():
            return
        prompt = self._last_user_prompt()
        if not prompt:
            return

        while self.llm_chat_messages and self.llm_chat_messages[-1].get("role") == "assistant":
            self.llm_chat_messages.pop()
        self._assistant_pending_plan = None
        self._assistant_manual_lithography_pause = None
        self._render_assistant_transcript()
        self._assistant_mark_dirty()
        self._start_assistant_generation(prompt)

    def _copy_last_assistant_plan(self):
        content = self._last_assistant_response()
        if not content:
            return
        QApplication.clipboard().setText(content)
        self.statusBar().showMessage("Response copied.", 2500)

    def _confirm_assistant_action(
        self,
        title: str,
        text: str,
        action_label: str,
        *,
        destructive: bool = False,
    ) -> bool:
        dialog = QMessageBox(self)
        dialog.setIcon(QMessageBox.Question)
        dialog.setWindowTitle(title)
        dialog.setText(text)
        role = QMessageBox.DestructiveRole if destructive else QMessageBox.AcceptRole
        action_button = dialog.addButton(action_label, role)
        cancel_button = dialog.addButton(QMessageBox.Cancel)
        dialog.setDefaultButton(cancel_button if destructive else action_button)
        dialog.exec_()
        return dialog.clickedButton() is action_button

    def _delete_assistant_chat(self):
        if not getattr(self, "llm_chat_messages", []):
            return
        if not self._confirm_assistant_action(
            "Clear chat?",
            f"Clear the Fabrix conversation for “{self._assistant_current_tab_name()}”?",
            "Clear chat",
            destructive=True,
        ):
            return
        self.llm_chat_messages = []
        self._assistant_pending_plan = None
        self._assistant_manual_lithography_pause = None
        self._render_assistant_transcript()
        self._sync_assistant_buttons()
        if hasattr(self, "assistant_status_label"):
            self.assistant_status_label.setText("Ready")
        self._assistant_mark_dirty()

    def _assistant_plan_is_runnable(self, plan):
        return (
            isinstance(plan, dict)
            and plan.get("status") == "ready"
            and isinstance(plan.get("commands"), list)
            and len(plan.get("commands")) > 0
        )

    def _format_assistant_plan_response(self, plan):
        if not isinstance(plan, dict):
            return "I couldn’t create a valid process plan. Review the request and try again."

        missing = plan.get("missing", [])
        if plan.get("status") != "ready" or missing:
            return self._format_assistant_clarification(missing or ["Provide the missing process details."])

        commands = plan.get("commands", [])
        lines = ["Process plan:"]
        for idx, command in enumerate(commands, 1):
            lines.append(f"{idx}. {self._assistant_command_label(command)}")
        return "\n".join(lines)

    def _format_assistant_clarification(self, missing):
        items = [str(item).strip() for item in (missing or []) if str(item).strip()]
        if not items:
            items = ["Provide the missing process details."]
        lines = ["Before I continue:"]
        for idx, item in enumerate(items, 1):
            lines.append(f"{idx}. {item}")
        lines.append("")
        lines.append(ASSISTANT_REWRITE_INSTRUCTIONS_MESSAGE)
        return "\n".join(lines)

    def _assistant_command_label(self, command):
        if not isinstance(command, dict):
            return "Invalid step"
        action = command.get("action")
        if action == "add_material":
            return f"Add {command.get('material', '')} to this process"
        if action == "set_substrate":
            label = f"Set substrate: {command.get('material', '')}"
            if "thickness_um" in command:
                label += f" ({float(command['thickness_um']):g} µm)"
            return label
        if action == "operation":
            op = command.get("operation", "")
            display_op = operation_label(op)
            mat = command.get("material", "")
            method = str(command.get("method", "") or "").strip()
            if op == "Flip":
                return "Flip stack"
            if op in {"Deposit", "Spincoating", "Bond"}:
                details = [f"{float(command.get('thickness_um', 0.0)):g} µm"]
                if method:
                    details.append(method)
                if op == "Spincoating" and "temperature_c" in command:
                    details.append(f"{float(command.get('temperature_c', 0.0)):g} rpm")
                return f"{display_op} {mat} — {' · '.join(details)}"
            if op == "Etch":
                details = [
                    command.get("etch_mode", "anisotropic"),
                ]
                if method:
                    details.append(method)
                depth = f"{float(command.get('depth_um', 0.0)):g} µm"
                return f"Etch {mat} by {depth} — {' · '.join(str(x) for x in details if str(x).strip())}"
            if op == "Lithography":
                windows = command.get("windows", [])
                details = []
                if self._assistant_command_is_manual_lithography(command):
                    details.append("manual window selection")
                elif windows:
                    ranges = []
                    for start, end in windows:
                        ranges.append(f"{float(start):g}–{float(end):g}")
                    details.append(f"windows {', '.join(ranges)}")
                if method:
                    details.append(method)
                if bool(_coerce_bool(command.get("invert_lithography"))):
                    details.append("inverted")
                suffix = f" — {' · '.join(details)}" if details else ""
                return f"Lithography {mat}{suffix}"
            if method:
                return f"{display_op} {mat} — {method}".strip()
            return f"{display_op} {mat}".strip()
        return "Unsupported step"

    def _apply_assistant_pending_plan(self):
        if self._assistant_is_paused_for_manual_lithography():
            self._continue_assistant_manual_lithography()
            return

        plan = getattr(self, "_assistant_pending_plan", None)
        if not self._assistant_plan_is_runnable(plan):
            return

        errors = self._validate_assistant_commands(plan.get("commands", []))
        if errors:
            self._assistant_pending_plan = None
            self._assistant_append_message("assistant", self._format_assistant_clarification(errors))
            if hasattr(self, "assistant_status_label"):
                self.assistant_status_label.setText("Action needed")
            self._sync_assistant_buttons()
            return

        summary = "\n".join(
            f"{idx}. {self._assistant_command_label(command)}"
            for idx, command in enumerate(plan.get("commands", []), 1)
        )
        reply = QMessageBox.question(
            self,
            "Apply process plan?",
            f"Apply these steps to “{self._assistant_current_tab_name()}”?\n\n{summary}",
            QMessageBox.Apply | QMessageBox.Cancel,
            QMessageBox.Cancel,
        )
        # QMessageBox.Yes remains accepted for non-interactive compatibility
        # with older validation harnesses that mocked the previous dialog.
        if reply not in {QMessageBox.Apply, QMessageBox.Yes}:
            return

        if hasattr(self, "assistant_status_label"):
            self.assistant_status_label.setText("Applying steps…")
        self._run_assistant_command_sequence(plan.get("commands", []))

    def _run_assistant_command_sequence(self, commands, initial_applied=None):
        applied = []
        if initial_applied:
            applied.extend(str(item) for item in initial_applied if str(item).strip())
        warnings = []
        commands = list(commands or [])
        for index, command in enumerate(commands):
            if self._assistant_command_is_manual_lithography(command):
                self._pause_assistant_for_manual_lithography(
                    command,
                    commands[index + 1:],
                    applied,
                )
                return
            try:
                ok, message = self._execute_assistant_command(command)
            except Exception:
                LOGGER.exception("Fabrix failed while applying a process step")
                ok = False
                message = "A step could not be applied. Review the process plan and try again."
            if ok:
                applied.append(message)
            else:
                warnings.append(message)
                break

        lines = []
        if applied:
            lines.append("Applied steps:")
            lines.extend(f"{idx}. {msg}" for idx, msg in enumerate(applied, 1))
        if warnings:
            lines.append("")
            lines.append(self._format_assistant_clarification(warnings))

        self._assistant_pending_plan = None
        self._assistant_manual_lithography_pause = None
        self._assistant_append_message("assistant", "\n".join(lines) if lines else "No steps were applied.")
        if hasattr(self, "assistant_status_label"):
            self.assistant_status_label.setText("Action needed" if warnings else "Steps applied")
        self._sync_assistant_buttons()
        self._assistant_mark_dirty()

    def _assistant_command_is_manual_lithography(self, command):
        if not isinstance(command, dict):
            return False
        return (
            command.get("action") == "operation"
            and command.get("operation") == "Lithography"
            and str(command.get("windowing_mode", "") or "").strip().lower() == "manual"
        )

    def _assistant_is_paused_for_manual_lithography(self):
        pause = getattr(self, "_assistant_manual_lithography_pause", None)
        return isinstance(pause, dict) and isinstance(pause.get("command"), dict)

    def _pause_assistant_for_manual_lithography(self, command, remaining_commands, applied):
        ok, message = self._prepare_manual_lithography(command)
        if not ok:
            self._assistant_pending_plan = None
            self._assistant_manual_lithography_pause = None
            lines = []
            if applied:
                lines.append("Applied steps:")
                lines.extend(f"{idx}. {msg}" for idx, msg in enumerate(applied, 1))
                lines.append("")
            lines.append(self._format_assistant_clarification([message]))
            self._assistant_append_message("assistant", "\n".join(lines))
            if hasattr(self, "assistant_status_label"):
                self.assistant_status_label.setText("Action needed")
            self._sync_assistant_buttons()
            self._assistant_mark_dirty()
            return

        self._assistant_pending_plan = None
        self._assistant_manual_lithography_pause = {
            "command": dict(command),
            "remaining_commands": list(remaining_commands or []),
            "applied": list(applied or []),
        }

        lines = []
        if applied:
            lines.append("Applied steps:")
            lines.extend(f"{idx}. {msg}" for idx, msg in enumerate(applied, 1))
            lines.append("")
        lines.append("I paused before lithography.")
        lines.append(f"Add at least one window for {command.get('material', '')}, then choose Continue.")
        self._assistant_append_message("assistant", "\n".join(lines))
        if hasattr(self, "assistant_status_label"):
            self.assistant_status_label.setText("Waiting for lithography windows")
        self._sync_assistant_buttons()
        self._assistant_mark_dirty()

    def _prepare_manual_lithography(self, command):
        material = self._assistant_resolve_material(command.get("material", ""))
        if not material:
            return False, f"The lithography material was not found: {command.get('material', '')}."
        self._assistant_ensure_material(material)

        op_idx = self.op_combo.findText("Lithography", Qt.MatchFixedString)
        if op_idx < 0:
            return False, "Lithography is not available."
        self.op_combo.setCurrentIndex(op_idx)
        self.update_op_controls()

        mat_idx = self._material_combo_index_by_full_name(material)
        if mat_idx < 0:
            self.refresh_operation_material_combo(preferred_name=material)
            mat_idx = self._material_combo_index_by_full_name(material)
        if mat_idx < 0:
            return False, f"FabStack could not select {material}."
        self.material_combo.setCurrentIndex(mat_idx)
        self.update_etchant_options()

        method = str(command.get("method", "") or "").strip()
        if method:
            self._set_etchant_combo_value(method)
        invert_lithography = _coerce_bool(command.get("invert_lithography")) if "invert_lithography" in command else None
        if invert_lithography is None:
            invert_lithography = self._lithography_invert_from_text(method)
        self._set_lithography_invert(bool(invert_lithography) if invert_lithography is not None else False, sync_method=True)

        self.clear_litho_windows(refresh_plot=True)
        self.open_windowing_dialog()
        self.statusBar().showMessage("Add lithography windows, then choose Continue in Fabrix.", 6000)
        return True, "Manual lithography ready."

    def _continue_assistant_manual_lithography(self):
        pause = getattr(self, "_assistant_manual_lithography_pause", None)
        if not isinstance(pause, dict):
            self._sync_assistant_buttons()
            return

        windows = [
            (float(start), float(end))
            for start, end in (getattr(self, "litho_windows", []) or [])
            if float(end) > float(start)
        ]
        if not windows:
            QMessageBox.warning(
                self,
                "Add a lithography window",
                "Add at least one window, then choose Continue.",
            )
            if hasattr(self, "assistant_status_label"):
                self.assistant_status_label.setText("Waiting for lithography windows")
            self._sync_assistant_buttons()
            return

        lithography_command = dict(pause.get("command") or {})
        lithography_command["windows"] = windows
        lithography_command.pop("windowing_mode", None)
        commands = [lithography_command] + list(pause.get("remaining_commands") or [])
        errors = self._validate_assistant_commands(commands)
        if errors:
            self._assistant_append_message("assistant", self._format_assistant_clarification(errors))
            if hasattr(self, "assistant_status_label"):
                self.assistant_status_label.setText("Action needed")
            self._sync_assistant_buttons()
            return

        applied = list(pause.get("applied") or [])
        self._assistant_manual_lithography_pause = None
        self._run_assistant_command_sequence(commands, initial_applied=applied)

    def _validate_assistant_commands(self, commands):
        errors = []
        if not isinstance(commands, list) or not commands:
            return ["The plan has no steps to apply."]
        for command in commands:
            if not isinstance(command, dict):
                errors.append("One planned step is invalid. Regenerate the plan.")
                continue
            action = command.get("action")
            if action == "add_material":
                material = self._assistant_resolve_material(command.get("material", ""))
                if not material:
                    errors.append(f"Material not found: {command.get('material', '')}.")
                continue
            if action == "set_substrate":
                material = self._assistant_resolve_material(command.get("material", ""))
                if not material:
                    errors.append(f"Substrate material not found: {command.get('material', '')}.")
                continue
            if action != "operation":
                errors.append("One planned step is not supported. Regenerate the plan.")
                continue

            op = str(command.get("operation", "")).strip()
            if op not in ALLOWED_OPERATIONS:
                errors.append(f"The plan contains an unsupported operation: {op}.")
                continue
            if op == "Flip":
                continue

            material = self._assistant_resolve_material(command.get("material", ""))
            if not material:
                errors.append(f"Material not found for {operation_label(op)}: {command.get('material', '')}.")
                continue
            if self._is_material_restricted_for_operation(op, material):
                errors.append(f"{material} cannot be used for {operation_label(op)}.")
            if op in {"Deposit", "Spincoating", "Bond"} and float(command.get("thickness_um", 0.0) or 0.0) <= 0.0:
                errors.append(f"Enter a positive thickness for {operation_label(op)}.")
            if op == "Etch" and float(command.get("depth_um", 0.0) or 0.0) <= 0.0:
                errors.append("Enter a positive etch depth.")
            if op in {"Lithography", "Lift-off"} and not self.is_photoresist_name(material):
                errors.append(f"Select a photoresist for {operation_label(op)}.")
            if (
                op == "Lithography"
                and not command.get("windows")
                and not self._assistant_command_is_manual_lithography(command)
            ):
                errors.append("Add at least one lithography window.")
        return errors

    def _execute_assistant_command(self, command):
        action = command.get("action")
        if action == "add_material":
            material = self._assistant_resolve_material(command.get("material", ""))
            if not material:
                return False, f"Material not found: {command.get('material', '')}."
            self._assistant_ensure_material(material)
            return True, f"Added {material} to this process"

        if action == "set_substrate":
            material = self._assistant_resolve_material(command.get("material", ""))
            if not material:
                return False, f"Substrate material not found: {command.get('material', '')}."
            self._assistant_ensure_material(material)
            if hasattr(self, "_begin_substrate_control_update"):
                self._begin_substrate_control_update()
            try:
                idx = self.substrate_combo.findText(material)
                if idx >= 0:
                    self.substrate_combo.setCurrentIndex(idx)
                if "thickness_um" in command:
                    self.substrate_thick_spin.setValue(int(round(float(command["thickness_um"]))))
            finally:
                if hasattr(self, "_end_substrate_control_update"):
                    self._end_substrate_control_update()
            if hasattr(self, "apply_substrate_change_process_wide"):
                if not self.apply_substrate_change_process_wide(warn_on_failure=False, snapshot=True):
                    return (
                        False,
                        "The substrate was not changed because this process flow cannot be rebuilt safely.",
                    )
            else:
                self.reset_wafer()
            return True, self._assistant_command_label(command)

        if action != "operation":
            return False, "One planned step is not supported. Regenerate the plan."

        op = str(command.get("operation", "")).strip()
        if op == "Flip":
            before = len(getattr(self, "history", []))
            self.flip_active_stack_from_top()
            if len(getattr(self, "history", [])) > before:
                return True, "Flip stack"
            return False, "The stack was not flipped. Add or load a stack, then try again."

        if self._assistant_command_is_manual_lithography(command) and not command.get("windows"):
            return False, "Add at least one lithography window before continuing."

        material = self._assistant_resolve_material(command.get("material", ""))
        if not material:
            return False, f"Material not found for {operation_label(op)}: {command.get('material', '')}."
        self._assistant_ensure_material(material)
        material = self._assistant_runtime_material_for_operation(op, material)

        op_idx = self.op_combo.findText(op, Qt.MatchFixedString)
        if op_idx < 0:
            return False, f"{operation_label(op)} is not available."
        self.op_combo.setCurrentIndex(op_idx)
        self.update_op_controls()

        mat_idx = self._material_combo_index_by_full_name(material)
        if mat_idx < 0:
            self.refresh_operation_material_combo(preferred_name=material)
            mat_idx = self._material_combo_index_by_full_name(material)
        if mat_idx < 0:
            return False, f"FabStack could not select {material}."
        self.material_combo.setCurrentIndex(mat_idx)
        self.update_etchant_options()
        self.update_deposit_methods_for_material()

        if op in {"Deposit", "Spincoating", "Bond"}:
            self.thick_spin.setValue(float(command["thickness_um"]))
        if op == "Etch":
            self.depth_spin.setValue(float(command["depth_um"]))
            mode = str(command.get("etch_mode", "anisotropic")).strip().lower()
            self.current_etch_mode = "isotropic" if mode == "isotropic" else "anisotropic"
            if hasattr(self, "isotropic_etch_checkbox"):
                self.isotropic_etch_checkbox.setChecked(self.current_etch_mode == "isotropic")

        if "temperature_c" in command:
            value = max(self.temp_spin.minimum(), min(self.temp_spin.maximum(), float(command["temperature_c"])))
            self.temp_spin.setValue(value)
        if "time_s" in command:
            value = max(self.exposure_spin.minimum(), min(self.exposure_spin.maximum(), float(command["time_s"])))
            self.exposure_spin.setValue(value)

        method = str(command.get("method", "") or "").strip()
        lithography_method = ""
        bake_method = ""
        if method:
            if op == "Deposit":
                idx = self.deposit_method_combo.findText(method, Qt.MatchFixedString)
                if idx < 0:
                    for i in range(self.deposit_method_combo.count()):
                        if self.deposit_method_combo.itemText(i).strip().lower() == method.lower():
                            idx = i
                            break
                if idx < 0:
                    return False, f"The {method} deposition method is not available for {material}."
                self.deposit_method_combo.setCurrentIndex(idx)
            elif op == "Lithography":
                lithography_method = method
                self.update_etchant_options()
                self._set_etchant_combo_value(method)
            elif op == "Bake/Annealing":
                bake_method = method
            elif op in {"Etch", "Bond", "Strip material", "Lift-off", "Clean / Surface prep"}:
                self._set_etchant_combo_value(method)

        if op == "Etch":
            mode = str(command.get("etch_mode", "anisotropic")).strip().lower()
            self.current_etch_mode = "isotropic" if mode == "isotropic" else "anisotropic"
            if hasattr(self, "isotropic_etch_checkbox"):
                self.isotropic_etch_checkbox.setChecked(self.current_etch_mode == "isotropic")

        if op == "Lithography":
            self.clear_litho_windows(refresh_plot=False)
            for start, end in command.get("windows", []):
                self.add_litho_window_range(float(start), float(end), show_warning=False, refresh_plot=False)
            invert_lithography = _coerce_bool(command.get("invert_lithography")) if "invert_lithography" in command else None
            if invert_lithography is None:
                invert_lithography = self._lithography_invert_from_text(method)
            if invert_lithography is None:
                invert_lithography = False
            self._set_lithography_invert(bool(invert_lithography), sync_method=True)
            method_invert = self._lithography_invert_from_text(lithography_method)
            if method_invert is not None and bool(method_invert) != bool(invert_lithography):
                lithography_method = self._lithography_method_for_invert(bool(invert_lithography))

        include_method = bool(
            str(command.get("method", "") or "").strip()
            or "temperature_c" in command
            or "time_s" in command
        )
        prev_include_method = bool(getattr(self, "include_method_in_step_titles", True))
        had_lithography_override = hasattr(self, "_assistant_lithography_method_override")
        prev_lithography_method = getattr(self, "_assistant_lithography_method_override", "")
        had_bake_override = hasattr(self, "_assistant_bake_method_override")
        prev_bake_method = getattr(self, "_assistant_bake_method_override", "")
        had_value_overrides = hasattr(self, "_operation_value_overrides")
        prev_value_overrides = getattr(self, "_operation_value_overrides", None)
        value_overrides = {}
        for source_key, target_key in (
            ("thickness_um", "thickness_um"),
            ("depth_um", "depth_um"),
            ("temperature_c", "temperature_c"),
            ("time_s", "time_s"),
        ):
            if source_key in command:
                value_overrides[target_key] = command.get(source_key)
        before = len(getattr(self, "history", []))
        try:
            if not include_method:
                self.include_method_in_step_titles = False
            if lithography_method:
                self._assistant_lithography_method_override = lithography_method
            if bake_method:
                self._assistant_bake_method_override = bake_method
            self._operation_value_overrides = value_overrides
            self.apply_operation()
        finally:
            self.include_method_in_step_titles = prev_include_method
            if had_value_overrides:
                self._operation_value_overrides = prev_value_overrides
            elif hasattr(self, "_operation_value_overrides"):
                delattr(self, "_operation_value_overrides")
            if had_lithography_override:
                self._assistant_lithography_method_override = prev_lithography_method
            elif hasattr(self, "_assistant_lithography_method_override"):
                delattr(self, "_assistant_lithography_method_override")
            if had_bake_override:
                self._assistant_bake_method_override = prev_bake_method
            elif hasattr(self, "_assistant_bake_method_override"):
                delattr(self, "_assistant_bake_method_override")
            if hasattr(self, "act_toggle_method_in_titles"):
                self.act_toggle_method_in_titles.setChecked(prev_include_method)
        after = len(getattr(self, "history", []))
        label = self._assistant_command_label(command)
        if after <= before:
            return (
                False,
                f"{label} made no change. Check the current stack and step settings, "
                "then try again.",
            )
        return True, label

    def _assistant_runtime_material_for_operation(self, operation: str, material: str) -> str:
        material = str(material or "").strip()
        if operation not in {"Etch", "Strip material", "Lift-off"} or not material:
            return material
        if not getattr(self, "material_names", None) or getattr(self, "xsec", None) is None:
            return material

        try:
            original_idx = self.material_names.index(material)
        except ValueError:
            original_idx = -1

        variant_candidates = []
        material_low = material.lower()
        for index, name in enumerate(self.material_names):
            name = str(name)
            low = name.lower()
            if index == original_idx:
                continue
            if low.startswith(material_low + " ") and ("baked" in low or "annealed" in low):
                variant_candidates.append((index, name))

        if not variant_candidates:
            return material

        grid = getattr(self.xsec, "grid", None)
        if grid is None:
            return material

        try:
            original_present = original_idx >= 0 and bool(np.any(grid == original_idx))
            visible_variants = [
                (index, name)
                for index, name in variant_candidates
                if bool(np.any(grid == index))
            ]
        except Exception:
            return material

        if visible_variants and not original_present:
            return visible_variants[0][1]
        return material

    def _assistant_resolve_material(self, name):
        target = str(name or "").strip()
        if not target:
            return ""
        candidates = []
        for material in getattr(self, "material_names", []):
            if str(material) != "Void":
                candidates.append(str(material))
        for names in getattr(self, "material_library_by_cat", {}).values():
            if isinstance(names, list):
                for material in names:
                    txt = str(material)
                    if txt != "Void" and txt not in candidates:
                        candidates.append(txt)

        target_low = target.lower()
        for material in candidates:
            if material.lower() == target_low:
                return material
        canonical = _canonical_material(target, candidates)
        if canonical:
            return canonical
        for material in candidates:
            short = self._material_short_name(material).lower()
            if short and short == target_low:
                return material
        for material in candidates:
            head = material.split("(", 1)[0].strip().lower()
            if head and head == target_low:
                return material
        for material in candidates:
            if target_low in material.lower() or material.lower() in target_low:
                return material
        return ""

    def _assistant_ensure_material(self, material):
        material = self._assistant_resolve_material(material)
        if not material or material == "Void":
            return False

        if material in self.material_names:
            self.deleted_materials.discard(material)
        else:
            self.material_names.append(material)
            self.material_colors.append(self._assistant_default_color_for_material(material))

        if material != "Void" and self.substrate_combo.findText(material) < 0:
            self.substrate_combo.addItem(material)

        self.refresh_operation_material_combo(preferred_name=material)
        self.refresh_materials_list()
        self.update_etchant_options()
        self.update_deposit_methods_for_material()
        return True

    def _assistant_default_color_for_material(self, material):
        category = self._assistant_material_category(material)
        colors = {
            "Substrates": "#b0b0b0",
            "Dielectrics": "#79aee6",
            "Polymers": "#64b58a",
            "Photoresists": "#c37ad9",
            "Metals": "#d4af37",
            "Ceramics": "#9a8f7a",
            "Solvents": "#8bc6ff",
            "Developers": "#7fc6a4",
            "Etchants": "#f08f5f",
        }
        preferred = colors.get(category, "#cccccc")
        if hasattr(self, "_unique_material_color"):
            return self._unique_material_color(preferred, exclude_name=material)
        return preferred

    def _assistant_material_category(self, material):
        for category, names in getattr(self, "material_library_by_cat", {}).items():
            if isinstance(names, list) and material in names:
                return str(category)
        return ""

    def _assistant_stack_layer_context(self):
        layers = []
        history = getattr(self, "history", [])
        if hasattr(self, "history_list") and self.history_list is not None:
            row = self.history_list.currentRow()
            if 0 <= row < len(history):
                history = history[:row + 1]

        for item in history:
            if not isinstance(item, dict):
                continue
            op = str(item.get("operation", "") or "").strip()
            material = str(item.get("material_name", "") or "").strip()
            if not material:
                continue

            if op in {"Deposit", "Spincoating", "Bond"}:
                try:
                    thickness = float(item.get("thickness_um", 0.0) or 0.0)
                except (TypeError, ValueError):
                    thickness = 0.0
                if thickness > 0.0:
                    layers.append({"material": material, "thickness_um": thickness})
                continue

            if op == "Etch":
                try:
                    remaining = float(item.get("depth_um", 0.0) or 0.0)
                except (TypeError, ValueError):
                    remaining = 0.0
                for layer in reversed(layers):
                    if remaining <= 0.0:
                        break
                    if layer.get("material") != material:
                        continue
                    take = min(float(layer.get("thickness_um", 0.0) or 0.0), remaining)
                    layer["thickness_um"] = max(0.0, float(layer.get("thickness_um", 0.0) or 0.0) - take)
                    remaining -= take
                layers = [layer for layer in layers if float(layer.get("thickness_um", 0.0) or 0.0) > 1e-12]
                continue

            if op == "Strip material":
                for index in range(len(layers) - 1, -1, -1):
                    if layers[index].get("material") == material:
                        layers.pop(index)
                        break
                continue

            if op == "Flip":
                layers.reverse()

        material_thickness = {}
        top_order = []
        for layer in reversed(layers):
            material = str(layer.get("material", "") or "").strip()
            if not material:
                continue
            if material not in material_thickness:
                material_thickness[material] = float(layer.get("thickness_um", 0.0) or 0.0)
            if material not in top_order:
                top_order.append(material)

        substrate = ""
        if hasattr(self, "substrate_combo"):
            substrate = str(self.substrate_combo.currentText() or "").strip()
        if substrate and substrate not in top_order:
            top_order.append(substrate)

        return material_thickness, top_order

    def _assistant_recent_operation(self):
        operation_terms = [
            ("Clean / Surface prep", ("surface preparation", "surface prep", "clean")),
            ("Spincoating", ("spin coating", "spin-coating", "spin coat", "spin-coat", "spincoat", "spincoating")),
            ("Strip material", ("remove photoresist", "remove resist", "remove pr", "strip")),
            ("Bake/Annealing", ("annealing", "anneal", "bake", "cure")),
            ("Lithography", ("lithography", "litho", "pattern", "develop")),
            ("Lift-off", ("lift-off", "lift off", "liftoff")),
            ("Deposit", (
                "deposit", "deposition", "deposion", "depostion", "deposite", "depositon",
                "evaporate", "sputter", "ald", "cvd",
            )),
            ("Bond", ("bonding", "bond")),
            ("Etch", ("etch", "rie", "drie")),
            ("Flip", ("flip",)),
        ]
        for msg in reversed(getattr(self, "llm_chat_messages", [])):
            if not isinstance(msg, dict) or msg.get("role") != "user":
                continue
            content = str(msg.get("content", "") if isinstance(msg, dict) else "").lower()
            if not content:
                continue
            best = (-1, "")
            for operation, terms in operation_terms:
                for term in terms:
                    idx = content.rfind(term)
                    if idx > best[0]:
                        best = (idx, operation)
            if best[1]:
                return best[1]
        return ""

    def _build_assistant_context(self):
        active_materials = [
            str(n)
            for n in getattr(self, "material_names", [])
            if str(n) != "Void" and str(n) not in getattr(self, "deleted_materials", set())
        ]
        library_materials = []
        non_stack_categories = {"Etchants", "Solvents", "Developers"}
        for category, names in getattr(self, "material_library_by_cat", {}).items():
            if str(category) in non_stack_categories:
                continue
            if isinstance(names, list):
                for name in names:
                    txt = str(name)
                    if txt and txt not in library_materials:
                        library_materials.append(txt)

        history_titles = []
        for item in getattr(self, "history", []):
            if isinstance(item, dict):
                title = str(item.get("title", "")).strip()
                if title:
                    history_titles.append(title)

        recent_user_requests = []
        recent_chat_messages = []
        for msg in getattr(self, "llm_chat_messages", [])[-8:]:
            if not isinstance(msg, dict):
                continue
            role = str(msg.get("role", "") or "").strip()
            content = str(msg.get("content", "") or "").strip()
            if role in {"user", "assistant"} and content:
                recent_chat_messages.append({"role": role, "content": content})
            if role == "user" and content:
                recent_user_requests.append(content)

        substrate = ""
        if hasattr(self, "substrate_combo") and hasattr(self, "substrate_thick_spin"):
            substrate = f"{self.substrate_combo.currentText()} ({self.substrate_thick_spin.value()} µm)"

        library_summary = []
        for cat, names in sorted(getattr(self, "material_library_by_cat", {}).items()):
            if not isinstance(names, list) or not names:
                continue
            shown = ", ".join(str(x) for x in names[:10])
            suffix = "..." if len(names) > 10 else ""
            library_summary.append(f"{cat}: {shown}{suffix}")

        compatibility_notes = []
        for material in active_materials[:16]:
            dep = []
            etch = []
            try:
                dep = self.compatible_deposit_methods(material)
            except Exception:
                dep = []
            try:
                etch = self.compatible_etchants(material)
            except Exception:
                etch = []
            if dep:
                compatibility_notes.append(f"Deposit {material}: {', '.join(dep[:8])}")
            if etch:
                compatibility_notes.append(f"Etch {material}: {', '.join(etch[:8])}")

        restrictions = []
        for op in ALLOWED_OPERATIONS:
            blocked = getattr(self, "operation_restrictions_by_operation", {}).get(op, [])
            if isinstance(blocked, list) and blocked:
                restrictions.append(f"{op}: {', '.join(str(x) for x in blocked[:12])}")

        material_thickness_um, top_material_order = self._assistant_stack_layer_context()

        return {
            "substrate": substrate,
            "active_materials": active_materials,
            "library_materials": library_materials,
            "photoresist_materials": list(getattr(self, "material_library_by_cat", {}).get("Photoresists", [])),
            "process_flow": history_titles,
            "material_thickness_um": material_thickness_um,
            "top_material_order": top_material_order,
            "recent_operation": self._assistant_recent_operation(),
            "recent_user_requests": recent_user_requests,
            "recent_chat_messages": recent_chat_messages,
            "library_summary": library_summary,
            "compatibility_notes": compatibility_notes,
            "operation_restrictions": restrictions,
        }
