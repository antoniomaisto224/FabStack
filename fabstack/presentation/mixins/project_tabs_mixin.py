"""Project tabs and autosave behavior for MainWindow."""

import hashlib
import os
import pickle
import time

from PyQt5.QtWidgets import QFrame, QHBoxLayout, QLabel, QPushButton, QTabBar

from fabstack.persistence.process_file import save_process_file


class ProjectTabsMixin:
    def _deepcopy_process_data(self, data: dict):
        try:
            return pickle.loads(pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL))
        except Exception:
            return data

    def _process_data_hash(self, data: dict) -> str:
        try:
            raw = pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception:
            return ""
        return hashlib.sha1(raw).hexdigest()

    def _next_untitled_tab_name(self) -> str:
        name = f"Untitled {self.untitled_tab_counter}"
        self.untitled_tab_counter += 1
        return name

    def _session_display_name(self, session: dict) -> str:
        path = session.get("path")
        if path:
            return os.path.basename(path)
        name = str(session.get("name", "")).strip()
        return name if name else "Untitled"

    def _install_project_tab_close_button(self, index: int):
        if index < 0 or not hasattr(self, "project_tabs"):
            return
        if index >= self.project_tabs.count():
            return

        close_button = QPushButton("×")
        close_button.setObjectName("simpleTabCloseButton")
        close_button.setFlat(True)
        close_button.setAutoDefault(False)
        close_button.setDefault(False)
        close_button.setFixedSize(16, 16)
        close_button.setToolTip("Close process")
        close_button.clicked.connect(
            lambda _checked=False, button=close_button: self._close_project_tab_from_button(button)
        )
        self.project_tabs.setTabButton(index, QTabBar.RightSide, close_button)

    def _close_project_tab_from_button(self, button: QPushButton):
        if not hasattr(self, "project_tabs"):
            return
        for index in range(self.project_tabs.count()):
            if self.project_tabs.tabButton(index, QTabBar.RightSide) is button:
                self.on_project_tab_close_requested(index)
                return

    def _update_project_tab_ui(self, index: int):
        if index < 0 or index >= len(self.project_sessions):
            return
        session = self.project_sessions[index]
        label = self._session_display_name(session)
        tooltip = session.get("path") or label
        self.project_tabs.setTabText(index, label)
        self.project_tabs.setTabToolTip(index, tooltip)
        if index == self.project_tabs.currentIndex():
            self._refresh_window_title_from_active_tab()

    def _refresh_window_title_from_active_tab(self):
        if self.current_process_path:
            self.setWindowTitle(f"{self.base_title} — {os.path.basename(self.current_process_path)}")
            return
        idx = self.project_tabs.currentIndex() if hasattr(self, "project_tabs") else -1
        if 0 <= idx < self.project_tabs.count():
            self.setWindowTitle(f"{self.base_title} — {self.project_tabs.tabText(idx)}")
        else:
            self.setWindowTitle(self.base_title)

    def _snapshot_tab_state(self, index: int):
        if self.xsec is None:
            return
        if index < 0 or index >= len(self.project_sessions):
            return
        session = self.project_sessions[index]
        live_data = self._deepcopy_process_data(self._make_process_dict())
        session["data"] = live_data
        session["path"] = self.current_process_path
        if not session.get("path"):
            txt = self.project_tabs.tabText(index) if 0 <= index < self.project_tabs.count() else ""
            if txt:
                session["name"] = txt
        current_hash = self._process_data_hash(live_data)
        if session.get("saved_hash") is None:
            session["saved_hash"] = current_hash
        session["dirty"] = (current_hash != session.get("saved_hash"))
        self._update_project_tab_ui(index)

    def _snapshot_active_tab_state(self):
        if self.active_project_tab_index >= 0:
            self._snapshot_tab_state(self.active_project_tab_index)

    def _is_unmodified_untitled_session(self, index: int) -> bool:
        if index < 0 or index >= len(self.project_sessions):
            return False
        session = self.project_sessions[index]
        if session.get("path"):
            return False
        name = str(session.get("name", "")).strip()
        tab_name = ""
        if 0 <= index < self.project_tabs.count():
            tab_name = str(self.project_tabs.tabText(index)).strip()
        if not (name.startswith("Untitled") or tab_name.startswith("Untitled")):
            return False
        if bool(session.get("dirty", False)):
            return False
        data = session.get("data")
        saved_hash = session.get("saved_hash")
        if not isinstance(data, dict) or not saved_hash:
            return False
        return self._process_data_hash(data) == saved_hash

    def _replace_session_in_tab(self, index: int, data: dict, path: str = None, name: str = None):
        if index < 0 or index >= len(self.project_sessions):
            return False

        session_name = name or (os.path.basename(path) if path else self._next_untitled_tab_name())
        data_copy = self._deepcopy_process_data(data)
        data_hash = self._process_data_hash(data_copy)
        self.project_sessions[index] = {
            "name": session_name,
            "path": path,
            "data": data_copy,
            "saved_hash": data_hash,
            "dirty": False,
            "created_ts": time.time(),
            "unsaved_prompt_shown": False,
        }

        self.tab_switch_in_progress = True
        self.project_tabs.setTabText(index, session_name)
        self.project_tabs.setTabToolTip(index, path or session_name)
        self._install_project_tab_close_button(index)
        self.project_tabs.setCurrentIndex(index)
        self.tab_switch_in_progress = False

        self.active_project_tab_index = index
        self._apply_session_to_workspace(index)
        return True

    def _apply_session_to_workspace(self, index: int):
        if index < 0 or index >= len(self.project_sessions):
            return
        session = self.project_sessions[index]
        data = self._deepcopy_process_data(session.get("data", {}))
        self._load_process_dict(data)
        self.current_process_path = session.get("path")
        self._update_project_tab_ui(index)
        self._refresh_window_title_from_active_tab()
        self._update_restore_reset_action_state()
        self._hide_unsaved_save_prompt()
        self._maybe_prompt_unsaved_active_tab()

    def _open_session_in_new_tab(self, data: dict, path: str = None, name: str = None):
        self._snapshot_active_tab_state()
        if self._is_unmodified_untitled_session(self.active_project_tab_index):
            if self._replace_session_in_tab(self.active_project_tab_index, data, path=path, name=name):
                return

        session_name = name or (os.path.basename(path) if path else self._next_untitled_tab_name())
        data_copy = self._deepcopy_process_data(data)
        data_hash = self._process_data_hash(data_copy)
        session = {
            "name": session_name,
            "path": path,
            "data": data_copy,
            "saved_hash": data_hash,
            "dirty": False,
            "created_ts": time.time(),
            "unsaved_prompt_shown": False,
        }
        self.project_sessions.append(session)

        self.tab_switch_in_progress = True
        new_index = self.project_tabs.addTab(session_name)
        self._install_project_tab_close_button(new_index)
        self.project_tabs.setCurrentIndex(new_index)
        self.tab_switch_in_progress = False

        self.active_project_tab_index = new_index
        self._apply_session_to_workspace(new_index)

    def _initialize_project_tabs_from_current_state(self):
        self.project_sessions = []
        self.active_project_tab_index = -1
        self._reset_restore_info_by_tab = {}

        self.tab_switch_in_progress = True
        while self.project_tabs.count() > 0:
            self.project_tabs.removeTab(0)
        self.tab_switch_in_progress = False

        first_name = self._next_untitled_tab_name()
        first_data = self._deepcopy_process_data(self._make_process_dict())
        first_hash = self._process_data_hash(first_data)
        self.project_sessions.append({
            "name": first_name,
            "path": self.current_process_path,
            "data": first_data,
            "saved_hash": first_hash,
            "dirty": False,
            "created_ts": time.time(),
            "unsaved_prompt_shown": False,
        })

        self.tab_switch_in_progress = True
        idx = self.project_tabs.addTab(first_name)
        self._install_project_tab_close_button(idx)
        self.project_tabs.setCurrentIndex(idx)
        self.tab_switch_in_progress = False

        self.active_project_tab_index = idx
        self._update_project_tab_ui(idx)
        self._refresh_window_title_from_active_tab()
        self._update_restore_reset_action_state()

    def on_project_tab_changed(self, index: int):
        if self.tab_switch_in_progress:
            return
        if index < 0 or index >= len(self.project_sessions):
            return

        prev = self.active_project_tab_index
        if prev == index:
            self._refresh_window_title_from_active_tab()
            return

        if prev >= 0:
            self._snapshot_tab_state(prev)

        self.active_project_tab_index = index
        self._apply_session_to_workspace(index)

    def on_project_tab_moved(self, from_index: int, to_index: int):
        if from_index == to_index:
            return
        if from_index < 0 or to_index < 0:
            return
        if from_index >= len(self.project_sessions) or to_index >= len(self.project_sessions):
            return
        session = self.project_sessions.pop(from_index)
        self.project_sessions.insert(to_index, session)
        self.active_project_tab_index = self.project_tabs.currentIndex()

    def on_project_tab_close_requested(self, index: int):
        if index < 0 or index >= len(self.project_sessions):
            return

        prev_index = self.project_tabs.currentIndex()
        if prev_index != index:
            self.project_tabs.setCurrentIndex(index)

        if not self._maybe_save_before_discard():
            if 0 <= prev_index < self.project_tabs.count():
                self.project_tabs.setCurrentIndex(prev_index)
            return

        closing_sid = id(self.project_sessions[index])
        self._reset_restore_info_by_tab.pop(closing_sid, None)
        if index == self.project_tabs.currentIndex():
            self._hide_unsaved_save_prompt()

        self.tab_switch_in_progress = True
        self.project_tabs.removeTab(index)
        self.project_sessions.pop(index)
        self.tab_switch_in_progress = False

        if not self.project_sessions:
            self.active_project_tab_index = -1
            self.current_process_path = None
            self._open_session_in_new_tab(self._default_new_process_data(), path=None)
            return

        new_index = min(index, len(self.project_sessions) - 1)
        self.tab_switch_in_progress = True
        self.project_tabs.setCurrentIndex(new_index)
        self.tab_switch_in_progress = False
        self.active_project_tab_index = new_index
        self._apply_session_to_workspace(new_index)

    def _maybe_save_all_tabs_before_close(self) -> bool:
        if not self.project_sessions:
            return True

        self._snapshot_active_tab_state()
        start_index = self.project_tabs.currentIndex()

        for idx in range(len(self.project_sessions)):
            if idx != self.project_tabs.currentIndex():
                self.project_tabs.setCurrentIndex(idx)
            if not self._maybe_save_before_discard():
                if 0 <= start_index < self.project_tabs.count():
                    self.project_tabs.setCurrentIndex(start_index)
                return False
        return True

    def _update_restore_reset_action_state(self):
        if not hasattr(self, "act_restore_reset"):
            return
        idx = int(getattr(self, "active_project_tab_index", -1))
        if idx < 0 or idx >= len(self.project_sessions):
            self.act_restore_reset.setEnabled(False)
            return
        sid = id(self.project_sessions[idx])
        self.act_restore_reset.setEnabled(sid in self._reset_restore_info_by_tab)

    def _apply_autosave_enabled_state(self):
        if getattr(self, "autosave_enabled", False):
            self.autosave_timer.start()
        else:
            self.autosave_timer.stop()
            self._hide_unsaved_save_prompt()

    def on_toggle_autosave(self, checked: bool):
        self.autosave_enabled = bool(checked)
        self._apply_autosave_enabled_state()

    def _hide_unsaved_save_prompt(self):
        w = getattr(self, "unsaved_save_prompt_widget", None)
        if w is not None:
            try:
                w.hide()
                w.deleteLater()
            except Exception:
                pass
        self.unsaved_save_prompt_widget = None

    def _show_unsaved_save_prompt(self):
        self._hide_unsaved_save_prompt()
        if not hasattr(self, "figure_scroll") or self.figure_scroll is None:
            return
        viewport = self.figure_scroll.viewport()
        if viewport is None:
            return

        frame = QFrame(viewport)
        frame.setObjectName("unsavedSavePrompt")
        frame.setFrameShape(QFrame.StyledPanel)
        frame.setStyleSheet(
            """
            QFrame#unsavedSavePrompt {
                background: rgba(20, 20, 20, 210);
                border: 1px solid rgba(180, 180, 180, 130);
                border-radius: 6px;
            }
            QFrame#unsavedSavePrompt QLabel {
                color: #e5e5e5;
                font-size: 11px;
            }
            """
        )

        row = QHBoxLayout(frame)
        row.setContentsMargins(8, 6, 8, 6)
        row.setSpacing(6)
        lbl = QLabel("Save this process now?")
        btn_save = QPushButton("Save")
        btn_later = QPushButton("Not now")
        btn_save.setFixedHeight(22)
        btn_later.setFixedHeight(22)
        row.addWidget(lbl, 1)
        row.addWidget(btn_save, 0)
        row.addWidget(btn_later, 0)

        def _save_now():
            if self.save_process():
                self._hide_unsaved_save_prompt()

        btn_save.clicked.connect(_save_now)
        btn_later.clicked.connect(self._hide_unsaved_save_prompt)

        frame.adjustSize()
        frame.move(10, 10)
        frame.show()
        frame.raise_()
        self.unsaved_save_prompt_widget = frame

    def _maybe_prompt_unsaved_active_tab(self):
        if not getattr(self, "autosave_enabled", False):
            return
        idx = int(getattr(self, "active_project_tab_index", -1))
        if idx < 0 or idx >= len(self.project_sessions):
            return
        session = self.project_sessions[idx]
        if session.get("path"):
            self._hide_unsaved_save_prompt()
            return
        # Prompt only if tab has unsaved changes and is older than the autosave interval.
        if not session.get("dirty", False):
            return
        created_ts = float(session.get("created_ts", time.time()))
        age_sec = time.time() - created_ts
        if age_sec < float(self.autosave_interval_minutes * 60):
            return
        if bool(session.get("unsaved_prompt_shown", False)):
            return
        session["unsaved_prompt_shown"] = True
        self._show_unsaved_save_prompt()

    def _autosave_saved_tabs(self):
        saved_count = 0
        for i, session in enumerate(self.project_sessions):
            path = str(session.get("path") or "").strip()
            if not path:
                continue
            if not bool(session.get("dirty", False)):
                continue
            data = session.get("data")
            if not isinstance(data, dict):
                continue
            try:
                save_process_file(path, data)
            except Exception:
                continue
            new_hash = self._process_data_hash(data)
            session["saved_hash"] = new_hash
            session["dirty"] = False
            if i == self.active_project_tab_index:
                self.current_process_path = path
            self._update_project_tab_ui(i)
            saved_count += 1
        return saved_count

    def _on_autosave_timer(self):
        if not getattr(self, "autosave_enabled", False):
            return
        self._snapshot_active_tab_state()
        saved = self._autosave_saved_tabs()
        if saved > 0:
            noun = "process" if saved == 1 else "processes"
            self.statusBar().showMessage(f"Autosaved {saved} {noun}.", 3000)
        self._maybe_prompt_unsaved_active_tab()
