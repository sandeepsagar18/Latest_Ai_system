import sys
from pathlib import Path
import unittest
from unittest.mock import MagicMock, patch

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

class TestApproachAViewIsolation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with patch("gui.SmartClassApp.update_clock"), patch("gui.SmartClassApp.refresh_dashboard_metrics"):
            from gui import SmartClassApp
            cls.app = SmartClassApp()
            cls.app.withdraw()

    @classmethod
    def tearDownClass(cls):
        try:
            cls.app.destroy()
        except:
            pass


    def test_state1_standby_kiosk_logged_out(self):
        """Logged out state: Standby kiosk hero is shown, all sensitive buttons/tabs are completely removed."""
        self.app.current_user = None
        self.app.update_session_display()
        self.app.update()

        # Main view state
        self.assertIsNotNone(self.app.standby_frame)
        self.assertEqual(self.app.standby_frame.winfo_manager(), "grid")
        self.assertEqual(self.app.tabview.winfo_manager(), "")  # grid_forget

        # Dynamic sidebar button references must be None
        self.assertIsNone(self.app.btn_attendance)
        self.assertIsNone(self.app.btn_admin)
        self.assertIsNone(self.app.btn_train)
        self.assertIsNone(self.app.btn_teacher)

        # In standby mode, no reset button should be packed
        self.assertEqual(self.app.btn_reset_attendance.winfo_manager(), "")

        # Test security guards reject unauthorized calls
        with patch.object(self.app, 'show_warning') as mock_warn:
            self.assertFalse(self.app.require_login())
            mock_warn.assert_called()

        with patch.object(self.app, 'show_warning') as mock_warn:
            self.assertFalse(self.app.require_admin())
            mock_warn.assert_called()

    def test_state2_teacher_workspace_isolation(self):
        """Teacher state: Tabview shown, only Faculty tools rendered. Admin Portal, Train, Delete, Register are absent."""
        teacher_user = {
            "id": 10,
            "name": "Prof. Alan Turing",
            "employee_id": "EMP-TURING",
            "role": "teacher",
            "department": "CSE"
        }
        self.app.current_user = teacher_user
        self.app.update_session_display()
        self.app.update()

        # Main view state
        self.assertEqual(self.app.standby_frame.winfo_manager(), "")  # hidden
        self.assertEqual(self.app.tabview.winfo_manager(), "grid")    # visible

        # Permitted Faculty tools must exist
        self.assertIsNotNone(self.app.btn_attendance, "Teacher must have Batch Class Scan")
        self.assertIsNotNone(self.app.btn_teacher, "Teacher must have My Allotted Classes")

        # Admin & destructive tools MUST be None (completely omitted from UI)
        self.assertIsNone(self.app.btn_admin, "Teacher must NEVER have Admin Command Portal button")
        self.assertIsNone(self.app.btn_train, "Teacher must NEVER have Train AI Brain button")
        self.assertEqual(self.app.btn_reset_attendance.winfo_manager(), "", "Reset attendance must be hidden from teacher")

        # Backend guards must reject any administrative attempts
        with patch.object(self.app, 'show_error') as mock_err:
            self.assertFalse(self.app.require_admin())
            mock_err.assert_called_with("Access Denied", unittest.mock.ANY)

    def test_state3_admin_command_suite(self):
        """Admin state: Tabview shown, full administrative suite rendered including Admin Portal, Train AI Brain, etc."""
        admin_user = {
            "id": 1,
            "name": "Super Admin",
            "employee_id": "ADMIN01",
            "role": "admin",
            "department": "ADMINISTRATION"
        }
        self.app.current_user = admin_user
        self.app.update_session_display()
        self.app.update()

        # Main view state
        self.assertEqual(self.app.standby_frame.winfo_manager(), "")  # hidden
        self.assertEqual(self.app.tabview.winfo_manager(), "grid")    # visible

        # All admin tools must be rendered
        self.assertIsNotNone(self.app.btn_admin, "Admin must have Admin Command Portal button")
        self.assertIsNotNone(self.app.btn_attendance, "Admin must have Batch Class Scan")
        self.assertIsNotNone(self.app.btn_train, "Admin must have Train AI Brain")
        self.assertIsNotNone(self.app.btn_teacher, "Admin must have Curriculum Matrix")
        self.assertEqual(self.app.btn_reset_attendance.winfo_manager(), "pack", "Reset attendance must be visible for admin")

        # require_admin should succeed
        self.assertTrue(self.app.require_admin())

    def test_logout_resets_to_standby(self):
        """Logging out resets the view immediately to Standby Kiosk with zero exposed buttons."""
        # First log in as admin
        self.app.current_user = {
            "id": 1, "name": "Admin", "employee_id": "ADMIN01", "role": "admin", "department": "ADMIN"
        }
        self.app.update_session_display()
        self.assertIsNotNone(self.app.btn_admin)

        # Log out
        with patch.object(self.app, 'show_info'):
            self.app.logout()

        self.assertIsNone(self.app.current_user)
        self.assertEqual(self.app.standby_frame.winfo_manager(), "grid")
        self.assertEqual(self.app.tabview.winfo_manager(), "")
        self.assertIsNone(self.app.btn_admin)
        self.assertIsNone(self.app.btn_attendance)
        self.assertIsNone(self.app.btn_train)

if __name__ == "__main__":
    unittest.main()
