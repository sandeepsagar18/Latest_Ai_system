import sqlite3
import hashlib
import hmac
import secrets
import pickle
import shutil
import socket
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from utils.config import BASE_DIR, ATTENDANCE_DIR

DB_PATH = BASE_DIR / "data" / "smartclass.db"
EMB_PATH = BASE_DIR / "data" / "embeddings.pkl"
BACKUP_DIR = BASE_DIR / "data" / "backups"
SYSTEM_HMAC_SECRET = b"SmartClass_Enterprise_HMAC_Secret_Salt_2026!#%"


def hash_password(password, salt=None):
    if salt is None:
        salt = secrets.token_hex(16)
    hashed = hashlib.pbkdf2_hmac(
        'sha256',
        password.encode('utf-8'),
        salt.encode('utf-8'),
        100000
    ).hex()
    return hashed, salt


def verify_password(password, stored_hash, salt):
    hashed, _ = hash_password(password, salt)
    return secrets.compare_digest(hashed, stored_hash)


def init_db():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # 1. Students Table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS students (
            roll_number TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            gender TEXT NOT NULL,
            degree TEXT NOT NULL,
            year TEXT NOT NULL,
            branch TEXT NOT NULL,
            section TEXT NOT NULL,
            registered_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    # Check if teachers table exists and has all required security columns
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='teachers'")
    teachers_table_exists = cursor.fetchone() is not None

    if teachers_table_exists:
        cursor.execute("PRAGMA table_info(teachers)")
        columns = [col[1] for col in cursor.fetchall()]
        if "password_hash" not in columns or "salt" not in columns:
            cursor.execute("DROP TABLE IF EXISTS attendance_records")
            cursor.execute("DROP TABLE IF EXISTS subjects")
            cursor.execute("DROP TABLE IF EXISTS teachers")
            teachers_table_exists = False
        else:
            if "failed_login_attempts" not in columns:
                cursor.execute("ALTER TABLE teachers ADD COLUMN failed_login_attempts INTEGER DEFAULT 0")
            if "locked_until" not in columns:
                cursor.execute("ALTER TABLE teachers ADD COLUMN locked_until TEXT DEFAULT NULL")

    # 2. Secure Teachers Table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS teachers (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            employee_id TEXT UNIQUE NOT NULL,
            name TEXT NOT NULL,
            department TEXT NOT NULL,
            email TEXT,
            password_hash TEXT NOT NULL,
            salt TEXT NOT NULL,
            role TEXT DEFAULT 'teacher',
            failed_login_attempts INTEGER DEFAULT 0,
            locked_until TEXT DEFAULT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    # 3. Subjects Table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS subjects (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            teacher_id INTEGER NOT NULL,
            subject_code TEXT NOT NULL,
            subject_name TEXT NOT NULL,
            degree TEXT NOT NULL,
            year TEXT NOT NULL,
            branch TEXT NOT NULL,
            section TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (teacher_id) REFERENCES teachers(id) ON DELETE CASCADE,
            UNIQUE(teacher_id, subject_code, degree, year, branch, section)
        )
    ''')

    # 4. Attendance Records Table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS attendance_records (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            subject_id INTEGER,
            subject_code TEXT,
            subject_name TEXT,
            teacher_name TEXT,
            roll_number TEXT,
            name TEXT,
            branch TEXT,
            section TEXT,
            date TEXT,
            time_marked TEXT,
            status TEXT,
            FOREIGN KEY (roll_number) REFERENCES students(roll_number),
            FOREIGN KEY (subject_id) REFERENCES subjects(id)
        )
    ''')

    # 5. Enterprise Security Audit Logs Table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS security_audit_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            event_type TEXT NOT NULL,
            severity TEXT NOT NULL,
            user_identifier TEXT,
            ip_or_host TEXT,
            details TEXT
        )
    ''')

    # 6. Cryptographic Attendance Checksums Table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS attendance_checksums (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT UNIQUE NOT NULL,
            file_hash TEXT NOT NULL,
            hmac_signature TEXT NOT NULL,
            created_by TEXT,
            created_at TEXT NOT NULL
        )
    ''')

    # 7. System Settings Table (Master Keys, Configs)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS system_settings (
            setting_key TEXT PRIMARY KEY,
            setting_value TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
    ''')

    # Default Admin Master Authorization Key
    cursor.execute("SELECT setting_value FROM system_settings WHERE setting_key = 'admin_master_key'")
    if not cursor.fetchone():
        cursor.execute("INSERT INTO system_settings (setting_key, setting_value, updated_at) VALUES (?, ?, ?)",
                       ("admin_master_key", "SmartClass@Admin#2026", datetime.now().isoformat()))

    conn.commit()
    conn.close()

    init_default_admin()


def init_default_admin():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT 1 FROM teachers WHERE role = 'admin'")
    admin_exists = cursor.fetchone() is not None
    if not admin_exists:
        p_hash, salt = hash_password("admin123")
        cursor.execute('''
            INSERT INTO teachers (employee_id, name, department, email, password_hash, salt, role)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', ("ADMIN01", "System Administrator", "ADMINISTRATION", "admin@smartclass.edu", p_hash, salt, "admin"))
        conn.commit()
    conn.close()


# SECURITY AUDIT LOGGING & HELPERS
def get_client_host():
    try:
        return socket.gethostname()
    except:
        return "127.0.0.1"


def log_security_event(event_type, severity, user_identifier, details):
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO security_audit_logs (timestamp, event_type, severity, user_identifier, ip_or_host, details)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), event_type, severity, str(user_identifier), get_client_host(), str(details)))
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"[SECURITY LOG ERROR] {e}")


# AUTHENTICATION & TEACHER MANAGEMENT
def register_teacher_secure(employee_id, name, department, email, password, role="teacher"):
    init_db()
    emp_clean = employee_id.strip().upper()
    if len(password) < 6:
        return False, "Password must be at least 6 characters long!", None

    p_hash, salt = hash_password(password)
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO teachers (employee_id, name, department, email, password_hash, salt, role, failed_login_attempts, locked_until)
            VALUES (?, ?, ?, ?, ?, ?, ?, 0, NULL)
        ''', (emp_clean, name.strip(), department.strip(), email.strip(), p_hash, salt, role))
        conn.commit()
        teacher_id = cursor.lastrowid
        conn.close()
        log_security_event("ACCOUNT_CREATED", "INFO", emp_clean, f"Registered new {role.upper()} account: {name} (Dept: {department})")
        return True, f"Account created securely for {name}!", teacher_id
    except sqlite3.IntegrityError:
        log_security_event("REGISTER_FAILED", "WARNING", emp_clean, f"Duplicate employee registration attempted: {emp_clean}")
        return False, f"Employee ID '{emp_clean}' is already registered!", None


def authenticate_teacher(employee_id, password):
    init_db()
    emp_clean = employee_id.strip().upper()
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        SELECT id, employee_id, name, department, email, password_hash, salt, role, failed_login_attempts, locked_until
        FROM teachers WHERE employee_id = ?
    ''', (emp_clean,))
    row = cursor.fetchone()

    if not row:
        conn.close()
        log_security_event("LOGIN_FAILED", "WARNING", emp_clean, f"Login attempt with non-existent ID: {emp_clean}")
        return False, "Invalid Employee ID or password!", None

    t_id, emp_id, name, dept, email, p_hash, salt, role, failed_attempts, locked_until = row
    failed_attempts = failed_attempts or 0
    now = datetime.now()

    # 1. Check if account is locked
    if locked_until:
        try:
            lock_time = datetime.fromisoformat(locked_until)
            if now < lock_time:
                conn.close()
                mins_left = max(1, int((lock_time - now).total_seconds() / 60) + 1)
                log_security_event("AUTH_LOCKED_ATTEMPT", "WARNING", emp_clean, f"Attempted login to locked account ({mins_left}m remaining)")
                return False, f"Account locked due to 5 failed attempts! Try again in {mins_left} minutes, or contact Administrator.", None
            else:
                # Lock expired
                cursor.execute('UPDATE teachers SET failed_login_attempts = 0, locked_until = NULL WHERE id = ?', (t_id,))
                conn.commit()
                failed_attempts = 0
        except Exception:
            pass

    # 2. Verify password
    if verify_password(password, p_hash, salt):
        cursor.execute('UPDATE teachers SET failed_login_attempts = 0, locked_until = NULL WHERE id = ?', (t_id,))
        conn.commit()
        conn.close()
        user_info = {
            "id": t_id,
            "employee_id": emp_id,
            "name": name,
            "department": dept,
            "email": email,
            "role": role
        }
        log_security_event("LOGIN_SUCCESS", "INFO", emp_clean, f"User {name} ({role}) logged in successfully.")
        return True, "Login successful!", user_info
    else:
        new_failed = failed_attempts + 1
        if new_failed >= 5:
            lock_until = now + timedelta(minutes=15)
            cursor.execute('UPDATE teachers SET failed_login_attempts = ?, locked_until = ? WHERE id = ?',
                           (new_failed, lock_until.isoformat(), t_id))
            conn.commit()
            conn.close()
            log_security_event("ACCOUNT_LOCKED", "CRITICAL", emp_clean, f"Account locked for 15 min after 5 failed login attempts.")
            return False, "Security Lockout: Account locked for 15 minutes due to 5 consecutive failed login attempts!", None
        else:
            cursor.execute('UPDATE teachers SET failed_login_attempts = ? WHERE id = ?', (new_failed, t_id))
            conn.commit()
            conn.close()
            remaining = 5 - new_failed
            log_security_event("LOGIN_FAILED", "WARNING", emp_clean, f"Failed password attempt ({new_failed}/5).")
            return False, f"Invalid Employee ID or password! ({remaining} attempt{'s' if remaining != 1 else ''} remaining before lockout)", None


def verify_teacher_password(teacher_id, password):
    init_db()
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('SELECT password_hash, salt FROM teachers WHERE id = ?', (teacher_id,))
    row = cursor.fetchone()
    conn.close()
    if not row:
        return False
    return verify_password(password, row[0], row[1])


def change_teacher_password(teacher_id, old_password, new_password):
    if len(new_password) < 6:
        return False, "New password must be at least 6 characters long!"

    if not verify_teacher_password(teacher_id, old_password):
        return False, "Current password is incorrect!"

    new_hash, new_salt = hash_password(new_password)
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        UPDATE teachers SET password_hash = ?, salt = ? WHERE id = ?
    ''', (new_hash, new_salt, teacher_id))
    conn.commit()
    conn.close()
    return True, "Password updated successfully!"


def admin_reset_teacher_password(teacher_id, new_password):
    if len(new_password) < 6:
        return False, "Password must be at least 6 characters long!"
    new_hash, new_salt = hash_password(new_password)
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('UPDATE teachers SET password_hash = ?, salt = ? WHERE id = ?', (new_hash, new_salt, teacher_id))
    conn.commit()
    conn.close()
    return True, "Password reset successfully by Administrator!"


def update_teacher_role(teacher_id, new_role):
    if new_role not in ["admin", "teacher"]:
        return False, "Invalid role specified!"
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('UPDATE teachers SET role = ? WHERE id = ?', (new_role, teacher_id))
    conn.commit()
    conn.close()
    return True, f"Role updated to {new_role.upper()}!"


def update_teacher_info(teacher_id, name, department, email):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('UPDATE teachers SET name = ?, department = ?, email = ? WHERE id = ?',
                   (name.strip(), department.strip(), email.strip(), teacher_id))
    conn.commit()
    conn.close()
    return True, "Teacher profile updated successfully!"


def get_all_teachers():
    init_db()
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('SELECT id, employee_id, name, department, email, role, created_at, failed_login_attempts, locked_until FROM teachers ORDER BY name ASC')
    rows = cursor.fetchall()
    conn.close()
    teachers = []
    for r in rows:
        teachers.append({
            "id": r[0],
            "employee_id": r[1],
            "name": r[2],
            "department": r[3],
            "email": r[4],
            "role": r[5],
            "created_at": r[6],
            "failed_login_attempts": r[7] or 0,
            "locked_until": r[8]
        })
    return teachers


def delete_teacher_secure(target_teacher_id, requesting_user, password):
    req_id = requesting_user.get("id")
    req_role = requesting_user.get("role")

    if req_role != "admin" and req_id != target_teacher_id:
        return False, "Unauthorized! You can only delete your own profile."

    if not verify_teacher_password(req_id, password):
        return False, "Security check failed: Incorrect password!"

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('DELETE FROM attendance_records WHERE subject_id IN (SELECT id FROM subjects WHERE teacher_id = ?)', (target_teacher_id,))
    cursor.execute('DELETE FROM subjects WHERE teacher_id = ?', (target_teacher_id,))
    cursor.execute('DELETE FROM teachers WHERE id = ?', (target_teacher_id,))
    conn.commit()
    conn.close()
    return True, "Teacher profile and associated data deleted securely."


# SUBJECT ALLOTMENTS
def add_subject(teacher_id, subject_code, subject_name, degree, year, branch, section):
    init_db()
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO subjects (teacher_id, subject_code, subject_name, degree, year, branch, section)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (teacher_id, subject_code.strip().upper(), subject_name.strip(), degree, year, branch, section))
        conn.commit()
        sub_id = cursor.lastrowid
        conn.close()
        return True, f"Subject '{subject_name}' allotted successfully to {branch}-{section}!", sub_id
    except sqlite3.IntegrityError:
        return False, f"Subject '{subject_code}' is already allotted to you for this section!", None


def reassign_subject_teacher(subject_id, new_teacher_id):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('UPDATE subjects SET teacher_id = ? WHERE id = ?', (new_teacher_id, subject_id))
    conn.commit()
    conn.close()
    return True, "Subject reassigned to new faculty member successfully!"


def get_subjects_by_teacher(teacher_id):
    init_db()
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        SELECT s.id, s.teacher_id, s.subject_code, s.subject_name, s.degree, s.year, s.branch, s.section, t.name, t.employee_id
        FROM subjects s
        JOIN teachers t ON s.teacher_id = t.id
        WHERE s.teacher_id = ?
        ORDER BY s.branch, s.section, s.subject_code
    ''', (teacher_id,))
    rows = cursor.fetchall()
    conn.close()
    subjects = []
    for r in rows:
        subjects.append({
            "id": r[0],
            "teacher_id": r[1],
            "subject_code": r[2],
            "subject_name": r[3],
            "degree": r[4],
            "year": r[5],
            "branch": r[6],
            "section": r[7],
            "teacher_name": r[8],
            "teacher_emp_id": r[9]
        })
    return subjects


def get_all_subjects():
    init_db()
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        SELECT s.id, s.teacher_id, s.subject_code, s.subject_name, s.degree, s.year, s.branch, s.section, t.name, t.employee_id
        FROM subjects s
        JOIN teachers t ON s.teacher_id = t.id
        ORDER BY t.name, s.branch, s.section, s.subject_code
    ''')
    rows = cursor.fetchall()
    conn.close()
    subjects = []
    for r in rows:
        subjects.append({
            "id": r[0],
            "teacher_id": r[1],
            "subject_code": r[2],
            "subject_name": r[3],
            "degree": r[4],
            "year": r[5],
            "branch": r[6],
            "section": r[7],
            "teacher_name": r[8],
            "teacher_emp_id": r[9]
        })
    return subjects


def delete_subject_secure(subject_id, requesting_user, password):
    req_id = requesting_user.get("id")
    req_role = requesting_user.get("role")

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT teacher_id FROM subjects WHERE id = ?", (subject_id,))
    row = cursor.fetchone()
    if not row:
        conn.close()
        return False, "Subject not found!"

    owner_id = row[0]
    if req_role != "admin" and req_id != owner_id:
        conn.close()
        return False, "Unauthorized! You can only delete your own subjects."

    if not verify_teacher_password(req_id, password):
        conn.close()
        return False, "Security check failed: Incorrect password!"

    cursor.execute('DELETE FROM attendance_records WHERE subject_id = ?', (subject_id,))
    cursor.execute('DELETE FROM subjects WHERE id = ?', (subject_id,))
    conn.commit()
    conn.close()
    return True, "Subject removed successfully."


# STUDENT MANAGEMENT & SECTION TRANSFERS
def student_exists(roll_number):
    init_db()
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('SELECT 1 FROM students WHERE roll_number = ?', (roll_number,))
    exists = cursor.fetchone() is not None
    conn.close()
    return exists


def get_all_students_master():
    init_db()
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        SELECT roll_number, name, gender, degree, year, branch, section, registered_at
        FROM students
        ORDER BY branch, section, roll_number ASC
    ''')
    rows = cursor.fetchall()
    conn.close()
    return rows


def get_student_info(roll_number):
    init_db()
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('SELECT roll_number, name, gender, degree, year, branch, section, registered_at FROM students WHERE roll_number = ?', (roll_number,))
    result = cursor.fetchone()
    conn.close()
    return result


def update_student_details(roll_number, name, gender, degree, year, branch, section):
    init_db()
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        UPDATE students
        SET name = ?, gender = ?, degree = ?, year = ?, branch = ?, section = ?
        WHERE roll_number = ?
    ''', (name.strip(), gender.strip(), degree.strip(), year.strip(), branch.strip(), section.strip(), str(roll_number).strip()))
    conn.commit()
    updated = cursor.rowcount > 0
    conn.close()
    return updated


def get_students_by_class(degree, year, branch, section):
    init_db()
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        SELECT roll_number, name, gender, degree, year, branch, section
        FROM students
        WHERE UPPER(degree) = UPPER(?) 
          AND UPPER(year) = UPPER(?) 
          AND UPPER(branch) = UPPER(?) 
          AND UPPER(section) = UPPER(?)
        ORDER BY roll_number ASC
    ''', (degree, year, branch, section))
    results = cursor.fetchall()
    conn.close()
    return results


def add_student(roll_number, name, gender, degree, year, branch, section):
    init_db()
    local_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO students (roll_number, name, gender, degree, year, branch, section, registered_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (roll_number, name, gender, degree, year, branch, section, local_time))
        conn.commit()
        conn.close()
        return True
    except sqlite3.IntegrityError:
        return False


def delete_student(roll_number):
    """Direct deletion helper for cleanup during uncompleted registrations."""
    init_db()
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('DELETE FROM attendance_records WHERE roll_number = ?', (roll_number,))
    cursor.execute('DELETE FROM students WHERE roll_number = ?', (roll_number,))
    conn.commit()
    conn.close()
    return True


def delete_student_secure(roll_number, requesting_user, password):
    req_id = requesting_user.get("id")
    if not verify_teacher_password(req_id, password):
        return False, "Security check failed: Incorrect password!"

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('DELETE FROM attendance_records WHERE roll_number = ?', (roll_number,))
    cursor.execute('DELETE FROM students WHERE roll_number = ?', (roll_number,))
    conn.commit()
    conn.close()
    return True, f"Student {roll_number} deleted successfully."


# ATTENDANCE PERSISTENCE & AUDIT LOGS (WITH STEP 8 DEDUPLICATION LOCK)
def save_attendance_entry(subject_id, subject_code, subject_name, teacher_name, roll_number, name, branch, section, date, time_marked, status="Present"):
    init_db()
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Step 8: Deduplication check: Has student already been marked present for this subject on this date?
    cursor.execute('''
        SELECT id FROM attendance_records
        WHERE roll_number = ? AND subject_id = ? AND date = ?
    ''', (str(roll_number), subject_id, date))
    existing = cursor.fetchone()

    if existing:
        # Update timestamp to latest verification rather than creating duplicate row
        cursor.execute('''
            UPDATE attendance_records
            SET time_marked = ?, status = ?
            WHERE id = ?
        ''', (time_marked, status, existing[0]))
    else:
        cursor.execute('''
            INSERT INTO attendance_records (subject_id, subject_code, subject_name, teacher_name, roll_number, name, branch, section, date, time_marked, status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (subject_id, subject_code, subject_name, teacher_name, roll_number, name, branch, section, date, time_marked, status))
    conn.commit()
    conn.close()


def get_attendance_for_subject_date(subject_id, date):
    init_db()
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        SELECT roll_number, name, branch, section, time_marked, status
        FROM attendance_records
        WHERE subject_id = ? AND date = ?
    ''', (subject_id, date))
    records = cursor.fetchall()
    conn.close()
    return records


def get_attendance_audit_logs(limit=250):
    init_db()
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        SELECT id, subject_code, subject_name, teacher_name, roll_number, name, branch, section, date, time_marked, status
        FROM attendance_records
        ORDER BY date DESC, time_marked DESC
        LIMIT ?
    ''', (limit,))
    rows = cursor.fetchall()
    conn.close()
    logs = []
    for r in rows:
        logs.append({
            "id": r[0],
            "subject_code": r[1],
            "subject_name": r[2],
            "teacher_name": r[3],
            "roll_number": r[4],
            "name": r[5],
            "branch": r[6],
            "section": r[7],
            "date": r[8],
            "time_marked": r[9],
            "status": r[10]
        })
    return logs


def delete_attendance_log(log_id):
    init_db()
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM attendance_records WHERE id = ?", (log_id,))
    conn.commit()
    conn.close()
    return True


def get_distinct_attendance_dates():
    """Returns list of distinct dates present in attendance_records, newest first."""
    init_db()
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT DISTINCT date FROM attendance_records ORDER BY date DESC")
    dates = [r[0] for r in cursor.fetchall() if r[0]]
    conn.close()
    return dates



def delete_attendance_by_student(roll_number, requesting_user=None, password=None, date_filter=None, subject_id_filter=None):
    """
    Deletes attendance records for a particular student.
    Optionally filters by specific date or subject.
    """
    init_db()
    if requesting_user and password:
        if not verify_teacher_password(requesting_user["id"], password):
            return False, "Security check failed: Incorrect Administrator password!", 0

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    query = "DELETE FROM attendance_records WHERE UPPER(roll_number) = UPPER(?)"
    params = [str(roll_number).strip()]

    if date_filter and date_filter != "ALL":
        query += " AND date = ?"
        params.append(date_filter)

    if subject_id_filter and subject_id_filter != "ALL":
        query += " AND subject_id = ?"
        params.append(int(subject_id_filter))

    cursor.execute(query, tuple(params))
    deleted_count = cursor.rowcount
    conn.commit()
    conn.close()

    # Also clean CSV files in ATTENDANCE_DIR matching date_filter (or today if not specified)
    if ATTENDANCE_DIR.exists():
        target_dates = []
        if date_filter and date_filter != "ALL":
            target_dates.append(date_filter.replace("-", ""))
        else:
            # Check all dates or all csv files
            target_dates = None

        csv_files = list(ATTENDANCE_DIR.glob("*.csv"))
        for csv_file in csv_files:
            try:
                # If target_dates specified, check filename or file content date
                if target_dates and not any(d in csv_file.name for d in target_dates):
                    continue
                df = pd.read_csv(csv_file)
                if 'Roll Number' in df.columns:
                    original_len = len(df)
                    df = df[df['Roll Number'].astype(str).str.strip().str.upper() != str(roll_number).strip().upper()]
                    if len(df) < original_len:
                        if len(df) == 0:
                            csv_file.unlink(missing_ok=True)
                        else:
                            df.to_csv(csv_file, index=False)
            except Exception:
                pass

    admin_id = requesting_user.get("employee_id", "ADMIN") if requesting_user else "SYSTEM"
    log_security_event("ATTENDANCE_DELETED", "WARNING", admin_id,
                       f"Deleted {deleted_count} attendance records for student '{roll_number}' (Date: {date_filter or 'ALL'})")
    return True, f"Successfully deleted {deleted_count} attendance records for student {roll_number}.", deleted_count


def delete_attendance_by_batch(branch, section, date_filter=None, subject_id_filter=None, requesting_user=None, password=None):
    """
    Deletes attendance records for a particular batch (branch and section).
    Optionally filters by date or subject. Also purges matching entries from CSV records.
    """
    init_db()
    if requesting_user and password:
        if not verify_teacher_password(requesting_user["id"], password):
            return False, "Security check failed: Incorrect Administrator password!", 0

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    query = "DELETE FROM attendance_records WHERE UPPER(branch) = UPPER(?) AND UPPER(section) = UPPER(?)"
    params = [str(branch).strip(), str(section).strip()]

    if date_filter and date_filter != "ALL":
        query += " AND date = ?"
        params.append(date_filter)

    if subject_id_filter and subject_id_filter != "ALL":
        query += " AND subject_id = ?"
        params.append(int(subject_id_filter))

    cursor.execute(query, tuple(params))
    deleted_count = cursor.rowcount
    conn.commit()
    conn.close()

    # Also clean CSV files in ATTENDANCE_DIR matching branch, section, and date_filter
    if ATTENDANCE_DIR.exists():
        target_dates = []
        if date_filter and date_filter != "ALL":
            target_dates.append(date_filter.replace("-", ""))
        else:
            target_dates = None

        csv_files = list(ATTENDANCE_DIR.glob("*.csv"))
        for csv_file in csv_files:
            try:
                if target_dates and not any(d in csv_file.name for d in target_dates):
                    continue
                # If filename has exact matching branch and section pattern, e.g. Attendance_CS-303_CSE_C_20260909_111010.csv
                fname_upper = csv_file.name.upper()
                batch_tag = f"_{branch.strip().upper()}_{section.strip().upper()}_"
                if batch_tag in fname_upper:
                    csv_file.unlink(missing_ok=True)
                    continue

                # Or inspect rows inside CSV
                df = pd.read_csv(csv_file)
                if 'Branch' in df.columns and 'Section' in df.columns:
                    original_len = len(df)
                    mask = (df['Branch'].astype(str).str.strip().str.upper() == str(branch).strip().upper()) & \
                           (df['Section'].astype(str).str.strip().str.upper() == str(section).strip().upper())
                    df = df[~mask]
                    if len(df) < original_len:
                        if len(df) == 0:
                            csv_file.unlink(missing_ok=True)
                        else:
                            df.to_csv(csv_file, index=False)
            except Exception:
                pass

    admin_id = requesting_user.get("employee_id", "ADMIN") if requesting_user else "SYSTEM"
    log_security_event("ATTENDANCE_BATCH_DELETED", "WARNING", admin_id,
                       f"Deleted {deleted_count} attendance records for Batch '{branch}-{section}' (Date: {date_filter or 'ALL'})")
    return True, f"Successfully deleted {deleted_count} attendance records for Batch {branch}-{section}.", deleted_count



# SYSTEM DIAGNOSTICS
def get_system_diagnostics():
    init_db()
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM students")
    student_count = cursor.fetchone()[0]
    cursor.execute("SELECT COUNT(*) FROM teachers")
    teacher_count = cursor.fetchone()[0]
    cursor.execute("SELECT COUNT(*) FROM subjects")
    subject_count = cursor.fetchone()[0]
    cursor.execute("SELECT COUNT(*) FROM attendance_records")
    attendance_count = cursor.fetchone()[0]
    conn.close()

    db_size_kb = DB_PATH.stat().st_size / 1024 if DB_PATH.exists() else 0
    emb_count = 0
    if EMB_PATH.exists():
        try:
            with open(EMB_PATH, "rb") as f:
                emb_count = len(pickle.load(f))
        except:
            pass

    scrfd_candidates = [
        BASE_DIR / "models" / "buffalo_m" / "det_2.5g.onnx",
        Path.home() / ".insightface" / "models" / "buffalo_m" / "det_2.5g.onnx"
    ]
    scrfd_exists = any(p.exists() for p in scrfd_candidates)

    arcface_candidates = [
        BASE_DIR / "models" / "buffalo_m" / "w600k_r50.onnx",
        Path.home() / ".insightface" / "models" / "buffalo_m" / "w600k_r50.onnx",
        BASE_DIR / "models" / ".deepface" / "weights" / "arcface_weights.h5",
        Path.home() / ".deepface" / "weights" / "arcface_weights.h5"
    ]
    arcface_exists = any(p.exists() for p in arcface_candidates)

    return {
        "students": student_count,
        "teachers": teacher_count,
        "subjects": subject_count,
        "attendance_records": attendance_count,
        "db_size_kb": round(db_size_kb, 2),
        "embeddings_count": emb_count,
        "scrfd_weights": scrfd_exists,
        "arcface_weights": arcface_exists
    }


# ACCOUNT UNLOCK & SECURITY CONTROLS
def unlock_teacher_account(teacher_id, admin_user=None):
    init_db()
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT employee_id, name FROM teachers WHERE id = ?", (teacher_id,))
    row = cursor.fetchone()
    if not row:
        conn.close()
        return False, "Teacher account not found!"
    emp_id, name = row
    cursor.execute("UPDATE teachers SET failed_login_attempts = 0, locked_until = NULL WHERE id = ?", (teacher_id,))
    conn.commit()
    conn.close()

    admin_tag = f"by Admin {admin_user['employee_id']}" if admin_user else "by System"
    log_security_event("ACCOUNT_UNLOCKED", "INFO", emp_id, f"Account {emp_id} ({name}) was unlocked {admin_tag}.")
    return True, f"Account for {name} ({emp_id}) has been unlocked successfully!"


# ADMIN MASTER AUTHORIZATION KEY MANAGEMENT
def verify_admin_master_key(candidate_key):
    init_db()
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT setting_value FROM system_settings WHERE setting_key = 'admin_master_key'")
    row = cursor.fetchone()
    conn.close()
    if not row:
        stored_key = "SmartClass@Admin#2026"
    else:
        stored_key = row[0]
    return secrets.compare_digest(candidate_key.strip(), stored_key.strip())


def get_admin_master_key():
    init_db()
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT setting_value FROM system_settings WHERE setting_key = 'admin_master_key'")
    row = cursor.fetchone()
    conn.close()
    return row[0] if row else "SmartClass@Admin#2026"


def update_admin_master_key(new_key, requesting_user=None):
    if len(new_key.strip()) < 8:
        return False, "Master Key must be at least 8 characters long!"
    init_db()
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        INSERT OR REPLACE INTO system_settings (setting_key, setting_value, updated_at)
        VALUES ('admin_master_key', ?, ?)
    ''', (new_key.strip(), datetime.now().isoformat()))
    conn.commit()
    conn.close()

    user_tag = requesting_user["employee_id"] if requesting_user else "SYSTEM"
    log_security_event("MASTER_KEY_ROTATED", "CRITICAL", user_tag, "Admin Master Authorization Key was rotated.")
    return True, "Admin Master Authorization Key updated successfully!"


# CRYPTOGRAPHIC ATTENDANCE FILE CHECKSUMS (TAMPER-PROOFING)
def generate_file_checksum(filepath, created_by="System"):
    path = Path(filepath)
    if not path.exists():
        return None, None

    with open(path, "rb") as f:
        content = f.read()

    file_hash = hashlib.sha256(content).hexdigest()
    hmac_sig = hmac.new(SYSTEM_HMAC_SECRET, content, hashlib.sha256).hexdigest()

    init_db()
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        INSERT OR REPLACE INTO attendance_checksums (filename, file_hash, hmac_signature, created_by, created_at)
        VALUES (?, ?, ?, ?, ?)
    ''', (path.name, file_hash, hmac_sig, created_by, datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    conn.commit()
    conn.close()

    log_security_event("ATTENDANCE_SIGNED", "INFO", created_by, f"Cryptographic HMAC signature registered for {path.name}")
    return file_hash, hmac_sig


def verify_file_integrity(filepath):
    path = Path(filepath)
    if not path.exists():
        return False, "File does not exist on disk."

    with open(path, "rb") as f:
        content = f.read()

    current_hash = hashlib.sha256(content).hexdigest()
    current_hmac = hmac.new(SYSTEM_HMAC_SECRET, content, hashlib.sha256).hexdigest()

    init_db()
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT file_hash, hmac_signature, created_by, created_at FROM attendance_checksums WHERE filename = ?", (path.name,))
    row = cursor.fetchone()
    conn.close()

    if not row:
        log_security_event("TAMPER_CHECK_FAILED", "WARNING", "System", f"Integrity check failed: {path.name} is not in checksum database.")
        return False, f"UNREGISTERED: File '{path.name}' has no official cryptographic signature registered in the database."

    stored_hash, stored_hmac, creator, created_at = row

    if secrets.compare_digest(current_hash, stored_hash) and secrets.compare_digest(current_hmac, stored_hmac):
        log_security_event("INTEGRITY_VERIFIED", "INFO", "System", f"Integrity verified for {path.name} (signed by {creator}).")
        return True, f"VERIFIED OFFICIAL RECORD\n\n• File: {path.name}\n• Signed By: {creator}\n• Timestamp: {created_at}\n• HMAC SHA-256 Signature Matches 100%!"
    else:
        log_security_event("TAMPER_DETECTED", "CRITICAL", "System", f"TAMPER ALERT: Checksum mismatch on {path.name}!")
        return False, f"🚨 CRITICAL WARNING: FILE TAMPER DETECTED!\n\n• File: {path.name}\n• Current Hash: {current_hash[:16]}...\n• Stored Hash:  {stored_hash[:16]}...\n\nThe content of this CSV has been modified outside the SmartClass system!"


# SECURITY AUDIT TRAIL QUERIES
def get_security_audit_logs(limit=250, severity_filter=None, event_filter=None):
    init_db()
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    query = "SELECT id, timestamp, event_type, severity, user_identifier, ip_or_host, details FROM security_audit_logs"
    params = []
    conditions = []

    if severity_filter and severity_filter != "ALL":
        conditions.append("severity = ?")
        params.append(severity_filter)
    if event_filter and event_filter != "ALL":
        conditions.append("event_type = ?")
        params.append(event_filter)

    if conditions:
        query += " WHERE " + " AND ".join(conditions)

    query += " ORDER BY id DESC LIMIT ?"
    params.append(limit)

    cursor.execute(query, params)
    rows = cursor.fetchall()
    conn.close()

    logs = []
    for r in rows:
        logs.append({
            "id": r[0],
            "timestamp": r[1],
            "event_type": r[2],
            "severity": r[3],
            "user_identifier": r[4],
            "ip_or_host": r[5],
            "details": r[6]
        })
    return logs


# DISASTER RECOVERY & ENCRYPTED BACKUP
def create_database_backup():
    init_db()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    target_backup_dir = BACKUP_DIR / f"Backup_{ts}"
    target_backup_dir.mkdir(parents=True, exist_ok=True)

    try:
        shutil.copy2(DB_PATH, target_backup_dir / "smartclass.db")
        if EMB_PATH.exists():
            shutil.copy2(EMB_PATH, target_backup_dir / "embeddings.pkl")

        # Save metadata manifest
        manifest_path = target_backup_dir / "backup_manifest.txt"
        with open(manifest_path, "w", encoding="utf-8") as f:
            f.write(f"SmartClass Vision Enterprise Backup\n")
            f.write(f"Created: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Host: {get_client_host()}\n")
            f.write(f"DB Size: {round(DB_PATH.stat().st_size / 1024, 2)} KB\n")

        log_security_event("BACKUP_CREATED", "INFO", "System", f"Database backup created at {target_backup_dir.name}")
        return True, f"Backup created successfully: {target_backup_dir.name}"
    except Exception as e:
        log_security_event("BACKUP_FAILED", "CRITICAL", "System", f"Backup failed: {str(e)}")
        return False, f"Backup failed: {str(e)}"


def get_backup_list():
    init_db()
    if not BACKUP_DIR.exists():
        return []
    backups = []
    for d in sorted(BACKUP_DIR.glob("Backup_*"), reverse=True):
        if d.is_dir():
            db_file = d / "smartclass.db"
            sz = round(db_file.stat().st_size / 1024, 2) if db_file.exists() else 0
            backups.append({
                "folder_name": d.name,
                "path": str(d),
                "size_kb": sz
            })
    return backups