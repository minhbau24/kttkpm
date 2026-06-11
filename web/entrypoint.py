import os
import shutil
import subprocess
import sys
from pathlib import Path

def main():
    database_path = os.environ.get("DATABASE_PATH")
    if database_path:
        db_path = Path(database_path)
        db_path.parent.mkdir(parents=True, exist_ok=True)
        if not db_path.exists():
            template_path = Path("/app/web/db.sqlite3")
            if template_path.exists():
                print(f"Database not found at volume mount {db_path}. Copying pre-seeded template...")
                shutil.copy(template_path, db_path)
                print("Database template copied successfully.")
            else:
                print("Database template not found at /app/web/db.sqlite3. An empty database will be initialized.")

    # Run migrations (safe and good practice on startup)
    print("Running database migrations...")
    subprocess.run([sys.executable, "manage.py", "migrate", "--noinput"])

    # Collect static files for production
    print("Collecting static files...")
    subprocess.run([sys.executable, "manage.py", "collectstatic", "--noinput"])

    # Start gunicorn WSGI server
    print("Starting Gunicorn WSGI server...")
    gunicorn_cmd = [
        "gunicorn",
        "ecom_project.wsgi:application",
        "--bind", "0.0.0.0:8000",
        "--workers", "3",
        "--timeout", "120"
    ]
    os.execvp("gunicorn", gunicorn_cmd)

if __name__ == "__main__":
    main()
