@echo off
REM Restarts the crawler if it has stopped. Registered with Task Scheduler to
REM run every 10 minutes. Safe to run manually; it does nothing when healthy.
cd /d "%~dp0"
.venv\Scripts\pythonw.exe -m valwr.collect.watchdog --hours 12
