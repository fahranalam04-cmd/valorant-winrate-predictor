@echo off
REM Overnight crawl. Runs independently of any editor or Claude session.
REM Safe to close: all progress lives in SQLite and resumes on restart.
title valwr crawler
cd /d "%~dp0"
.venv\Scripts\python.exe -u -m valwr.collect --minutes 600 --seed none
echo.
echo Crawl finished. Press any key to close.
pause
