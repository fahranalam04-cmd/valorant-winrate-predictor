@echo off
REM Live crawl monitor. Safe to open, close and reopen -- it only reads the
REM database, so it never touches the crawl itself.
title valwr crawl monitor
cd /d "%~dp0"
.venv\Scripts\python.exe -m valwr.collect.watch
pause
