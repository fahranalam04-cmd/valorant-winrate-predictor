@echo off
REM Back the database up: a full copy to %USERPROFILE%\valwr-backups and, when
REM OneDrive is present, a slim copy (no raw API responses) to
REM OneDrive\valwr-backups. Every copy is checked before older ones are pruned.
REM The weekly scheduled task runs the same thing with --after-game.
title valwr backup
cd /d "%~dp0"
.venv\Scripts\python.exe -m valwr.store.backup %*
echo.
pause
