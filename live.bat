@echo off
REM Live win-rate prediction. Detects the match you are in and predicts it.
REM Read-only: never writes to the game, never picks agents.
title valwr live
cd /d "%~dp0"
.venv\Scripts\python.exe -m valwr.live
echo.
pause
