@echo off
title valwr dashboard
cd /d "%~dp0"

set "PY=.venv\Scripts\python.exe"
if not exist "%PY%" goto novenv

cls
echo.
echo   ===============================================================
echo     VALWR LIVE DASHBOARD
echo   ===============================================================
echo.
echo   Opens a page in your browser showing the current match:
echo   both win probabilities, both teams ranked, and what drove it.
echo.
echo   Local only. Bound to 127.0.0.1, not reachable from your network.
echo   Read-only: never writes to the game, never picks agents.
echo.

REM Informational only -- unlike live.bat this does NOT wait for the game.
REM The page itself says when VALORANT is not running, and picks it up when
REM you launch it, so there is no reason to block here.
%PY% tools\preflight.py --wait 0
if errorlevel 3 goto broken
if errorlevel 2 goto start
if errorlevel 1 goto broken

:start
echo.
echo   Starting. Your browser should open automatically.
echo   Press Ctrl+C here to stop the dashboard.
echo.
echo   ---------------------------------------------------------------
%PY% -m valwr.dash %*
echo   ---------------------------------------------------------------
echo.
echo   Dashboard stopped.
pause
exit /b 0

:broken
echo.
echo   Fix the items listed above, then run this again.
echo.
pause
exit /b 1

:novenv
echo.
echo   Cannot find %PY%
echo.
echo   Create the virtual environment first:
echo       python -m venv .venv
echo       .venv\Scripts\pip install -e .
echo.
pause
exit /b 3
