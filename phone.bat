@echo off
title valwr dashboard (network)
cd /d "%~dp0"

set "PY=.venv\Scripts\python.exe"
if not exist "%PY%" goto novenv

cls
echo.
echo   ===============================================================
echo     VALWR LIVE DASHBOARD  --  readable from your phone
echo   ===============================================================
echo.
echo   Same dashboard as dashboard.bat, bound to your whole network
echo   instead of just this machine, so a phone on the same wifi can
echo   open it.
echo.
echo   WHAT THIS EXPOSES: the match you are currently in, including
echo   the other players' gamertags and their stored statistics.
echo   Anything on this network can read it while this window is open.
echo   Use it on a network you control, and close this when you are
echo   done. dashboard.bat is the local-only version.
echo.
echo   Read-only: never writes to the game, never picks agents.
echo.

REM Every address this machine answers on, so you can pick the wifi one
REM rather than guessing. Loopback and link-local are filtered out.
echo   Open one of these on your phone:
for /f "tokens=2 delims=:" %%A in ('ipconfig ^| findstr /c:"IPv4 Address"') do (
    for /f "tokens=* delims= " %%B in ("%%A") do echo       http://%%B:8787/
)
echo.

%PY% tools\preflight.py --wait 0
if errorlevel 3 goto broken
if errorlevel 1 if not errorlevel 2 goto broken

echo.
echo   Starting. Press Ctrl+C here to stop.
echo   ---------------------------------------------------------------
%PY% -m valwr.dash --host 0.0.0.0 --no-browser %*
echo   ---------------------------------------------------------------
echo.
echo   Dashboard stopped. It is no longer reachable from the network.
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
