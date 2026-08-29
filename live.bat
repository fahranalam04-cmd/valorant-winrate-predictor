@echo off
title valwr live
cd /d "%~dp0"

set "PY=.venv\Scripts\python.exe"

REM Without this you get cmd's bare "The system cannot find the file
REM specified.", which says nothing about what to do.
if not exist "%PY%" goto novenv

cls
echo.
echo   ===============================================================
echo     VALWR LIVE  --  win prediction and player ranking
echo   ===============================================================
echo.
echo   Read-only. This never writes to the game, never picks agents,
echo   and never touches game memory.
echo.

REM Waits up to 5 minutes so you can click this first, then launch the
REM game. Exit 1 means something needs fixing; 2 means no game.
REM Override with:  set VALWR_WAIT=0 & live.bat   (skip the wait)
if not defined VALWR_WAIT set "VALWR_WAIT=300"
%PY% tools\preflight.py --wait %VALWR_WAIT%
if errorlevel 2 goto nogame
if errorlevel 1 goto broken

echo.
echo   Watching for a match. Queue, or start a custom game.
echo   Custom games work: you can start one alone to test.
echo   Press Ctrl+C to stop.
echo.
echo   ---------------------------------------------------------------
%PY% -m valwr.live %*
echo   ---------------------------------------------------------------
echo.
echo   Stopped.
pause
exit /b 0

:nogame
echo.
echo   VALORANT is not running, so there is nothing to watch.
echo   Start the game, then run this again.
echo.
pause
exit /b 2

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
