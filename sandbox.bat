@echo off
title valwr sandbox
cd /d "%~dp0"

set "PY=.venv\Scripts\python.exe"
set "ONESHOT="
set "empties=0"

REM Without this you get cmd's bare "The system cannot find the file
REM specified.", which is what made a failing run so hard to read.
if not exist "%PY%" goto novenv

REM An argument runs one option and exits: `sandbox.bat 5`, or
REM `sandbox.bat 11 fair_match`. Without it the menu can only be driven by
REM hand, which is why it went untested.
if not "%~1"=="" set "ONESHOT=1"
if defined ONESHOT set "choice=%~1"
if defined ONESHOT goto dispatch

:menu
cls
echo.
echo   ===============================================================
echo     VALWR PREDICTION SANDBOX
echo   ===============================================================
echo.
echo     1   Run the tests                 27 invariants, ~9 sec
echo     2   List scenarios and coverage   what exists, 52/52 features
echo.
echo     --- see it work ------------------------------------------
echo     3   Shrinkage demo                3 games at 100%% vs 500 at 55%%
echo     4   Single smurf, full detail     rosters, factors, mirror
echo     5   Full catalog                  all 159 scenarios, ~20 sec
echo.
echo     --- the findings -----------------------------------------
echo     6   Linear vs gradient booster    the symmetry bug
echo     7   Rating sweep                  a feature that works
echo     8   Win-rate sweep                the inverted one
echo.
echo     --- deeper ------------------------------------------------
echo     9   Variance, 100 samples         does noise flip it
echo    10   Benchmark compare             what changed since last save
echo    11   Pick your own scenario
echo.
echo     0   Exit
echo.
set "choice="
set /p choice=Choose a number then press Enter:
echo.

REM `set /p` leaves the variable untouched at end-of-file, so a piped or
REM redirected run used to spin here forever -- 1,052 redraws in 15 seconds
REM before a timeout killed it. Three empty reads means nobody is typing.
if not "%choice%"=="" goto dispatch
set /a empties+=1
if %empties% geq 3 goto noinput
goto menu

:dispatch
set "empties=0"
if "%choice%"=="1"  goto tests
if "%choice%"=="2"  goto listall
if "%choice%"=="3"  goto shrink
if "%choice%"=="4"  goto smurf
if "%choice%"=="5"  goto catalog
if "%choice%"=="6"  goto gbm
if "%choice%"=="7"  goto sweeprating
if "%choice%"=="8"  goto sweepwr
if "%choice%"=="9"  goto variance
if "%choice%"=="10" goto compare
if "%choice%"=="11" goto custom
if "%choice%"=="0"  exit /b 0
if defined ONESHOT goto badarg
goto menu

:tests
%PY% -m pytest test\test_sandbox.py -q
goto done

:listall
%PY% -m valwr.sandbox list
goto done

:shrink
echo   A player with 3 games at 100%% against one with 500 games at 55%%.
echo   If shrinkage works, the 3-game player should NOT be favoured.
echo.
%PY% -m valwr.sandbox run --scenario shrinkage
goto done

:smurf
%PY% -m valwr.sandbox run --scenario single_smurf
goto done

:catalog
%PY% -m valwr.sandbox run --scenario all
goto done

:gbm
echo   Two identical teams. A fair model must say 50%%.
echo.
echo   --- shipped linear model ---
%PY% -m valwr.sandbox --model logistic run --scenario fair_match
echo.
echo   --- gradient booster ---
%PY% -m valwr.sandbox --model gbm run --scenario fair_match
goto done

:sweeprating
%PY% -m valwr.sandbox sweep --feature rating
goto done

:sweepwr
echo   Higher win rate should mean a higher predicted chance of winning.
echo   Watch what actually happens.
echo.
%PY% -m valwr.sandbox sweep --feature wr
goto done

:variance
%PY% -m valwr.sandbox run --mode variance --scenario single_smurf --samples 100 --seed 42
goto done

:compare
%PY% -m valwr.sandbox compare
goto done

:custom
echo   Examples: fair_match  bad_map  five_stack_vs_solos  contra_rank_vs_rating
echo   Or a category: carry  coverage  boundary  contradiction  distribution
echo.
set "target=%~2"
if not "%target%"=="" goto runcustom
set /p target=Scenario or category:
if not "%target%"=="" goto runcustom
if defined ONESHOT exit /b 1
goto menu
:runcustom
REM Quoted: the name is typed at a prompt, and unquoted a "&" in it would run
REM whatever followed as a second command.
%PY% -m valwr.sandbox run --scenario "%target%"
goto done

:done
echo.
echo   ---------------------------------------------------------------
if defined ONESHOT exit /b 0
pause
goto menu

:badarg
echo   Unknown option "%choice%". Valid options are 0-11.
exit /b 1

:noinput
echo   No input received -- exiting. Run sandbox.bat with a number to
echo   pick an option directly, for example:  sandbox.bat 2
exit /b 0

:novenv
echo.
echo   Cannot find %PY%
echo.
echo   Create the virtual environment first:
echo       python -m venv .venv
echo       .venv\Scripts\pip install -e .
echo.
if not "%~1"=="" exit /b 3
pause
exit /b 3
