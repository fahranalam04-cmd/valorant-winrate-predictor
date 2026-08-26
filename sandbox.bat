@echo off
title valwr sandbox
cd /d "%~dp0"

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
goto menu

:tests
.venv\Scripts\python.exe -m pytest test\test_sandbox.py -q
goto done

:listall
.venv\Scripts\python.exe -m valwr.sandbox list
goto done

:shrink
echo   A player with 3 games at 100%% against one with 500 games at 55%%.
echo   If shrinkage works, the 3-game player should NOT be favoured.
echo.
.venv\Scripts\python.exe -m valwr.sandbox run --scenario shrinkage
goto done

:smurf
.venv\Scripts\python.exe -m valwr.sandbox run --scenario single_smurf
goto done

:catalog
.venv\Scripts\python.exe -m valwr.sandbox run --scenario all
goto done

:gbm
echo   Two identical teams. A fair model must say 50%%.
echo.
echo   --- shipped linear model ---
.venv\Scripts\python.exe -m valwr.sandbox --model logistic run --scenario fair_match
echo.
echo   --- gradient booster ---
.venv\Scripts\python.exe -m valwr.sandbox --model gbm run --scenario fair_match
goto done

:sweeprating
.venv\Scripts\python.exe -m valwr.sandbox sweep --feature rating
goto done

:sweepwr
echo   Higher win rate should mean a higher predicted chance of winning.
echo   Watch what actually happens.
echo.
.venv\Scripts\python.exe -m valwr.sandbox sweep --feature wr
goto done

:variance
.venv\Scripts\python.exe -m valwr.sandbox run --mode variance --scenario single_smurf --samples 100 --seed 42
goto done

:compare
.venv\Scripts\python.exe -m valwr.sandbox compare
goto done

:custom
echo   Examples: fair_match  bad_map  five_stack_vs_solos  contra_rank_vs_rating
echo   Or a category: carry  coverage  boundary  contradiction  distribution
echo.
set "target="
set /p target=Scenario or category:
if "%target%"=="" goto menu
.venv\Scripts\python.exe -m valwr.sandbox run --scenario %target%
goto done

:done
echo.
echo   ---------------------------------------------------------------
pause
goto menu
