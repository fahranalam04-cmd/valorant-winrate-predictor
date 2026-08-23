@echo off
REM Supervised overnight crawl. Restarts itself on any failure and logs to
REM data\crawl.log. Closing this window DOES stop it -- use the Scheduled
REM Task if you need it to survive that.
title valwr crawler (supervised)
cd /d "%~dp0"
.venv\Scripts\python.exe -u -m valwr.collect.supervise --hours 10
echo.
echo Crawl finished. Press any key to close.
pause
