@echo off
REM Stops any running valwr crawler. Progress is in SQLite; nothing is lost.
title stop valwr crawler
taskkill /F /FI "IMAGENAME eq pythonw.exe" /FI "WINDOWTITLE eq *" >nul 2>&1
wmic process where "name='pythonw.exe' and commandline like '%%valwr%%'" delete >nul 2>&1
wmic process where "name='python.exe' and commandline like '%%valwr.collect%%'" delete >nul 2>&1
echo Crawler stopped. Progress is saved.
pause
