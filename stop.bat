@echo off
REM Stops any running valwr crawler. Progress is in SQLite; nothing is lost.
REM
REM Matched on the command line, never on the process name alone. An earlier
REM version ran `taskkill /IM pythonw.exe`, which ended every windowless Python
REM program on the machine, valwr or not. PowerShell's CIM cmdlets rather than
REM wmic, which current Windows 11 builds no longer ship -- there the old lines
REM failed silently and still printed "Crawler stopped".
title stop valwr crawler
powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "$p = @(Get-CimInstance Win32_Process -Filter \"Name LIKE 'python%%.exe'\" | Where-Object { $_.CommandLine -match 'valwr\.collect' });" ^
  "$p | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue };" ^
  "if ($p.Count) { Write-Host ('Stopped ' + $p.Count + ' crawler process(es). Progress is saved.') } else { Write-Host 'No crawler was running.' }"
pause
