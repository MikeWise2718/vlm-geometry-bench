@echo off
REM Pull vlm-geometry-bench results from QNAP NAS (snorlax)
REM Usage: pull-results.cmd

set "REMOTE_SRC=snorlax:/share/transfer/vlm-geometry-bench/results/"
set "LOCAL_RESULTS=D:/python/vlm-geometry-bench/results/"

echo Pulling results from NAS...
rsync -avz --progress "%REMOTE_SRC%" "%LOCAL_RESULTS%"

if %ERRORLEVEL% EQU 0 (
    echo Pull complete.
) else (
    echo Pull failed with error %ERRORLEVEL%
)
