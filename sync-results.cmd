@echo off
REM Sync vlm-geometry-bench results to QNAP NAS (snorlax)
REM Usage: sync-results.cmd

set "LOCAL_RESULTS=D:/python/vlm-geometry-bench/results/"
set "REMOTE_DEST=snorlax:/share/transfer/vlm-geometry-bench/results/"

echo Syncing results to NAS...
rsync -avz --progress "%LOCAL_RESULTS%" "%REMOTE_DEST%"

if %ERRORLEVEL% EQU 0 (
    echo Sync complete.
) else (
    echo Sync failed with error %ERRORLEVEL%
)
