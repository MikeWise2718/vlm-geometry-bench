@echo off
REM Manual VLM testing with Ollama - replicates benchmark prompts
REM Usage: manual-test.cmd [model] [testsuite-path]
REM Default model: llava:7b
REM Default testsuite: D:/python/imagegen/testsuite
REM
REM Uses OLLAMA_HOST from environment to select Ollama server.
REM NAS copy: snorlax:/share/transfer/vlm-geometry-bench/testsuite/
REM
REM In the ollama interactive prompt, just include the image path in your message:
REM   Describe this image D:/python/imagegen/testsuite/SOME_IMAGE.png

set "MODEL=%~1"
if "%MODEL%"=="" set "MODEL=llava:7b"
set "TS=%~2"
if "%TS%"=="" set "TS=D:/python/imagegen/testsuite"

echo ============================================================
echo  VLM Geometry Bench - Manual Testing
echo  Model: %MODEL%
echo  Testsuite: %TS%
echo  OLLAMA_HOST: %OLLAMA_HOST%
echo ============================================================
echo.
echo SAMPLE IMAGES (copy paths for use in prompts):
echo.
echo   Easy:
echo     %TS%/CTRL_empty_wb.png         (0 spots, empty)
echo     %TS%/CTRL_single_wb.png        (1 spot)
echo.
echo   Medium:
echo     %TS%/USSS_s3_d20_wb.png        (20 random spots, size 3)
echo     %TS%/USSS_s3_d50_wb.png        (50 random spots, size 3)
echo     %TS%/HSFR_sp12_wb.png          (45 hex pattern spots)
echo.
echo   Hard:
echo     %TS%/USSS_s3_d100_wb.png       (100 random spots, size 3)
echo     %TS%/HSDN_sp12_d10_n5_wb.png   (hex with defects+noise)
echo.
echo ============================================================
echo  TASK PROMPTS (paste these, adding image path at the end)
echo ============================================================
echo.
echo --- COUNT (expected: see spot counts above) ---
echo Examine this image and count the circular spots or dots. Count ALL visible circular spots/dots. Include spots of any size. Do not count partial spots at edges. Respond with ONLY a single integer number. How many spots are in this image? %TS%/USSS_s3_d50_wb.png
echo.
echo --- PATTERN (expected: EMPTY, SINGLE, RANDOM, HEXAGONAL) ---
echo Examine this image and classify the arrangement pattern of the spots. Choose ONE of: EMPTY, SINGLE, RANDOM, HEXAGONAL, GRID. Respond with ONLY one word. What is the pattern type? %TS%/HSFR_sp12_wb.png
echo.
echo --- LOCATE ---
echo Examine this image and identify the location of each circular spot. Use normalized coordinates where (0.0, 0.0) is top-left and (1.0, 1.0) is bottom-right. Format: x, y per line. List all spot coordinates: %TS%/CTRL_single_wb.png
echo.
echo --- DEFECT (use with HSDN images) ---
echo Examine this image showing spots in a hexagonal pattern. Look for MISSING (gaps), NOISE (extra spots), DISPLACEMENT (shifted spots). Respond: DEFECTS_FOUND: YES/NO, MISSING_COUNT: N, NOISE_COUNT: N, CONFIDENCE: HIGH/MEDIUM/LOW %TS%/HSDN_sp12_d10_n5_wb.png
echo.
echo ============================================================
echo  Just copy-paste a line above into the ollama prompt.
echo  The image path at the end tells ollama to load that image.
echo ============================================================
echo.
echo Starting ollama with %MODEL%...
echo.

ollama run %MODEL%
