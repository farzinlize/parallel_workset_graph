@echo off
SETLOCAL
SET version=3
SET dataset=0
SET app_name=ag
SET source=main.cu kernels.cu structures.cpp sequential.cpp desicion_maker.c fuzzy_timing.c report.cpp
SET options=
SET __defines=-DWINDOWS
if "%~1" == "" goto :command
if "%1" == "clean" goto :clean

:loop_args
if "%1" == "debug" set __defines=%__defines% -DDEBUG
if "%1" == "test" set __defines=%__defines% -DTEST
if "%1" == "csr" set __defines=%__defines% -DCSR_VALIDATION
if "%1" == "detail" set __defines=%__defines% -DDETAIL
if "%1" == "dp" goto :dynamic_parallelism
if "%1" == "nvGraph" goto :nvGraph
if "%1" == "-dataset" goto :set_dataset
if "%1" == "-name" goto :set_name
shift
if NOT "%~1"=="" goto :loop_args

:command
echo %source% %__defines% %options%
nvcc -o build/%app_name%_%version%.exe %source% %options% %__defines% -DDATASET_INDEX=%dataset% -DVERSION=%version%
ENDLOCAL
goto :eof

:set_dataset
shift
set dataset=%1
goto :loop_args

:set_name
shift
set app_name="%1"
goto :loop_args

:dynamic_parallelism
shift
set options=%options% -arch=sm_35 -rdc=true -lcudadevrt
set __defines=%__defines% -DDP
set app_name=ag_dp
goto :loop_args

:nvGraph
shift
set options=%options% -lnvgraph
set __defines=%__defines% -DNVG
goto :loop_args

:clean
echo clean
del /Q build\*
ENDLOCAL