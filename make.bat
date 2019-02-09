@echo off
SETLOCAL
SET dataset=0
SET app_name=ag.exe
SET source=main.cu kernels.cu structures.cpp desicion_maker.c
SET __defines=
if "%~1" == "" goto :command
if "%1" == "clean" goto :clean

:loop_args
if "%1" == "debug" set __defines=%__defines% -DDEBUG
if "%1" == "test" set __defines=%__defines% -DTEST
if "%1" == "csr" set __defines=%__defines% -DCSR_VALIDATION
if "%1" == "-dataset" goto :set_dataset
if "%1" == "-name" goto :set_name
shift
if NOT "%~1"=="" goto :loop_args

:command
echo %source% %__defines%
nvcc -o %app_name% %source% %__defines% -DDATASET_INDEX=%dataset%
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

:clean
echo clean
del *.lib *.exp *.exe
ENDLOCAL