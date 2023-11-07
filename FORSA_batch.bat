@echo off

rem set CONDAPATH=/cygdrive/d/SOFTWARES/Anaconda_Installed/envs/pytorch_nemo_07/python
set CONDAPATH=D:\SOFTWARES\Anaconda_Installed
rem Define here the name of the environment
set ENVNAME=TorchGPU

rem The following command activates the base environment.
rem call C:\ProgramData\Miniconda3\Scripts\activate.bat C:\ProgramData\Miniconda3
if %ENVNAME%==base (set ENVPATH=%CONDAPATH%) else (set ENVPATH=%CONDAPATH%\envs\%ENVNAME%)
rem Activate the conda environment
rem Using call is required here, see: https://stackoverflow.com/questions/24678144/conda-environments-and-bat-files
call %CONDAPATH%\Scripts\activate.bat %ENVPATH%

rem Run a python script in that environment
rem python NEMO_main_CIFAR10.py --ptq True --epoch 25 --lr 0.0001 --init mnist_cnn_fp.pt --qat True --bit %%q  --pretrain True
FOR %%q IN (4) DO (
	FOR %%f IN (3 2) DO (
		python D:\PhD_work\Thesis\FORSA\main_LeNet_FORSA.py --bit %%q --frac_size %%f
    )
)
rem Deactivate the environment
call conda deactivate

pause