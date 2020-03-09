#include <cuda_runtime.h>
#include "Define.h"

#pragma once

void InitializePlugin();
void ClearPlugin();

int __declspec(dllexport) __stdcall GetDeviceCount();
int __declspec(dllexport) __stdcall GetMaxThreadPerBlock(int device);
int __declspec(dllexport) __stdcall GetMaxThreadPerBlockInAllDevice();

struct OptimalLaunchParam
{
	int itemCount;
	dim3 blockCountInGrid;
	dim3 threadCountinBlock;
};

__host__ void GetOptimalBlockAndThreadDim(IN int deviceCount, IN int warpCount, OUT OptimalLaunchParam& outParam);
