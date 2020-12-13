#pragma warning( disable : 4819 )

#include <cuda_runtime.h>
#include "Define.h"

#pragma once

namespace RadGrabber
{
	void InitializePlugin();
	void ClearPlugin();

	int __declspec(dllexport) __stdcall GetDeviceCount();
	int __declspec(dllexport) __stdcall GetMaxThreadPerBlock(int device);
	int __declspec(dllexport) __stdcall GetMaxThreadPerBlockInAllDevice();

	struct OptimalLaunchParam
	{
		dim3 blockCountInGrid;
		dim3 threadCountinBlock;

		int GetMaxThreadCount()
		{
			return blockCountInGrid.x * blockCountInGrid.y * blockCountInGrid.z * threadCountinBlock.x * threadCountinBlock.y * threadCountinBlock.z;
		}
	};

	__host__ void GetOptimalBlockAndThreadDim(IN int deviceIndex, OUT OptimalLaunchParam& outParam);
	__host__ void GetOptimalBufferItemCount(IN int deviceIndex, OUT int& size);
}
