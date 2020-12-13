#include <cuda_runtime.h>
#include <cstdlib>

#include "Define.h"
#include "DeviceConfig.h"
#include "Util.h"

namespace RadGrabber
{

	int deviceCount;
	cudaDeviceProp* deviceProps;

	void InitializePlugin()
	{
		cudaGetDeviceCount(&deviceCount);
		deviceProps = static_cast<cudaDeviceProp*>(malloc(sizeof(cudaDeviceProp) * deviceCount));
		for (int i = 0; i < deviceCount; i++)
			cudaGetDeviceProperties(deviceProps + i, i);
	}

	void ClearPlugin()
	{
		deviceCount = 0;
		SAFE_HOST_FREE(deviceProps);
	}

	int __declspec(dllexport) __stdcall GetDeviceCount()
	{
		return deviceCount;
	}

	int __declspec(dllexport) __stdcall GetMaxThreadPerBlock(int device)
	{
		ASSERT(0 <= device && device < deviceCount);
		return deviceProps[device].maxThreadsPerBlock;
	}

	int __declspec(dllexport) __stdcall GetMaxThreadPerBlockInAllDevice()
	{
		int max = 0;
		for (int i = 1, max = deviceProps[0].maxThreadsPerBlock; i < deviceCount; i++)
			if (max < deviceProps[i].maxThreadsPerBlock)
				max = deviceProps[i].maxThreadsPerBlock;
		return max;
	}

	__host__ void GetOptimalBlockAndThreadDim(IN int deviceIndex, OUT OptimalLaunchParam& outParam)
	{
		int deviceCount;
		gpuErrchk(cudaGetDeviceCount(&deviceCount));
		ASSERT(0 <= deviceIndex && deviceIndex < deviceCount);

		cudaDeviceProp prop;
		gpuErrchk(cudaGetDeviceProperties(&prop, deviceIndex));
		/*
			TODO:: Calculate optimal block and thread count
		*/

		int optimalBlockCount, optimalThreadCount;

		optimalThreadCount = 256;
		optimalBlockCount = 65536 / optimalThreadCount;

		Log("%d,%d\n", optimalBlockCount, optimalThreadCount);

		outParam.threadCountinBlock = dim3(optimalThreadCount, 1, 1);
		outParam.blockCountInGrid = dim3(optimalBlockCount, 1, 1);
	}

	__host__ void GetOptimalBufferItemCount(IN int deviceIndex, OUT int& size)
	{
		int deviceCount;
		gpuErrchk(cudaGetDeviceCount(&deviceCount));

		ASSERT(0 <= deviceIndex && deviceIndex < deviceCount);

		cudaDeviceProp prop;
		cudaGetDeviceProperties(&prop, deviceIndex);

		size = prop.maxThreadsPerMultiProcessor * prop.multiProcessorCount;
	}
}