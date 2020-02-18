#include <cuda_runtime.h>
#include <cstdlib>

#include "Define.h"
#include "DeviceConfig.h"

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

__host__ void GetOptimalBlockAndThreadDim(IN int deviceIndex, IN int warpCount, OUT OptimalLaunchParam& outParam)
{
	int deviceCount;
	cudaGetDeviceCount(&deviceCount);

	ASSERT(0 <= deviceIndex && deviceIndex < deviceCount);

	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, deviceIndex);

	int optimalBlockCount, optimalThreadCount;
	/*
		TODO: Calculate optimal block and thread count
	*/
}
