#include "Filters.h"
#include "DeviceConfig.h"

namespace RadGrabber
{
	__global__ void ValidatedBlurKernel(ColorRGBA* colors, int width, int height)
	{
	}

	void ValidatedBlur(ColorRGBA* colors, int width, int height)
	{
		int deviceID;
		cudaGetDevice(&deviceID);
		OptimalLaunchParam param;
		GetOptimalBlockAndThreadDim(deviceID, param);

		int colorCount = width * height, gridCount;
	}
}