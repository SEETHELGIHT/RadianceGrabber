#include <cuda_runtime_api.h>
#define _CRT_SECURE_NO_WARNINGS
#include <cstdio>
#include <chrono>

#include "Util.h"
#include "ColorTarget.h"
#include "Aggregate.h"
#include "integrator.h"
#include "Marshal.cuh"
#include "Sample.cuh"
#include "DataTypes.cuh"
#include "DeviceConfig.h"
#include "Unity/RenderAPI.h"
#include "Pipeline.h"

namespace RadGrabber
{
	namespace Test
	{
		int MatrixTest()
		{
			Matrix4x4 mat;

			return 0;
		}
	}
}
