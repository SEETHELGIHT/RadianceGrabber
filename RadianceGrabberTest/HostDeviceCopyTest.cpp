#include <cuda_runtime_api.h>
#include <cstdio>
#include <chrono>

#include "integrator.h"
#include "Marshal.cuh"
#include "Util.h"
#include "Sample.cuh"
#include "DataTypes.cuh"
#include "DeviceConfig.h"
#include "Pipeline.h"

namespace RadGrabber
{
	namespace Test
	{
		int HostDeviceCopyTest()
		{
			OptimalLaunchParam p;
			GetOptimalBlockAndThreadDim(0, p);

			int threadCnt = p.threadCountinBlock.x, blockCnt = p.blockCountInGrid.x, 
				segmentCount = blockCnt * threadCnt;

			FrameRequest* req = new FrameRequest();
			FILE* fp = fopen("./frq.framerequest", "rb");
			LoadFrameRequest(fp, &req, malloc);
			fclose(fp);

			FrameInput* din;
			AllocateDeviceMem(req, &din);

			curandState* randStates = (curandState*)MAllocDevice(sizeof(curandState) * segmentCount);
			MSet(randStates, 0, sizeof(curandState) * segmentCount);
			PathSegment2* segments = (PathSegment2*)MAllocDevice(sizeof(PathSegment2) * segmentCount);
			MSet(segments, 0, sizeof(PathSegment2) * segmentCount);
			PathSegment2* hostSegments = (PathSegment2*)malloc(sizeof(PathSegment2) * segmentCount);
			memset(hostSegments, 0, sizeof(PathSegment2) * segmentCount);

			PathContant c;
			c.maxDepth = 50;
			c.textureResolution = Vector2i(50, 25);
			SetPathConstant(c);

			CheckItems* item;
			item = (CheckItems*)MAllocManaged(sizeof(CheckItems));

			std::chrono::system_clock::time_point lastUpdateTime = std::chrono::system_clock::now();
			Initialize << <blockCnt, threadCnt >> > (
				0, 
				segmentCount, 
				segments, 
				randStates, 
				std::chrono::duration_cast<std::chrono::milliseconds>(lastUpdateTime.time_since_epoch()).count(), 
				*din, 
				0
				);

			gpuErrchk(cudaDeviceSynchronize());

			CheckItems it = *item;

			MCopy(hostSegments, segments, segmentCount * sizeof(PathSegment2), cudaMemcpyKind::cudaMemcpyDeviceToHost);

			for (int i = 0; i < 10; i++)
			{
				printf("Ray::direction::(%.2f, %.2f, %.2f)\n", hostSegments[i].ray.direction.x, hostSegments[i].ray.direction.y, hostSegments[i].ray.direction.z);
				printf("PathSegment2::pixelIndex::%d\n", hostSegments[i].pixelIndex);
			}

			SAFE_DEVICE_DELETE(randStates);
			SAFE_DEVICE_DELETE(segments);
			SAFE_HOST_FREE(hostSegments);

			gpuErrchk(cudaDeviceReset());

			return 0;
		}
	}
}