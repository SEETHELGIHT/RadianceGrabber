#include <random>

#include "Marshal.cuh"
#include "AcceleratedAggregate.h"
#include "ConfigUtility.h"
#include "Pipeline.h"

namespace RadGrabber
{
	namespace Test
	{
		__global__ void RayIntersectionTest(Ray* rays, IAggregate* dagg, int *results)
		{
			int threadIndex = getGlobalIdx_3D_3D();
			results[threadIndex] = dagg->IntersectOnly(rays[threadIndex], threadIndex);
		}

#define RAY_COUNT 256

		int HitTestKernelTest()
		{
			int result;
			int mode, iterateCount = 1;
			printf("Interactive :: 0, Iterate And Count :: 1, 0 or 1 ? ");
			scanf("%d", &mode);
			if (mode)
			{
				printf("Iterate Count ? ");
				scanf("%d", &iterateCount);
			}

			Utility::TestProjectConfig config;
			config.RefreshValues();

			FILE* fp = nullptr;
			fopen_s(&fp, config.frameRequestPath, "rb");
			FrameRequest* req = new FrameRequest();
			LoadFrameRequest(fp, &req, malloc);
			fclose(fp);

			auto min = req->input.GetMutable(0);
			auto imin = req->input.GetImmutable();

			FrameInput* in;
			AllocateDeviceFrameRequest(req, &in);
			AcceleratedAggregate	*agg = AcceleratedAggregate::GetAggregateHost(req->input.GetMutable(0), req->input.GetImmutable(), 1),
									*dagg = AcceleratedAggregate::GetAggregateDevice(&req->input, in, 0, RAY_COUNT);

			std::default_random_engine generator;
			std::uniform_real_distribution<float> distribution(0, 1);
			
			Ray hostRays[RAY_COUNT];
			Ray *deviceRays = (Ray*)MAllocDevice(sizeof(Ray) * RAY_COUNT);
			int hostRayCheck[RAY_COUNT];
			int *deviceRayCheck = (int*)MAllocDevice(sizeof(int) * RAY_COUNT);

			for (int i = 0; i < iterateCount; mode == 1 ? i++ : i = 0)
			{
				for (int j = 0; j < RAY_COUNT; j++)
				{
					GetRay(
						req->input.in.cameraBuffer[req->input.in.selectedCameraIndex].projectionInverseMatrix,
						req->input.in.cameraBuffer[req->input.in.selectedCameraIndex].cameraInverseMatrix,
						Vector2f(distribution(generator), distribution(generator)),
						hostRays[j]
					);
				}

				gpuErrchk(cudaMemcpy(deviceRays, hostRays, sizeof(Ray) * RAY_COUNT, cudaMemcpyKind::cudaMemcpyHostToDevice));
				gpuErrchk(cudaMemsetAsync(deviceRayCheck, 0, sizeof(int) * RAY_COUNT));

				dim3 gridDim = dim3(1, 1, 1),
					 blockDim = dim3(RAY_COUNT, 1, 1);
				RayIntersectionTest<<<gridDim, blockDim, 0, 0>>>(deviceRays, dagg, deviceRayCheck);
				gpuErrchk(cudaDeviceSynchronize());

				gpuErrchk(cudaMemcpy(hostRayCheck, deviceRayCheck, sizeof(int) * RAY_COUNT, cudaMemcpyKind::cudaMemcpyDeviceToHost));

				int diffCount = 0, equlCount = 0;

				for (int j = 0; j < RAY_COUNT; j++)
				{
					if (hostRayCheck[j] == (int)agg->IntersectOnly(hostRays[j], 0))
						equlCount++;
					else
						diffCount++;
				}
				
				gpuErrchk(cudaMemset(deviceRayCheck, 0, sizeof(int) * RAY_COUNT));

				printf("equal :: %d, different :: %d, %.6lf\n", equlCount, diffCount, (double)diffCount / (equlCount + diffCount));

				getchar();
			}

			gpuErrchk(cudaFree(deviceRays));
			gpuErrchk(cudaFree(deviceRayCheck));
			gpuErrchk(cudaDeviceReset());

			return 0;
		}
	}
}