#include <cstdlib>
#include <chrono>

#include <cuda.h>
#include <curand_kernel.h>

#include "Marshal.h"
#include "Pipeline.h"
#include "DeviceConfig.h"
#include "Aggregate.h"

namespace RadGrabber
{
	struct PathSegment
	{
		Ray ray;
		ColorRGBA attenuation;
		int pixelIndex;
		int remainingBounces;
	};

	
#pragma pack(push, 4)
	struct PathInit
	{
		int maxDepth;
	};

	struct PathSegment2
	{ 
		union
		{
			struct
			{
				int foundLightSourceFlag : 1;
				int endOfDepthFlag : 1;
				int missedRay : 1;
			};
			int endOfRay : 3;
		};
		int remainingBounces : 13;
		int pixelIndex;
		ColorRGBA attenuation;
	};

#pragma pack(pop)

	__forceinline__ __device__ Vector2i GetImageIndex(int startIndex, int pixelIndex, const Vector2i& texRes)
	{
		return Vector2i(pixelIndex % texRes.x, (int)pixelIndex / texRes.x);
	}

	__constant__ PathInit pathInitParam;

	/*
		Kernel : initialize
			initalize: random state
			initalize: Perspective Camera . Generate Ray
			initalize: segments
	*/
	__global__ void InitParams(curandState* states, Ray* rays, PathSegment2* segments, long long currentTime, CameraChunk* c, const Vector2i& textureResolution)
	{
		int threadIndex = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y;
		curand_init(currentTime, threadIndex, 0, states + threadIndex);

		// Generate Ray
		Vector2i imageIndex = GetImageIndex(0, threadIndex, textureResolution);
		float theta = c->verticalFOV * PI / 180;
		Vector2f size;
		size.y = tan(theta / 2);
		size.x = c->aspect * size.y;
		Vector3f direciton =
			c->position - size.x * c->right - size.y * c->up - c->forward +
			((float)imageIndex.x * c->right + (float)imageIndex.y * c->up - c->position);
		rays[threadIndex] = Ray(c[threadIndex].position, direciton);

		for (int offset = 0; offset < textureResolution.x * textureResolution.y; offset += blockDim.x * blockDim.y * blockDim.z)
		{
			segments[threadIndex + offset].foundLightSourceFlag = 0;
			segments[threadIndex + offset].endOfDepthFlag = 0;
			segments[threadIndex + offset].missedRay = 0;
			segments[threadIndex + offset].attenuation = ColorRGBA::One();
			segments[threadIndex + offset].pixelIndex = threadIndex;
			segments[threadIndex + offset].remainingBounces = pathInitParam.maxDepth;
		}
	}

	__global__ void IntersectionTest(Ray* rays, SurfaceIntersection* isects, FrameInput* deviceInput, PathSegment2* segments, int segmentCount,)
	{
		int threadIndex = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y;
		
		if (threadIndex < segmentCount) return;
		if (segments[threadIndex].foundLightSourceFlag || segments[threadIndex].endOfDepthFlag) return;

		segments[threadIndex].missedRay = !IntersectGeometryLinear(rays[threadIndex], deviceInput, isects[threadIndex]);
	}

	__global__ void ComputeScatteringAndColor(Ray* rays, SurfaceIntersection* isects, FrameInput* deviceInput, PathSegment2* segments, int segmentCount)
	{
		int threadIndex = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y;

		// End
		if (segments[threadIndex].missedRay)
		{
		}
		else if (segments[threadIndex].foundLightSourceFlag)
		{
		}
		else if (segments[threadIndex].endOfDepthFlag)
		{
		}
		// Scattering 
		else 
		{
		}
	}

	__host__ int IncrementalPTSampling(IN FrameRequest* hostReq, IN FrameInput* deviceInput)
	{
		int samplingCount = 0, prevDrawSamplingCount = samplingCount;
		const int
			drawSamplingCount = 4,
			limitSamplingCount = hostReq->opt.maxSamplingCount,
			updateMilliSecond = 1000;

		PathInit initParam;
		initParam.maxDepth = 50;

		ASSERT_IS_FALSE(cudaMemcpyToSymbol(&pathInitParam, &initParam, 1, 0, cudaMemcpyHostToDevice));

		OptimalLaunchParam param;
		GetOptimalBlockAndThreadDim(0, 4, param);

		curandState* randStateBuffer = nullptr;

		ASSERT_IS_FALSE(cudaMalloc(&randStateBuffer, sizeof(curandState) * param.itemCount));
		ASSERT_IS_FALSE(cudaMemset(randStateBuffer, 0, sizeof(curandState) * param.itemCount));

		Ray* rays = nullptr;

		ASSERT_IS_FALSE(cudaMalloc(&rays, sizeof(Ray) * param.itemCount));
		ASSERT_IS_FALSE(cudaMemset(rays, 0, sizeof(Ray) * param.itemCount));

		std::chrono::system_clock::time_point lastUpdateTime = std::chrono::system_clock::now();

		PathSegment2* allSegments = nullptr;

		ASSERT_IS_FALSE(cudaMalloc(&allSegments, sizeof(PathSegment2) * (hostReq->opt.resultImageResolution.x * hostReq->opt.resultImageResolution.y)));
		ASSERT_IS_FALSE(cudaMemset(allSegments, 0, sizeof(PathSegment2) * (hostReq->opt.resultImageResolution.x * hostReq->opt.resultImageResolution.y)));

		IAggregate* deviceAggregate = LinearAggregate::GetAggregate(hostReq->input.GetMutable(0), FrameInput::GetFrameMutableFromFrame(deviceInput));

		InitParams<<< param.blockCountInGrid, param.threadCountinBlock, 0, 0 >>>(
			randStateBuffer,
			rays,
			allSegments,
			std::chrono::duration_cast<std::chrono::milliseconds>(lastUpdateTime.time_since_epoch()).count(),
			hostReq->input.cameraBuffer[hostReq->opt.selectedCameraIndex],
			hostReq->opt.resultImageResolution
			);

		int segmentCount = hostReq->output.pixelBufferSize.x * hostReq->output.pixelBufferSize.y;

		while (samplingCount < limitSamplingCount)
		{
			for (int pixelIndex = 0; pixelIndex < segmentCount; pixelIndex += param.itemCount)
			{
				IntersectionTest <<< param.blockCountInGrid, param.threadCountinBlock, 0, 0 >>> (
					rays, allSegments + pixelIndex, segmentCount, deviceInput
					);
				ComputeScatteringAndColor <<< param.blockCountInGrid, param.threadCountinBlock, 0, 0 >>> (
					rays, allSegments + pixelIndex, segmentCount, deviceInput
					);
			}

			samplingCount++;

			if (samplingCount - prevDrawSamplingCount > drawSamplingCount ||
				std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now() - lastUpdateTime).count() >= updateMilliSecond)
			{
				hostReq->opt.updateFunc();

				prevDrawSamplingCount = drawSamplingCount;
				lastUpdateTime = std::chrono::system_clock::now();
			}
		}

		return 0;
	}
}
