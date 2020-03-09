#include <chrono>
#include <ratio>

#include "ColorTarget.h"
#include "Integrator.h"
#include "Aggregate.h"
#include "Marshal.h"
#include "DataTypes.cuh"
#include "DeviceConfig.h"
#include "MemUtil.h"

using namespace RadGrabber;

#define PI 3.14159265358979323846

/*
	TODO:: Path Tracing 구현
	TODO:: MLT 구현
*/
/*
	Path Tracing
	1. 메터리얼 별로 BxDF 설정 이후 ray 방향 정하기
		opaque, transparent 냐에 따라서 달라짐.
		alpha clipping 으로 통과 혹은 처리

		if alpha clipping? && texture sampling == 0: break;

		if emission
			끝, 색 계산 후 마침.

		if opaque
			BRDF 사용하여 ray direction + rgb color
		else
			그냥 통과 + rgb filter(use BTDF 지만)

	2. ray 부딫친 위치에서 빛 샘플링 하기

		광원 + emmision Object 처리
		(현재 광원은 따로 처리 되어 있음)

	4. russian roulette 으로 중간에 멈추기 처리?
		구현 이후에
	5. Subsurface Scattering, Transmission 구현은 나중에 ㅎㅎ
*/

/*
	Perspective Camera : Generate Ray
*/
__forceinline__ __device__ Ray GenerateRay(CameraChunk* c, int pixelIndex, const Vector2i& textureResolution)
{
	float theta = c->verticalFOV * PI / 180;
	Vector2f size;
	size.y = tan(theta / 2);
	size.x = c->aspect * size.y;
	Vector3f direciton =
		c->position - size.x * c->right - size.y * c->up - c->forward +
		((float)(pixelIndex % textureResolution.x) / textureResolution.x) * c->right +
		(float)((int)pixelIndex / textureResolution.x) * c->up - c->position;

	return Ray(c->position, direciton);
}

__forceinline__ __host__ __device__ float gamma(int n) {
	return (n * MachineEpsilon) / (1 - n * MachineEpsilon);
}

__forceinline__ __device__ Vector2i GetImageIndex(int startIndex, int pixelIndex, const Vector2i& texRes)
{
	return Vector2i(pixelIndex % texRes.x, (int)pixelIndex / texRes.x);
}

struct PathContant
{
	int maxDepth;
	Vector2i textureResolution;
};
 
__constant__ PathContant g_PathContants;

__host__ PathIntegrator::PathIntegrator(RequestOption opt)
{
	OptimalLaunchParam param;
	GetOptimalBlockAndThreadDim(0, 4, param);

	mSegmentCount = param.itemCount;

	mRandStates = (curandState*)MAllocDevice(sizeof(curandState) * param.itemCount);
	MSet(mRandStates, 0, sizeof(curandState) * param.itemCount);

	mRays = (Ray*)MAllocDevice(sizeof(Ray) * param.itemCount);
	MSet(mRays, 0, sizeof(Ray) * param.itemCount);

	mSegments = (PathSegment2*)MAllocDevice(sizeof(PathSegment2) * param.itemCount);
	MSet(mSegments, 0, sizeof(PathSegment2) * param.itemCount);
}

__host__ PathIntegrator::~PathIntegrator()
{
	SAFE_DEVICE_DELETE(mRandStates);
	SAFE_DEVICE_DELETE(mRays);
	SAFE_DEVICE_DELETE(mSegments);
}

__host__ void PathIntegrator::Render(const IAggregate& scene, const CameraChunk* c, IColorTarget& target, RequestOption opt)
{
	int samplingCount = 0, prevDrawSamplingCount = samplingCount;
	const int
		drawSamplingCount = 4,
		updateMilliSecond = 1000;

	PathContant initParam;
	initParam.maxDepth = opt.maxSamplingCount;
	initParam.textureResolution = opt.resultImageResolution;

	MCopyToSymbol(&g_PathContants, &initParam, sizeof(PathContant), 0, cudaMemcpyHostToDevice);

	OptimalLaunchParam param;
	GetOptimalBlockAndThreadDim(0, 4, param);

	std::chrono::system_clock::time_point lastUpdateTime = std::chrono::system_clock::now();
	KLAUNCH_ARGS4(Initialize, param.blockCountInGrid, param.threadCountinBlock, 0, 0)(std::chrono::duration_cast<std::chrono::milliseconds>(lastUpdateTime.time_since_epoch()).count(), c);

	int frameSegmentCount = target.GetFrameWidth() * target.GetFrameHeight();

	while (samplingCount < opt.maxSamplingCount)
	{
		for (int pixelIndex = 0; pixelIndex < frameSegmentCount; pixelIndex += param.itemCount)
		{
			KLAUNCH_ARGS4(IntersectTest, param.blockCountInGrid, param.threadCountinBlock, 0, 0)(scene);
			KLAUNCH_ARGS4(ScatteringAndAccumAttenuation, param.blockCountInGrid, param.threadCountinBlock, 0, 0)(target);
		}

		samplingCount++;

		if (samplingCount - prevDrawSamplingCount > drawSamplingCount ||
			std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now() - lastUpdateTime).count() >= updateMilliSecond)
		{
			if (opt.updateFunc) opt.updateFunc();

			prevDrawSamplingCount = drawSamplingCount;
			lastUpdateTime = std::chrono::system_clock::now();
		}
	}
}

__global__ void PathIntegrator::Initialize(long long currentTime, const CameraChunk* c)
{
	int threadIndex = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y;
	curand_init(currentTime, threadIndex, 0, mRandStates + threadIndex);

	// Generate Ray
	Vector2i imageIndex = GetImageIndex(0, threadIndex, g_PathContants.textureResolution);
	float theta = c->verticalFOV * PI / 180;
	Vector2f size;
	size.y = tan(theta / 2);
	size.x = c->aspect * size.y;
	Vector3f direciton =
		c->position - size.x * c->right - size.y * c->up - c->forward +
		((float)imageIndex.x * c->right + (float)imageIndex.y * c->up - c->position);
	mRays[threadIndex] = Ray(c[threadIndex].position, direciton);

	for (int offset = 0; offset < mSegmentCount; offset += blockDim.x * blockDim.y * blockDim.z)
	{
		mSegments[threadIndex + offset].foundLightSourceFlag = 0;
		mSegments[threadIndex + offset].endOfDepthFlag = 0;
		mSegments[threadIndex + offset].missedRay = 0;
		mSegments[threadIndex + offset].attenuation = ColorRGBA::One();
		mSegments[threadIndex + offset].pixelIndex = threadIndex;
		mSegments[threadIndex + offset].remainingBounces = g_PathContants.maxDepth;
	}
}


__global__ void PathIntegrator::IntersectTest(const IAggregate& scene)
{
	int threadIndex = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y;

	if (threadIndex < mSegmentCount) return;
	if (mSegments[threadIndex].foundLightSourceFlag || 
		mSegments[threadIndex].endOfDepthFlag) return;

	mSegments[threadIndex].missedRay = !scene.Intersect(mRays[threadIndex], mIsects[threadIndex]);
	mSegments[threadIndex].remainingBounces--;
}

__global__ void PathIntegrator::ScatteringAndAccumAttenuation(IColorTarget& target)
{
	int threadIndex = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y;

	if (mSegments[threadIndex].missedRay)
	{
		/*
			TODO:: sampling uniform random dist, BSDF
		*/

		goto FindNextPixel;
	}
	else if (mSegments[threadIndex].remainingBounces == 0)
	{
		/*
			TODO:: invalidate? 
		*/

		goto FindNextPixel;
	}
	else
	{
		if (mIsects[threadIndex].isGeometry)
		{
			/*
				TODO:: calculate color as attenuation by skybox
			*/
		}
		else
		{
			/*
				TODO:: calculate color as attenuation by light source
				
			*/

			goto FindNextPixel;
		}
	}

FindNextPixel:
	/*
		TODO:: find next pixel
	*/
}

__host__ void MLTIntergrator::Render(const IAggregate & scene, const CameraChunk* c, IColorTarget& target, RequestOption opt)
{
}
