#include "DataTypes.cuh"
#include "Integrator.h"
#include "Marshal.cuh"

#pragma once

namespace RadGrabber
{
	struct FrameMutableInput;
	struct FrameImmutableInput;
	struct CameraChunk;

	struct PathSemgentInternal
	{
		int onWorking : 1;
		union
		{
			struct
			{
				uint foundLightSourceFlag : 1;
				uint endOfDepthFlag : 1;
				uint missedRay : 1;
			};
			unsigned int endOfRay : 3;
		};
		int remainBounces : 12;
		int remainSamplingCount : 16;

		__forceinline__ __host__ __device__  PathSemgentInternal() {}
	};

	struct PathThreadSegment
	{
		int pixelIndex;
		//union
		//{
		//	struct
		//	{
		//		int onWorking : 1;
		//		union
		//		{
		//			struct
		//			{
		//				uint foundLightSourceFlag : 1;
		//				uint endOfDepthFlag : 1;
		//				uint missedRay : 1;
		//			};
		//			unsigned int endOfRay : 3;
		//		};
		//		int remainBounces : 12;
		//		int remainSamplingCount : 16;
		//	};
		//	struct PathSemgentInternal in;
		//};

		__forceinline__ __host__ __device__ PathThreadSegment() {}
	};

	enum class PathPixelSegmentResult
	{
		None				= 0x00000000,
		TermHitLight		= 0x00000001,
		TermRoulette		= 0x00000002,
		TermBlackOut		= 0x00000004,
		TermLimitBounce		= 0x00000008,
		TermMiss			= 0x00000010,
		TermError			= 0x00000020,
		TermGeometryPDFZero = 0x00000040,
		TermLightPDFZero	= 0x00000080,
		StartBounce		= 0x00000100,
		StartSampling	= 0x00000200,
	};

	struct PathPixelSegment
	{
		uint onWorking : 1;
		uint finishedWork : 1;
		uint remainBounces : 6;
		uint remainSampleCount : 9;
		uint reserved : 15;
		ColorRGB luminance;
		ColorRGB throughput;
		Ray ray;
		SurfaceIntersection isect;
		uint flag;
	};

	enum class PathIntegratorPixelContent : uint
	{
		Luminance				= 0x0000,
		Bounce					= 0x0001,
		ReverseBounce			= 0x0002,
		LastThroughput			= 0x0003,
		MinThroughput			= 0x0004,
		FlagTermHitLight		= 0x1001,
		FlagTermRoulette		= 0x1002,
		FlagTermBlackOut		= 0x1003,
		FlagTermLimitBounce		= 0x1004,
		FlagTermMiss			= 0x1005,
		FlagTermError			= 0x1006,
		FlagTermGeometryPDFZero	= 0x1007,
		FlagTermLightPDFZero	= 0x1008,
		FlagStartSampling		= 0x1009,
		FlagStartBounce			= 0x100a,
	};

	struct PathConstant
	{
		int incrementalMode;
		int maxSamplingCount;
		int maxDepthCount;
		int frameWidth;
		int frameHeight;
		int mutableIndex;
		int segCnt;
		int threadIterationCount;
		long long currentTime;
		PathIntegratorPixelContent pixelContent;

		union
		{
			CameraChunk camera;
			byte cameraData[sizeof(CameraChunk)];
		};
		union
		{
			SkyboxChunk skybox;
			byte skyboxData[sizeof(SkyboxChunk)];
		};
		union
		{
			FrameImmutableInput immutable;
			byte immutableData[sizeof(FrameImmutableInput)];
		};
		union
		{
			FrameMutableInput mutableIn;
			byte mutableData[sizeof(FrameMutableInput)];
		};

		ColorRGBA* cb;
		const IMultipleInput* in;
		const IAggregate* agg;
		const IIteratableAggregate* iagg;
		PathThreadSegment* segs;
		PathPixelSegment* pixels;
		curandState* randStates;

		int* devicePixelEndCount;
		int* nextPixelCount;

		__forceinline__ __host__ __device__ PathConstant() {}
		__host__ __device__ void SetConstantValues(bool isIncremental, int maxSampleCount, int maxDepthCount, Vector2i texSize, int segmentCount, long long currentTime, int threadIterationCount)
		{
			this->incrementalMode = isIncremental;
			this->maxSamplingCount = maxSampleCount;
			this->maxDepthCount = maxDepthCount;
			this->frameWidth = texSize.x;
			this->frameHeight = texSize.y;
			this->segCnt = segmentCount;
			this->currentTime = currentTime;
			this->threadIterationCount = threadIterationCount;
		}

		__host__ __device__ void SetFrameConstantValues(int mutableIndex, const FrameMutableInput& min)
		{
			this->mutableIndex = mutableIndex;
			this->camera = min.cameraBuffer[min.selectedCameraIndex];
			this->skybox = min.skyboxMaterialBuffer[this->camera.skyboxIndex];
		}
	};

	class PathIntegrator : public ICancelableIntergrator, public ICancelableIterativeIntergrator
	{
	public:
		__host__ PathIntegrator(const OptimalLaunchParam& param, Vector2i resultImageResolution, bool incrementalMode = true);
		__host__ PathIntegrator(const OptimalLaunchParam& param, Vector2i resultImageResolution, PathIntegratorPixelContent pixelContent, bool incrementalMode = true);
		__host__ ~PathIntegrator();

		// IIntegrator을(를) 통해 상속됨
		__host__ virtual void RenderStraight(const IAggregate* getDeviceScene(int mutableIndex), const HostDevicePair<IMultipleInput*>& in, IColorTarget& target, const RequestOption& opt, OptimalLaunchParam& param) override;
		__host__ virtual void RenderIncremental(const IAggregate* getDeviceScene(int mutableIndex), const HostDevicePair<IMultipleInput*>& in, IColorTarget& target, const RequestOption& opt, OptimalLaunchParam& param) override;

		__host__ virtual void RenderStraight(const IIteratableAggregate* getDeviceScene(int mutableIndex), const HostDevicePair<IMultipleInput*>& in, IColorTarget& target, const RequestOption& opt, OptimalLaunchParam& param) override;
		__host__ virtual void RenderIncremental(const IIteratableAggregate* getDeviceScene(int mutableIndex), const HostDevicePair<IMultipleInput*>& in, IColorTarget& target, const RequestOption& opt, OptimalLaunchParam& param) override;

	private:
		int mSegmentCount;
		PathThreadSegment* mDeviceSegments;
		PathThreadSegment* mHostSegments;
		curandState* mRandStates;
		PathPixelSegment* mPixelSegments;
		PathIntegratorPixelContent mPixelContent;
		bool mIncrementalMode;

	public:
		// ICancelable을(를) 통해 상속됨
		__host__ virtual void ReserveCancel() override;
		__host__ virtual bool IsCancel() override;

	private:
		bool reserveCancel;

	};

	PathIntegrator* CreatePathIntegrator(const OptimalLaunchParam& param, Vector2i resultImageResolution);

	__global__ void InitializeKernel();
	__global__ void IntersectKernel();
	__global__ void ColorKernel();
}