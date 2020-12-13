#include "DataTypes.cuh"
#include "Integrator.h"
#include "Aggregate.h"

#pragma once

namespace RadGrabber
{
	class IIteratableAggregate;
	class IColorTarget;

	struct IterativePathSemgentInternal
	{
		union
		{
			struct
			{
				uint isInitialized : 1;
				// branching flag
				uint isIntersection : 1;
				uint isInfLightIntersection : 1;
				// logical data
				uint remainBounces : 6;		// maximum 64
				uint remainIterateCount : 15;	// maximum remain count
				// 8
				uint reserved : 8;
			};
			float flag;
		};
		AATraversalSegment traversalSegement;

		__forceinline__ __host__ __device__ IterativePathSemgentInternal() {}

		__forceinline__ __host__ __device__ void InitAs(const IterativePathSemgentInternal& i)
		{
			this->flag = i.flag;
			this->traversalSegement.InitNoRay(i.traversalSegement);
		}
	};
	struct IterativeThreadSegment
	{
		int pidx;
		IterativePathSemgentInternal in;
		Ray rayInMS;
		union
		{
			int lightIndex;
		};
	};

	struct IterativePixelSegment
	{
		Ray rayInWS;
		ColorRGB atten;
		SurfaceIntersection isect;
		ColorRGB infLightColor;
	};

	struct IterativePathConstant
	{
		int currentSamplingCount;
		union
		{
			struct
			{
				float cameraProjectionInverseArray[16];
				float cameraSpaceInverseArray[16];
			};
			struct
			{
				Matrix4x4 cameraProjectionInverseMatrix;
				Matrix4x4 cameraSpaceInverseMatrix;
			};
		};

		// only constant values
		int frameWidth;
		int frameHeight;
		long long currentTime;
		IterativePathSemgentInternal pathSegmnetInitValue;
		int mutableIndex;
		int segCnt;

		// pointer constant values
		const IMultipleInput* in;
		const IIteratableAggregate* agg;

		int* pixelProcessEndCount;
		int* samplingCount;

		// value per thread
		int* threadSegmentInitFlag; // initFlag.length = initFlag.length / 32
		IterativeThreadSegment* threadSegments;
		curandState* threadRandStates;

		// value per pixel
		ColorRGBA* cb;
		IterativePixelSegment* pixelSegments;

		// calculated color pixel segment index
		int colorPixelIndexArrayCapcity;
		int* colorPixelIndexStartAndCount; // [0]: start index, [1]: item count
		int* colorPixelIndexArray;

		__forceinline__ __host__ __device__ IterativePathConstant() {}
	};
	extern __constant__ struct IterativePathConstant g_IterativePathConstant;

	struct IterativeThreadSegment;
	struct IterativePixelSegment;

	__global__ void HitTestKernel();
	__global__ void VisibleLightTestKernel();
	__global__ void ColorCalculationKernel();
	__global__ void ResetThreadSegments();

	class IterativePathIntegrator : public ICancelableIterativeIntergrator
	{
	public:
		__host__ IterativePathIntegrator(const RequestOption& opt, const OptimalLaunchParam& param);
		__host__ ~IterativePathIntegrator();

	public:
		// IIntegrator을(를) 통해 상속됨
		__host__ virtual void RenderStraight(const IIteratableAggregate* getDeviceScene(int mutableIndex), const HostDevicePair<IMultipleInput*>& in, IColorTarget& target, const RequestOption& opt, OptimalLaunchParam& param) override;
		__host__ virtual void RenderIncremental(const IIteratableAggregate* getDeviceScene(int mutableIndex), const HostDevicePair<IMultipleInput*>& in, IColorTarget& target, const RequestOption& opt, OptimalLaunchParam& param) override;

	private:
		int mSegmentCount;
		int* mSegmentInitFlags;
		IterativeThreadSegment* mDeviceSegments;
		IterativeThreadSegment* mHostSegments;
		curandState* mRandStates;
		IterativePixelSegment* mDevicePixelSegments;

	public:
		// ICancelable을(를) 통해 상속됨
		__host__ virtual void ReserveCancel() override;
		__host__ virtual bool IsCancel() override;

	private:
		bool reserveCancel;

	};

	IterativePathIntegrator* CreateIterativePathIntegrator(const RequestOption& opt, const OptimalLaunchParam& param);
}