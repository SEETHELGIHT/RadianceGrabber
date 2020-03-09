#include "Define.h"

#pragma once

namespace RadGrabber
{
	class IAggregate;
	class IColorTarget;
	struct CameraChunk;
	struct RequestOption;

	class IIntegrator abstract
	{
	public:
		__host__ virtual void Render(const IAggregate& scene, const CameraChunk* c, IColorTarget& target, RequestOption opt) PURE;
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

	class PathIntegrator : public IIntegrator
	{
	public:
		__host__ PathIntegrator(RequestOption opt);
		__host__ ~PathIntegrator();

		// IIntegrator을(를) 통해 상속됨
		__host__ virtual void Render(const IAggregate & scene, const CameraChunk* c, IColorTarget& target, RequestOption opt) override;

	private:
		__global__ void Initialize(long long currentTime, const CameraChunk* c);
		__global__ void IntersectTest(const IAggregate& scene);
		__global__ void ScatteringAndAccumAttenuation(IColorTarget& target);

	private:
		int mMaxDepth;
		int mSegmentCount;
		PathSegment2* mSegments;
		Ray* mRays;
		SurfaceIntersection* mIsects;
		curandState* mRandStates;

	};

	class MLTIntergrator : public IIntegrator
	{
	public:

		// IIntegrator을(를) 통해 상속됨
		__host__ virtual void Render(const IAggregate & scene, const CameraChunk* c, IColorTarget& target, RequestOption opt) override;
	};
}
