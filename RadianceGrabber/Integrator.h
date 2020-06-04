#include "DataTypes.cuh"

#pragma once

namespace RadGrabber
{
	class IAggregate;
	class IMultipleInput;
	struct RequestOption;
	class IColorTarget;
	struct OptimalLaunchParam;

	class IIntegrator abstract
	{
	public:
		__host__ virtual void Render(const IAggregate& scene, const IMultipleInput& hin, const IMultipleInput& din, const RequestOption& opt, IColorTarget& target, OptimalLaunchParam& param) PURE;
	};

	class ICanclable abstract
	{
	public:
		__host__ virtual void ReserveCancel() PURE;
		__host__ virtual bool IsCancel() PURE;
	};

	class ICancelableIntergrator abstract : public IIntegrator, public ICanclable
	{
	};

	struct PathSegment2
	{
		union
		{
			struct
			{
				unsigned int foundLightSourceFlag : 1;
				unsigned int endOfDepthFlag : 1;
				unsigned int missedRay : 1;
			};
			unsigned int endOfRay : 3;
		};
		int remainingBounces : 13;
		int pixelIndex;
		Vector4f attenuation;
		Ray ray;
		union
		{
			struct
			{
				Vector3f position;
				Vector3f normal;
				Vector3f tangent;
				union
				{
					struct
					{
						int isGeometry : 1;
						int materialIndex : 31;
						Vector2f uv;
					};
					struct
					{
						int isNotLight : 1;
						int lightIndex : 31;
						Vector3f color;
					};
				};
			};
			SurfaceIntersection isect;
		};
	};

	struct FrameMutableInput;
	struct FrameImmutableInput;
	struct CameraChunk;

	class PathIntegrator : public ICancelableIntergrator
	{
	public:
		__host__ PathIntegrator(const RequestOption& opt);
		__host__ ~PathIntegrator();

		// IIntegrator을(를) 통해 상속됨
		__host__ virtual void Render(const IAggregate& scene, const IMultipleInput& hin, const IMultipleInput& din, const RequestOption& opt, IColorTarget& target, OptimalLaunchParam& param) override;

		// ICanclable을(를) 통해 상속됨
		__host__ virtual void ReserveCancel() override;
		__host__ virtual bool IsCancel() override;

	private:
		int mMaxDepth;
		int mSegmentCount;
		PathSegment2* mSegments;
		PathSegment2* mHostSegments;
		curandState* mRandStates;
		bool reserveCancel;

	};

	PathIntegrator* CreatePathIntegrator(const RequestOption& opt);

	class MLTIntergrator : public IIntegrator
	{
	public:

		// IIntegrator을(를) 통해 상속됨
		__host__ virtual void Render(const IAggregate& scene, const IMultipleInput& hin, const IMultipleInput& din, const RequestOption& opt, IColorTarget& target, OptimalLaunchParam& param) override;
	};

	MLTIntergrator* CreateMLTIntegrator(const RequestOption& opt);

	__host__ __device__ void SampleSkyboxAndInfLight(const IAggregate& dag, const FrameMutableInput* min, const FrameImmutableInput* imin, const Ray & ray, const SurfaceIntersection& isect, int threadIndex, OUT Vector3f& sat, OUT Vector3f& lat);
	__host__ __device__ Vector3f SampleFiniteLight(const FrameMutableInput* min , const Ray & ray, const SurfaceIntersection& isect, int threadIndex);

	struct PathContant;
	struct CameraChunk;
	struct FrameInput;

	__global__ void Initialize(int threadOffset, int pixelItemCount, PathSegment2* segments, /*Ray* rays, SurfaceIntersection* isects, */curandState* states, long long currentTime, const IMultipleInput& din, int mutableIndex);
	__global__ void IntersectTest(int threadOffset, int segmentCount, PathSegment2* segments, /*Ray* rays, SurfaceIntersection* isects, */const IAggregate& scene);
	__global__ void ScatteringAndAccumAttenuation(const IAggregate& dag, int threadOffset, int segmentCount, PathSegment2* segments, /*Ray* rays, SurfaceIntersection* isects, */curandState* states, int samplingCount, const IMultipleInput& din, int mutableIndex, IColorTarget& target);

	__host__ void SetPathConstant(PathContant c);
	__host__ void SetPathConstant(PathContant* pc);
}
