//#include "DataTypes.cuh"
#include "Marshal.h"

#pragma once

namespace RadGrabber
{
	class IAggregate
	{
	public:
		__host__ __device__ virtual bool Intersect(IN const Ray& ray, OUT SurfaceIntersection& isect) PURE;
	};

	class LinearAggretion : public IAggregate
	{
	public:
		

	private:
		MeshRendererChunk* mMRs;
		SkinnedMeshRendererChunk* mSkinnedMRs;
	};
}
