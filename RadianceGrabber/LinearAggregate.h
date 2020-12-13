#include "Aggregate.h"

#pragma once

namespace RadGrabber
{
	class LinearAggregate : public IAggregate
	{
	public:
		__host__ __device__ LinearAggregate();
		__host__ LinearAggregate(const FrameMutableInput* min, const FrameImmutableInput* imin);
		__device__ LinearAggregate(const FrameMutableInput* min, const FrameImmutableInput* imin, int meshCount, MeshChunk* meshChunkDevice);
		//__host__ ~LinearAggregate();

		// IAggregate을(를) 통해 상속됨, float minDistance, float maxDistance
		__host__ __device__ virtual bool Intersect(const Ray & ray, SurfaceIntersection & isect, int threadCount, uint flag = 0) const override;
		__host__ __device__ virtual bool Intersect(const Ray& ray, float minDistance, float maxDistance, int threadCount, uint flag = 0) const override;
		__host__ __device__ virtual bool IntersectOnly(const Ray& ray, int threadCount, uint flag = 0) const override;
		__host__ __device__ virtual int Size() const override { return sizeof(LinearAggregate); };

		__host__ static LinearAggregate* GetAggregateHost(const FrameMutableInput* min, const FrameImmutableInput* imin);
		__host__ static LinearAggregate* GetAggregateDevice(IMultipleInput* hin, IMultipleInput* din, int mutableIndex);
		__host__ static void DestroyDeviceAggregate(LinearAggregate* agg);

	public:
		int mMRCount;
		MeshRendererChunk* mMRs;
		int mSkinnedMRCount;
		SkinnedMeshRendererChunk* mSkinnedMRs;
		int mLightCount;
		LightChunk* mLights;
		int mTextureCount;
		Texture2DChunk* mTextures;
		int mMeshCount;
		int mMRMeshCount;
		MeshChunk* mMeshs;
		bool mMeshArrayAllocated;
		int mMaterialCount;
		MaterialChunk* mMaterials;
	};
}
