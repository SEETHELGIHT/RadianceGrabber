#include "Aggregate.h"
#include "AcceleratedAggregateInternal.h"
#include "Marshal.cuh"

#pragma once

namespace RadGrabber
{

	class AcceleratedAggregate : public IIteratableAggregate, public IAggregate
	{
	public:
		__host__ AcceleratedAggregate();
		__host__ __device__ AcceleratedAggregate(
			const FrameMutableInput* min, const FrameImmutableInput* imin, int countOfStack,
			TransformBVHBuildData* transformNodeGroup, TransformBVHTraversalStack* transformNodeTraversalStack,
			StaticMeshBuildData* staticMeshNodeGroups, StaticMeshKDTreeTraversalStack* staticMeshTraversalStack,
			DynamicMeshBuildData* dynamicMeshNodeGroups, DynamicMeshBVHTraversalStack* dynamicMeshTraversalStack
		);
		__host__ ~AcceleratedAggregate();

		__host__ __device__ virtual bool Intersect(const Ray & ray, SurfaceIntersection & isect, int threadCount, uint flag = 0) const override;
		__host__ __device__ virtual bool Intersect(const Ray& ray, float minDistance, float maxDistance, int threadCount, uint flag = 0) const override;
		__host__ __device__ virtual bool IntersectOnly(const Ray & ray, int threadCount, uint flag = 0) const override;
		__host__ __device__ virtual int Size() const override { return sizeof(AcceleratedAggregate); };
		__host__ __device__ virtual bool IterativeIntersect(const Ray & ray, Ray& rayInMS, SurfaceIntersection & isect, int threadCount, int& iterateCount, AATraversalSegment& travSeg, uint flag = 0) const override;
		__host__ __device__ virtual bool IterativeIntersectOnly(const Ray & ray, Ray& rayInMS, int threadCount, int& iterateCount, AATraversalSegment& travSeg, uint flag = 0) const override;

		__host__ static AcceleratedAggregate* GetAggregateHost(const FrameMutableInput* min, const FrameImmutableInput* imin, int threadCount);
		__host__ static AcceleratedAggregate* GetAggregateDevice(IMultipleInput* hin, IMultipleInput* din, int mutableIndex, int threadCount);
		__host__ static void DestroyDeviceAggregate(AcceleratedAggregate* agg);

		__host__ static void GetAggregateDevice(int aagCount, AcceleratedAggregate* aag, IMultipleInput* hin, IMultipleInput* din, int threadCount);
		__host__ static void DestroyDeviceAggregate(int aagCount, AcceleratedAggregate* dagg);
		

	public:
		int mMRCount;
		MeshRendererChunk* mMRs;
		int mSkinnedMRCount;
		SkinnedMeshRendererChunk* mSkinnedMRs;
		int mLightCount;
		LightChunk* mLights;
		int mTextureCount;
		Texture2DChunk* mTextures;
		int mSkinnedMeshCount;
		MeshChunk* mSkinnedMeshs;
		int mStaticMeshCount;
		MeshChunk* mStaticMeshs;
		int mMaterialCount;
		MaterialChunk* mMaterials;

		int countOfStack;

		TransformBVHBuildData* mTransformNodeGroup;
		TransformBVHTraversalStack* mTransformNodeTraversalStack;

		StaticMeshBuildData* mStaticMeshNodeGroups;
		StaticMeshKDTreeTraversalStack* mStaticMeshTraversalStack;

		DynamicMeshBuildData* mDynamicMeshNodeGroups;
		DynamicMeshBVHTraversalStack* mDynamicMeshTraversalStack;

	};

}