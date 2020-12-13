#include <vector>
#include <list>
#include <algorithm>
#include <cfloat>

#include "LinearAggregate.h"
#include "Util.h"
#include "Sample.cuh"

namespace RadGrabber
{
	__host__ LinearAggregate::LinearAggregate()
	{
	}

	__host__ LinearAggregate::LinearAggregate(const FrameMutableInput* min, const FrameImmutableInput* imin)
	{
		mMeshArrayAllocated = true;

		mMRs = min->meshRendererBuffer;
		mMRCount = min->meshRendererBufferLen;
		mSkinnedMRs = min->skinnedMeshRendererBuffer;
		mSkinnedMRCount = min->skinnedMeshRendererBufferLen;
		mLights = min->lightBuffer;
		mLightCount = min->lightBufferLen;
		mTextures = imin->textureBuffer;
		mTextureCount = imin->textureBufferLen;
		mMaterials = min->materialBuffer;
		mMaterialCount = min->materialBufferLen;

		mMRMeshCount = imin->meshBufferLen;
		mMeshCount = imin->meshBufferLen + imin->skinnedMeshBufferLen;

		mMeshs = (MeshChunk*)malloc(mMeshCount * sizeof(MeshChunk));
		gpuErrchk(cudaMemcpy(mMeshs, imin->meshBuffer, imin->meshBufferLen * sizeof(MeshChunk), cudaMemcpyKind::cudaMemcpyHostToHost));
		gpuErrchk(cudaMemcpy(mMeshs + imin->meshBufferLen, imin->skinnedMeshBuffer, imin->skinnedMeshBufferLen * sizeof(MeshChunk), cudaMemcpyKind::cudaMemcpyHostToHost));
	}

	__device__ LinearAggregate::LinearAggregate(const FrameMutableInput* min, const FrameImmutableInput* imin, int meshCount, MeshChunk* meshChunkDevice)
	{
		mMeshArrayAllocated = true;

		mMRs = min->meshRendererBuffer;
		mMRCount = min->meshRendererBufferLen;
		mSkinnedMRs = min->skinnedMeshRendererBuffer;
		mSkinnedMRCount = min->skinnedMeshRendererBufferLen;
		mLights = min->lightBuffer;
		mLightCount = min->lightBufferLen;
		mTextures = imin->textureBuffer;
		mTextureCount = imin->textureBufferLen;
		mMaterials = min->materialBuffer;
		mMaterialCount = min->materialBufferLen;

		mMRMeshCount = imin->meshBufferLen;

		mMeshCount = meshCount;
		mMeshs = meshChunkDevice;
	}

	__host__ __device__ bool LinearAggregate::Intersect(const Ray & ray, SurfaceIntersection & isect, int threadIndex, uint flag) const
	{
		float distance = FLT_MAX, lastDistance = FLT_MAX;
		SurfaceIntersection isectBuffer;
		int submeshIndex = -1;

		if (!(flag & (uint)AggregateItem::StaticMesh))
		for (int i = 0; i < mMRCount; i++)
		{
			if (mMRs[i].boundingBox.Intersect(ray))
			{
				if (IntersectCheckAndGetGeometric(ray, mMRs + i, mMeshs, mMaterials, mTextures, isectBuffer, lastDistance, submeshIndex))
				{
					if (lastDistance < distance)
					{
						isect = isectBuffer;
						distance = lastDistance;
					}
				}
			}
		}

		if (!(flag & (uint)AggregateItem::DynamicMesh))
		for (int i = 0; i < mSkinnedMRCount; i++)
		{
			if (mSkinnedMRs[i].boundingBox.Intersect(ray))
			{
				if (IntersectCheckAndGetGeometric(ray, (MeshRendererChunk*)(mSkinnedMRs + i), mMeshs, mMaterials, mTextures, isectBuffer, lastDistance, submeshIndex))
				{
					if (lastDistance < distance)
					{
						isect = isectBuffer;
						distance = lastDistance;
					}
				}
			}
		}

		if (!(flag & (uint)AggregateItem::Light))
		for (int i = 0; i < mLightCount; i++)
		{
			if (mLights[i].IntersectRay(ray, isectBuffer, lastDistance))
			{
				if (lastDistance < distance)
				{
					isect = isectBuffer;
					isect.isGeometry = 0;
					isect.itemIndex = i;
					distance = lastDistance;
				}
			}
		}

		return distance != FLT_MAX;
	}

	__host__ __device__ bool LinearAggregate::Intersect(const Ray& ray, float minDistance, float maxDistance, int threadCount, uint flag) const
	{
		float distance = FLT_MAX, lastDistance = FLT_MAX;
		SurfaceIntersection isectBuffer;
		int submeshIndex = -1;

		if (!(flag & (uint)AggregateItem::StaticMesh))
			for (int i = 0; i < mMRCount; i++)
			{
				if (mMRs[i].boundingBox.Intersect(ray))
				{
					if (IntersectCheckWithDistance(ray, mMRs + i, mMeshs, mMaterials, mTextures, lastDistance))
					{
						if (minDistance <= lastDistance && lastDistance <= maxDistance)
						if (lastDistance < distance)
							distance = lastDistance;
					}
				}
			}

		if (!(flag & (uint)AggregateItem::DynamicMesh))
			for (int i = 0; i < mSkinnedMRCount; i++)
			{
				if (mSkinnedMRs[i].boundingBox.Intersect(ray))
				{
					if (IntersectCheckWithDistance(ray, (MeshRendererChunk*)(mSkinnedMRs + i), mMeshs, mMaterials, mTextures, lastDistance))
					{
						if (minDistance <= lastDistance && lastDistance <= maxDistance)
						if (lastDistance < distance)
							distance = lastDistance;
					}
				}
			}

		if (!(flag & (uint)AggregateItem::Light))
			for (int i = 0; i < mLightCount; i++)
			{
				if (mLights[i].IntersectRay(ray, isectBuffer, lastDistance))
				{
					if (minDistance <= lastDistance && lastDistance <= maxDistance)
					if (lastDistance < distance)
						distance = lastDistance;
				}
			}

		return distance != FLT_MAX;
	}

	__host__ __device__ bool LinearAggregate::IntersectOnly(const Ray & ray, int threadIndex, uint flag) const
	{
		if (!(flag & (uint)AggregateItem::StaticMesh))
		for (int i = 0; i < mMRCount; i++)
			if (mMRs[i].boundingBox.Intersect(ray))
				if (IntersectCheck(ray, mMRs + i, mMeshs, mMaterials, mTextures))
					return true;

		if (!(flag & (uint)AggregateItem::DynamicMesh))
		for (int i = 0; i < mSkinnedMRCount; i++)
			if (mSkinnedMRs[i].boundingBox.Intersect(ray))
				if (IntersectCheck(ray, (MeshRendererChunk*)(mSkinnedMRs + i), mMeshs, mMaterials, mTextures))
					return true;

		if (!(flag & (uint)AggregateItem::Light))
		for (int i = 0; i < mLightCount; i++)
			if (mLights[i].IntersectRayOnly(ray))
				return true;

		return false;
	}

	__global__ void SetAggregate2(LinearAggregate* allocated, LinearAggregate val)
	{
		if (threadIdx.x == 0)
			allocated = new (allocated)LinearAggregate(val);
	}

	__host__ LinearAggregate* LinearAggregate::GetAggregateHost(const FrameMutableInput* min, const FrameImmutableInput* imin)
	{
		return new LinearAggregate(min, imin);
	}

	__global__ void SetAggregate(LinearAggregate* allocated, IMultipleInput* in, int mutableIndex, int meshCount, MeshChunk* meshChunks)
	{
		if (threadIdx.x == 0)
			allocated = new (allocated)LinearAggregate(in->GetMutable(mutableIndex), in->GetImmutable(), meshCount, meshChunks);
	}

	__global__ void SetMeshBuffer(MeshChunk* allocated, int meshCount, IMultipleInput* din)
	{
		if (threadIdx.x == 0)
		{
			const FrameImmutableInput* dimin = din->GetImmutable();
			int idx = 0;
			for (; idx < dimin->meshBufferLen; idx++)
				allocated[idx] = dimin->meshBuffer[idx];
			for (; idx < dimin->skinnedMeshBufferLen; idx++)
				allocated[idx] = dimin->skinnedMeshBuffer[idx - dimin->meshBufferLen];
		}
	}

	__host__ LinearAggregate* LinearAggregate::GetAggregateDevice(IMultipleInput* hin, IMultipleInput* din, int mutableIndex)
	{
		const FrameMutableInput* hmin = hin->GetMutable(mutableIndex);
		const FrameImmutableInput* himin = hin->GetImmutable();
		int meshCount = himin->meshBufferLen + himin->skinnedMeshBufferLen;
		MeshChunk* meshChunks = (MeshChunk*)MAllocDevice(meshCount * sizeof(MeshChunk));
		SetMeshBuffer << <1, 1 >> > (meshChunks, meshCount, din);

		LinearAggregate* deviceAggreatePtr = (LinearAggregate*)MAllocDevice(sizeof(LinearAggregate));
		SetAggregate << < 1, 1 >> > (deviceAggreatePtr, din, mutableIndex, meshCount, meshChunks);

		return deviceAggreatePtr;
	}

	__host__ void LinearAggregate::DestroyDeviceAggregate(LinearAggregate* agg)
	{
		LinearAggregate hagg;
		gpuErrchk(cudaMemcpy(&hagg, agg, sizeof(LinearAggregate), cudaMemcpyKind::cudaMemcpyDeviceToHost));

		cudaFree(hagg.mMeshs);
		cudaFree(agg);
	}
}