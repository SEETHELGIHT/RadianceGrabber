#include "Marshal.h"
#include "Interfaces.h"

#pragma once

namespace RadGrabber
{
	class LinearAggregate : public IAggregate
	{
	public:
		__host__ LinearAggregate() {}
		__host__ LinearAggregate(const GeometryInput* param);
		__host__ ~LinearAggregate();

		// IAggregate을(를) 통해 상속됨
		__host__ __device__ virtual void InitAggregate(const GeometryInput* param) override;
		// IAggregate을(를) 통해 상속됨
		__device__ virtual bool Intersect(const Ray & ray, SurfaceIntersection & isect) override;

		__host__ static LinearAggregate* GetAggregate(const GeometryInput* hostParam, const GeometryInput* deviceParam);

	private:
		__device__ bool Intersect(const Ray & ray, const MeshRendererChunk* mrc, SurfaceIntersection & isect, float& lastDitance);

	private:
		int mMRCount;
		MeshRendererChunk* mMRs;
		int mSkinnedMRCount;
		SkinnedMeshRendererChunk* mSkinnedMRs;
		int mLightCount;
		LightChunk* mLights;
		int mMeshCount;
		int mMRMeshCount;
		MeshChunk* mMeshs;
		bool mMeshArrayAllocated;
	};
	enum class BVHSplitAxis
	{
		None	= 0x00,
		X		= 0x01,
		Y		= 0x02,
		Z		= 0x03
	};
	enum class TransformBVHItemKind
	{
		MeshRenderer = 0x01,
		SkinnedMeshRenderer = 0x02,
		Light = 0x03
	};
	struct TransformBVHNode
	{
		Bounds bound;

		union
		{
			struct
			{
				BVHSplitAxis splitAxis : 2;
				int nextNode1Index : 31;
				int nextNode2Index : 31;
			};
			struct
			{
				int reserved : 2;
				TransformBVHItemKind kind : 2;
				int meshIndex : 28;
			};
		};
	};
	struct MeshBVHNode
	{
		Bounds bound;

		union
		{
			struct
			{
				BVHSplitAxis splitAxis : 2;
				int nextNode1Index : 31;
				int nextNode2Index : 31;
			};
			struct
			{
				int reserved : 2;
				int triangleIndex : 30;
			};
		};
	};

	class TwoLayerBVH : public IAggregate
	{
	public:
		__host__ TwoLayerBVH() {}
		__host__ TwoLayerBVH(const GeometryInput* param);

		// IAggregate을(를) 통해 상속됨
		__host__ virtual void InitAggregate(const GeometryInput* param) override;
		// IAggregate을(를) 통해 상속됨
		__device__ virtual bool Intersect(const Ray & ray, SurfaceIntersection & isect) override;

	private:
		TransformBVHNode* mTransformRootNode;
		int mMeshRootCount;
		MeshBVHNode* mMeshRootNodes;
	};
}
