#include "Aggregate.h"
#include "Marshal.cuh"
#include "Sample.cuh"

#pragma once

namespace RadGrabber
{
#define ITERATIVE_COST_COUNT 1

	enum class SpaceSplitAxis : unsigned int
	{
		None = 0x00,
		X = 0x01,
		Y = 0x02,
		Z = 0x03
	};

	enum class TransformItemKind : unsigned int
	{
		MeshRenderer = 0x01,
		SkinnedMeshRenderer = 0x02,
		Light = 0x03
	};

	struct TransformNode
	{
		union
		{
			struct
			{
				uint isNotInternal : 2;
				int rightChildOffset : 30;
			};
			struct
			{
				TransformItemKind kind : 2;
				int itemIndex : 30;
			};
		};
		Bounds bound;
	};

	struct TransformBVHBuildSegment
	{
		Bounds bound;
		uint processed : 1;
		int parentIndex : 31;
		union
		{
			struct
			{
				uint isNotInternal : 2;
				int leftChildOffset : 31;
				int rightChildOffset : 31;
			};
			struct
			{
				TransformItemKind kind : 2;
				int itemIndex : 30;
			};
		};

		inline TransformBVHBuildSegment() :
			processed(0), parentIndex(0), bound(), isNotInternal(0), leftChildOffset(0), rightChildOffset(0)
		{}
		inline TransformBVHBuildSegment(TransformItemKind kind, int itemIndex, Bounds bound) :
			processed(0), parentIndex(0), bound(bound), kind(kind), itemIndex(itemIndex)
		{}
		inline TransformBVHBuildSegment(int leftChildOffset, int rightChildOffset, Bounds bound) :
			processed(0), parentIndex(0), bound(bound), isNotInternal(0), leftChildOffset(leftChildOffset), rightChildOffset(rightChildOffset)
		{}
	};
	struct TransformBVHTraversalStack
	{
		int stackCapacity;
		int* traversalStack;

		int listCount;
		int listCapacity;
		int* nodeIndexList;
	};
	struct TransformBVHBuildData
	{
		int count;
		int capacity;
		TransformNode* transformNodes;
	};


	struct DynamicMeshNode
	{
		union
		{

		};
	};
	struct DynamicMeshBuildData
	{
		int count;
		int capacity;
		DynamicMeshNode* meshNodes;
	};
	struct DynamicMeshBVHTraversalSegment
	{

	};
	struct DynamicMeshBVHTraversalStack
	{
		int stackCapacity;
		DynamicMeshBVHTraversalSegment* traversalStack;

	};

	struct StaticMeshNode
	{
		union
		{
			struct
			{
				SpaceSplitAxis splitAxis : 2;
				unsigned int rightChildOffset : 30;
				float overMinHyperPlaneOffset;
				float underMaxHyperPlaneOffset;
			};
			struct
			{
				unsigned int reserved : 2;
				unsigned int primitiveIndex1 : 31;
				unsigned int primitiveIndex2 : 31;
				//unsigned int primitiveIndex3 : 32;
			};
		};

		inline StaticMeshNode(SpaceSplitAxis ax, int childOffset, float overMinHyperPlaneOffset, float underMaxHyperPlaneOffset) :
			splitAxis(ax), rightChildOffset(childOffset), overMinHyperPlaneOffset(overMinHyperPlaneOffset), underMaxHyperPlaneOffset(underMaxHyperPlaneOffset)
		{ }
		inline StaticMeshNode(int primitiveIndex1, int primitiveIndex2) :
			reserved(0), primitiveIndex1(primitiveIndex1), primitiveIndex2(primitiveIndex2)
		{ }
	};
	struct StaticMeshBuildData
	{
		int count;
		int capacity;
		StaticMeshNode* meshNodes;
	};
	struct StaticMeshKDTreeTraversalSegment
	{
		int itemIndex;
		Bounds bound;
	};
	struct StaticMeshKDTreeTraversalStack
	{
		int stackCapacity;
		StaticMeshKDTreeTraversalSegment* traversalStack;
	};

	struct TransformRef
	{
		Vector3f				position;
		Quaternion				quaternion;
		Vector3f				scale;
		Matrix4x4				transformMatrix;
		Matrix4x4				transformInverseMatrix;
	};

	__host__ bool BuildTransformBVH(
		IN int meshRendererLen, IN MeshRendererChunk* meshRenderers,
		IN int skinnedMeshRendererLen, IN SkinnedMeshRendererChunk* skinnedMeshRenderers,
		IN int lightLen, IN LightChunk* lights,
		OUT TransformBVHBuildData* data
	);

	__forceinline__ __host__ __device__ bool TraversalTransformBVH(
		const Ray& rayInWS, const TransformNode* transformNodes, TransformBVHTraversalStack* stack
	)
	{
		int stackCount = 1;
		stack->traversalStack[0] = 0;
		stack->listCount = 0;

		while (stackCount > 0)
		{
			int nodeIndex = stack->traversalStack[--stackCount];

			if (transformNodes[nodeIndex].bound.Intersect(rayInWS))
			{
				if (transformNodes[nodeIndex].isNotInternal)
				{
					ASSERT(stack->listCount + 1 < stack->listCapacity);
					stack->nodeIndexList[stack->listCount++] = nodeIndex;
				}
				else
				{
					stack->traversalStack[stackCount++] = nodeIndex + transformNodes[nodeIndex].rightChildOffset;
					stack->traversalStack[stackCount++] = nodeIndex + 1;
				}
			}
		}

		return stack->listCount > 0;
	}

	__forceinline__ __host__ __device__ bool TraversalTransformBVH(
		const Ray& rayInWS, const TransformNode* transformNodes, TransformBVHTraversalStack* stack,
		int& iterateCount, struct AATraversalSegment& seg
	)
	{
		if (!seg.initilaized)
		{
			seg.findPrimitive = 0;

			seg.initilaized = 1;
			seg.upperStackCount = 1;
			seg.lastDistance = FLT_MAX;

			stack->traversalStack[0] = 0;
			stack->listCount = 0;
		}

		while (seg.upperStackCount > 0 && iterateCount > 0)
		{
			iterateCount -= ITERATIVE_COST_COUNT;
			int nodeIndex = stack->traversalStack[--seg.upperStackCount];

			if (transformNodes[nodeIndex].bound.Intersect(rayInWS))
			{
				if (transformNodes[nodeIndex].isNotInternal)
				{
					ASSERT(stack->listCount + 1 < stack->listCapacity);
					stack->nodeIndexList[stack->listCount++] = nodeIndex;
				}
				else
				{
					stack->traversalStack[seg.upperStackCount++] = nodeIndex + transformNodes[nodeIndex].rightChildOffset;
					stack->traversalStack[seg.upperStackCount++] = nodeIndex + 1;
				}
			}
		}

		return seg.upperStackCount <= 0;
	}
	__host__ bool BuildStaticMeshKDTree(IN MeshChunk* c, OUT StaticMeshBuildData* data);

	__host__ __device__ bool TraversalStaticMeshKDTree(
		const Ray& rayInWS, const Ray& rayInMS,
		float minDistance, float maxDistance,
		const MeshChunk* mc, const MeshRendererChunk* mrc, const MaterialChunk* materials, const Texture2DChunk* textures,
		const StaticMeshNode* meshNodes, StaticMeshKDTreeTraversalStack* stack,
		INOUT float& distance, OUT SurfaceIntersection& isect
	);
	__host__ __device__ bool TraversalStaticMeshKDTree(
		const Ray& rayInWS, const Ray& rayInMS,
		float minDistance, float maxDistance,
		const MeshChunk* mc, const MeshRendererChunk* mrc, const MaterialChunk* materials, const Texture2DChunk* textures,
		const StaticMeshNode* meshNodes, StaticMeshKDTreeTraversalStack* stack,
		INOUT float& distance
	);
	__host__ __device__ bool TraversalStaticMeshKDTree(
		const Ray& rayInWS, const Ray& rayInMS,
		const MeshChunk* mc, const MeshRendererChunk* mrc, const MaterialChunk* materials, const Texture2DChunk* textures,
		const StaticMeshNode* meshNodes, StaticMeshKDTreeTraversalStack* stack
	);
	__host__ __device__ bool TraversalStaticMeshKDTree(
		const Ray& rayInWS, const Ray& rayInMS,
		float minDistance, float maxDistance,
		const MeshChunk* mc, const MeshRendererChunk* mrc, const MaterialChunk* materials, const Texture2DChunk* textures,
		int& iterateCount, struct AATraversalSegment& seg,
		const StaticMeshNode* meshNodes, StaticMeshKDTreeTraversalStack* stack,
		OUT SurfaceIntersection& isect
	);
	__host__ __device__ bool TraversalStaticMeshKDTree(
		const Ray& rayInWS, const Ray& rayInMS,
		float minDistance, float maxDistance,
		const MeshChunk* mc, const MeshRendererChunk* mrc, const MaterialChunk* materials, const Texture2DChunk* textures,
		int& iterateCount, struct AATraversalSegment& seg,
		const StaticMeshNode* meshNodes, StaticMeshKDTreeTraversalStack* stack
	);

	__host__ bool BuildDynamicMeshBVH(IN MeshChunk* c, OUT DynamicMeshBuildData* data);
	__host__ bool UpdateDynamicMeshBVH(IN MeshChunk* c, OUT DynamicMeshNode** nodeArray);

	__forceinline__ __host__ __device__ bool TraversalDynamicMeshBVH(
		const Ray& rayInMS,
		float minDistance, float maxDistance,
		const MeshChunk* mc, const MeshRendererChunk* mrc,
		const MaterialChunk* materials, const Texture2DChunk* textures,
		const DynamicMeshNode* meshNodes, DynamicMeshBVHTraversalStack* stack,
		INOUT float& distance, OUT SurfaceIntersection& isect
	)
	{
		return false;
	}
	__forceinline__ __host__ __device__ bool TraversalDynamicMeshBVH(
		const Ray& rayInMS,
		float minDistance, float maxDistance,
		const MeshChunk* mc, const MeshRendererChunk* mrc,
		const MaterialChunk* materials, const Texture2DChunk* textures,
		const DynamicMeshNode* meshNodes, DynamicMeshBVHTraversalStack* stack,
		INOUT float& distance
	)
	{
		return false;
	}
	__forceinline__ __host__ __device__ bool TraversalDynamicMeshBVH(
		const Ray& rayInMS,
		const MeshChunk* mc, const MeshRendererChunk* mrc,
		const MaterialChunk* materials, const Texture2DChunk* textures,
		const DynamicMeshNode* meshNodes, DynamicMeshBVHTraversalStack* stack
	)
	{
		return false;
	}

	__forceinline__ __host__ __device__ bool TraversalDynamicMeshBVH(
		const Ray& rayInMS,
		float minDistance, float maxDistance,
		const MeshChunk* mc, const MeshRendererChunk* mrc,
		const MaterialChunk* materials, const Texture2DChunk* textures,
		int& iterateCount, struct AATraversalSegment& seg,
		const DynamicMeshNode* meshNodes, DynamicMeshBVHTraversalStack* stack,
		OUT SurfaceIntersection& isect
	)
	{
		return false;
	}
	__forceinline__ __host__ __device__ bool TraversalDynamicMeshBVH(
		const Ray& rayInMS,
		float minDistance, float maxDistance,
		const MeshChunk* mc, const MeshRendererChunk* mrc,
		const MaterialChunk* materials, const Texture2DChunk* textures,
		int& iterateCount, struct AATraversalSegment& seg,
		const DynamicMeshNode* meshNodes, DynamicMeshBVHTraversalStack* stack
	)
	{
		return false;
	}
}

