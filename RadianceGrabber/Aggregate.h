#include "Marshal.cuh"

#pragma once

namespace RadGrabber
{
	class IAggregate abstract
	{
	public:
		__host__ __device__ virtual bool Intersect(const Ray& ray, SurfaceIntersection& isect, int threadCount) const PURE;
		__host__ __device__ virtual bool IntersectOnly(const Ray& ray, int threadCount) const PURE;

	public:
		__forceinline__ __host__ __device__ bool static IntersectTriangleAndGetGeometric(
			int i0, int i1, int i2, int sbidx,
			const Ray & rayInMS,
			const MeshChunk* mc, const MeshRendererChunk* mrc, const MaterialChunk* materials, const Texture2DChunk* textures,
			SurfaceIntersection & isect, float& distance, float& lastDistance, int& submeshIndex
		);
		__host__ __device__ static bool IntersectCheckAndGetGeometric(const Ray & rayInWS, const MeshRendererChunk* mrc, const MeshChunk* meshes, const MaterialChunk* materials, const Texture2DChunk* textures, SurfaceIntersection & isect, float& lastDistance, int& submeshIndex);
		__host__ __device__ static bool IntersectCheck(const Ray & ray, const MeshRendererChunk* mrc, const MeshChunk* meshes, const MaterialChunk* materials, const Texture2DChunk* textures);
	};

	class Sphere : public IAggregate
	{
	public:
		__host__ __device__ Sphere(const Vector3f& position, float radius);
		__host__ __device__ virtual bool Intersect(const Ray& ray, SurfaceIntersection& isect, int threadCount) const override;
		__host__ __device__ virtual bool IntersectOnly(const Ray& ray, int threadCount) const override;

		__host__ static Sphere* GetSphereDevice(const Vector3f& position, float radius);

	private:
		Vector3f position;
		float radius;

	};

	class LinearAggregate : public IAggregate
	{
	public:
		__host__ __device__ LinearAggregate();
		__host__ LinearAggregate(const FrameMutableInput* min, const FrameImmutableInput* imin);
		__device__ LinearAggregate(const FrameMutableInput* min, const FrameImmutableInput* imin, int meshCount, MeshChunk* meshChunkDevice);
		//__host__ ~LinearAggregate();

		// IAggregate을(를) 통해 상속됨
		__host__ __device__ virtual bool Intersect(const Ray & ray, SurfaceIntersection & isect, int threadCount) const override;
		__host__ __device__ virtual bool IntersectOnly(const Ray& ray, int threadCount) const override;

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

	enum class SpaceSplitAxis : unsigned int
	{
		None	= 0x00,
		X		= 0x01,
		Y		= 0x02,
		Z		= 0x03
	};

	enum class TransformItemKind : unsigned int
	{
		MeshRenderer = 0x01,
		SkinnedMeshRenderer = 0x02,
		Light = 0x03
	};

	struct TransformNode
	{
		Bounds bound;

		union
		{
			struct
			{
				unsigned int isNotInternal : 1;
				SpaceSplitAxis splitAxis : 2;
				int rightChildOffset : 29;                          
			};
			struct
			{
				unsigned int isLeaf : 1;
				TransformItemKind kind : 2;
				int itemIndex : 29;
			};
		};
	};
	struct TransformBVHTraversalSegment
	{
		int nodeIndex;
	};
	struct TransformBVHTraversalStack
	{
		int stackCount;
		int stackCapacity;
		TransformBVHTraversalSegment* traversalStack;
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

	class AcceleratedAggregate : public IAggregate
	{
	public:
		__host__ AcceleratedAggregate();
		__host__ __device__ AcceleratedAggregate(
			const FrameMutableInput* min, const FrameImmutableInput* imin,
			TransformBVHBuildData* transformNodeGroup, TransformBVHTraversalStack* transformNodeTraversalStack,
			StaticMeshBuildData* staticMeshNodeGroups, StaticMeshKDTreeTraversalStack* staticMeshTraversalStack,
			DynamicMeshBuildData* dynamicMeshNodeGroups, DynamicMeshBVHTraversalStack* dynamicMeshTraversalStack
		);
		__host__ ~AcceleratedAggregate();

		__host__ __device__ virtual bool Intersect(const Ray & ray, SurfaceIntersection & isect, int threadCount) const override;
		__host__ __device__ virtual bool IntersectOnly(const Ray & ray, int threadCount) const override;

		__host__ static bool BuildTransformBVH(
			IN int meshRendererLen,			IN MeshRendererChunk* meshRenderers, 
			IN int skinnedMeshRendererLen,	IN SkinnedMeshRendererChunk* skinnedMeshRenderers, 
			IN int lightLen,				IN LightChunk* lights,
			OUT TransformBVHBuildData* data
			);
		__host__ __device__ static bool TraversalTransformBVH(
			const Ray& rayInWS,
			int transformNodeCount, const TransformNode* transformNodes, TransformBVHTraversalStack* stack
		);

		__host__ static bool BuildStaticMeshKDTree(IN MeshChunk* c, OUT StaticMeshBuildData* data);
		__host__ __device__ static bool TraversalStaticMeshKDTree(
			const Ray& rayInMS,
			const MeshChunk* mc, const MeshRendererChunk* mrc,
			const MaterialChunk* materials, const Texture2DChunk* textures,
			const StaticMeshNode* meshNodes, StaticMeshKDTreeTraversalStack* stack,
			INOUT float& distance, OUT SurfaceIntersection& isect
		);
		__host__ __device__ static bool TraversalStaticMeshKDTree(
			const Ray& rayInMS,
			const MeshChunk* mc, const MeshRendererChunk* mrc,
			const MaterialChunk* materials, const Texture2DChunk* textures,
			const StaticMeshNode* meshNodes, StaticMeshKDTreeTraversalStack* stack
		);

		__host__ static bool BuildDynamicMeshBVH(IN MeshChunk* c, OUT DynamicMeshBuildData* data);
		__host__ static bool UpdateDynamicMeshBVH(IN MeshChunk* c, OUT DynamicMeshNode** nodeArray);
		__host__ __device__ static bool TraversalDynamicMeshBVH(
			const Ray& rayInMS,
			const MeshChunk* mc, const MeshRendererChunk* mrc,
			const MaterialChunk* materials, const Texture2DChunk* textures,
			const DynamicMeshNode* meshNodes, DynamicMeshBVHTraversalStack* stack,
			INOUT float& distance, OUT SurfaceIntersection& isect
		);
		__host__ __device__ static bool TraversalDynamicMeshBVH(
			const Ray& rayInMS,
			const MeshChunk* mc, const MeshRendererChunk* mrc,
			const MaterialChunk* materials, const Texture2DChunk* textures,
			const DynamicMeshNode* meshNodes, DynamicMeshBVHTraversalStack* stack
		);

		__host__ static AcceleratedAggregate* GetAggregateHost(const FrameMutableInput* min, const FrameImmutableInput* imin, int threadCount);
		__host__ static AcceleratedAggregate* GetAggregateDevice(IMultipleInput* hin, IMultipleInput* din, int mutableIndex, int threadCount);
		__host__ static void DestroyDeviceAggregate(AcceleratedAggregate* agg, int threadCount);

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

		TransformBVHBuildData* mTransformNodeGroup;
		int mTransformNodeStackCapacity;
		TransformBVHTraversalStack* mTransformNodeTraversalStack;

		StaticMeshBuildData* mStaticMeshNodeGroups;
		int mStaticMeshStackCapacity;
		StaticMeshKDTreeTraversalStack* mStaticMeshTraversalStack;

		DynamicMeshBuildData* mDynamicMeshNodeGroups;
		int mDynamicMeshStackCapacity;
		DynamicMeshBVHTraversalStack* mDynamicMeshTraversalStack;
	};
}
