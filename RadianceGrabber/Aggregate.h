#include "Marshal.cuh"

#pragma once

namespace RadGrabber
{
	__host__ __device__ bool IntersectCheckAndGetGeometric(
		const Ray & rayInWS, const MeshRendererChunk* mrc, const MeshChunk* meshes, const MaterialChunk* materials, const Texture2DChunk* textures, SurfaceIntersection & isect, float& lastDistance, int& submeshIndex
	);
	__host__ __device__ bool IntersectCheck(
		const Ray & ray, const MeshRendererChunk* mrc, const MeshChunk* meshes, const MaterialChunk* materials, const Texture2DChunk* textures
	);
	__host__ __device__ bool IntersectTriangleAndGetGeometric(
		int i0, int i1, int i2, int sbidx,
		const Ray & rayInMS,
		const MeshChunk* mc, const MeshRendererChunk* mrc, const MaterialChunk* materials, const Texture2DChunk* textures,
		SurfaceIntersection & isect, float& distance, float& lastDistance, int& submeshIndex
	);
	__host__ __device__ bool IntersectCheckWithDistance(
		const Ray & rayInWS,
		const MeshRendererChunk* mrc, const MeshChunk* meshes, const MaterialChunk* materials, const Texture2DChunk* textures,
		float& lastDistance
	);

	enum class AggregateItem : uint
	{
		StaticMesh			= 0x01 << 0,
		DynamicMesh			= 0x01 << 1,
		Light				= 0x01 << 2,
		All					= StaticMesh | DynamicMesh | Light
	};

	class IAggregate abstract
	{
	public:
		__host__ __device__ virtual bool Intersect(const Ray& ray, SurfaceIntersection& isect, int threadCount, uint flag = 0) const PURE;
		__host__ __device__ virtual bool Intersect(const Ray& ray, float minDistance, float maxDistance, int threadCount, uint flag = 0) const PURE;
		__host__ __device__ virtual bool IntersectOnly(const Ray& ray, int threadCount, uint flag = 0) const PURE;
		__host__ __device__ virtual int Size() const PURE;
	};

	struct AATraversalSegment
	{
		union
		{
			struct
			{
				uint initilaized : 1;
				uint isLowerTransform : 1;
				uint upperStackCount : 14;
				uint findPrimitive : 1;
				uint reserved : 15;
				float lastDistance;
			};
			struct
			{
				uint initilaized : 1;
				uint isNotUpperTransform : 1;
				uint upperStackIndex : 14;
				uint findPrimitive : 1;
				uint lowerStackCount : 15;
				float lastDistance;
			};
			float fd[2];
		};

		__host__ __device__ AATraversalSegment() {}

		__host__ __device__ void InitNoRay(const AATraversalSegment& s)
		{
			this->fd[0] = s.fd[0];
			this->fd[1] = s.fd[1];
		}
	};

	__forceinline__ __host__ __device__ void InitTraversalSegment(AATraversalSegment& seg)
	{
		seg.fd[0] = 0;
		seg.fd[1] = FLT_MAX;
	}

	class IIteratableAggregate abstract
	{
	public:
		__host__ __device__ virtual bool IterativeIntersect(const Ray& ray, Ray& rayInMS, SurfaceIntersection& isect, int threadCount, int& iterateCount, AATraversalSegment& travSeg, uint flag = 0) const PURE;
		__host__ __device__ virtual bool IterativeIntersectOnly(const Ray& ray, Ray& rayInMS, int threadCount, int& iterateCount, AATraversalSegment& travSeg, uint flag = 0) const PURE;
	};
}
