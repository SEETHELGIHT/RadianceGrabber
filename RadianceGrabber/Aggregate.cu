#include <vector>
#include <list>
#include <algorithm>
#include <cfloat>

#include "Aggregate.h"
#include "Util.h"
#include "Sample.cuh"

#define ITERATIVE_COST_COUNT 1

namespace RadGrabber
{
	__forceinline__ __host__ __device__ float AreaFromTriangle(const Vector3f& p0, const Vector3f& p1, const Vector3f& p2) 
	{
		return 0.5f * Cross(p1 - p0, p2 - p0).magnitude();
	}

	__host__ __device__ bool IntersectTriangleAndGetGeometric(
			int i0, int i1, int i2, int sbidx,
			const Ray & rayInMS,
			const MeshChunk* mc, const MeshRendererChunk* mrc, const MaterialChunk* materials, const Texture2DChunk* textures,
			SurfaceIntersection & isect, float& distance, float& lastDistance, int& submeshIndex
		)
	{
		const Vector3f&
			p0 = mc->positions[i0],
			p1 = mc->positions[i1],
			p2 = mc->positions[i2];
		Vector3f bc;

		if (
			IntersectRayAndTriangle
			(
				rayInMS,
				mc->positions[i0],
				mc->positions[i1],
				mc->positions[i2],
				distance,
				bc
			)
			)
		{
			const Vector3f&
				n0 = mc->normals[i0],
				n1 = mc->normals[i1],
				n2 = mc->normals[i2];
			isect.normal = (bc.x * n0 + bc.y * n1 + bc.z * n2).normalized();

			if (Dot(isect.normal, -rayInMS.direction) <= 0) return false;

			isect.normal = mrc->transformMatrix.TransformVector(isect.normal).normalized();

			if (mc->uvs && materials[mrc->materialArrayPtr[sbidx]].URPLit.IsAlphaClipping())
			{
				const Vector2f& uv0 = mc->uvs[i0], uv1 = mc->uvs[i1], uv2 = mc->uvs[i2];
				Vector2f uv = bc.x * uv0 + bc.y * uv1 + bc.z * uv2;

				int baseMapIndex = materials[mrc->materialArrayPtr[sbidx]].URPLit.baseMapIndex;
				ColorRGBA c = textures[baseMapIndex].Sample8888(uv);

				if (c.a == 0)
					return false;

				isect.uv = uv;
			}

			lastDistance = distance;
			submeshIndex = sbidx;

			isect.isHit = 1;
			isect.isGeometry = 1;
			isect.itemIndex = mrc->materialArrayPtr[sbidx];

			// PBRTv3:: Interpolate $(u,v)$ parametric coordinates and hit point
			isect.position = bc.x * p0 + bc.y * p1 + bc.z * p2;
			isect.position = mrc->transformMatrix.TransformPoint(isect.position);

			const Vector3f&
				t0 = mc->tangents[i0],
				t1 = mc->tangents[i1],
				t2 = mc->tangents[i2];
			isect.tangent = (bc.x * t0 + bc.y * t1 + bc.z * t2).normalized();
			isect.tangent = mrc->transformMatrix.TransformVector(isect.tangent);

			return true;
		}

		return false;
	}

	__host__ __device__ bool IntersectCheckAndGetGeometric(
		const Ray & rayInWS, 
		const MeshRendererChunk* mrc, const MeshChunk* meshes, const MaterialChunk* materials, const Texture2DChunk* textures, 
		SurfaceIntersection & isect, float& lastDistance, int& submeshIndex
	)
	{
		ASSERT_IS_NOT_NULL(mrc);

		const MeshChunk* mc = meshes + mrc->meshRefIndex;
		bool intersect = false;
		float distance = FLT_MAX;

		Ray rayInMS = rayInWS;
		rayInMS.origin = mrc->transformInverseMatrix.TransformPoint(rayInMS.origin);
		rayInMS.direction = mrc->transformInverseMatrix.TransformVector(rayInMS.direction);

		for (int sbidx = 0; sbidx < mc->submeshCount; sbidx++)
		{
			if (mc->submeshArrayPtr[sbidx].bounds.Intersect(rayInMS))
			{
				ASSERT((mc->submeshArrayPtr[sbidx].topology == eUnityMeshTopology::Triangles));

				int primitiveCount = mc->submeshArrayPtr[sbidx].indexCount,
					start = mc->submeshArrayPtr[sbidx].indexStart;
				for (int i = 0; i < primitiveCount; i += 3)
				{
					if (
						IntersectTriangleAndGetGeometric(
							mc->indices[i + start + 0], mc->indices[i + start + 1], mc->indices[i + start + 2], sbidx, rayInMS, 
							mc, mrc, materials, textures, 
							isect, distance, lastDistance, submeshIndex
						)
						)
						intersect = true;
				}
			}
		}

		return intersect;
	}

	__host__ __device__ bool IntersectCheckWithDistance(
		const Ray & rayInWS,
		const MeshRendererChunk* mrc, const MeshChunk* meshes, const MaterialChunk* materials, const Texture2DChunk* textures, 
		float& lastDistance
	)
	{
		ASSERT_IS_NOT_NULL(mrc);

		const MeshChunk* mc = meshes + mrc->meshRefIndex;

		Ray rayInMS = rayInWS;
		rayInMS.origin = mrc->transformInverseMatrix.TransformPoint(rayInMS.origin);
		rayInMS.direction = mrc->transformInverseMatrix.TransformVector(rayInMS.direction);

		for (int sbidx = 0; sbidx < mc->submeshCount; sbidx++)
		{
			if (mc->submeshArrayPtr[sbidx].bounds.Intersect(rayInMS))
			{
				ASSERT((mc->submeshArrayPtr[sbidx].topology == eUnityMeshTopology::Triangles));

				int indexCount = mc->submeshArrayPtr[sbidx].indexCount,
					start = mc->submeshArrayPtr[sbidx].indexStart;
				for (int i = 0; i < indexCount; i += 3)
				{
					const Vector3f&
						p0 = mc->positions[mc->indices[i + start + 0]],
						p1 = mc->positions[mc->indices[i + start + 1]],
						p2 = mc->positions[mc->indices[i + start + 2]];
					Vector3f bc;

					if (
						IntersectRayAndTriangle
						(
							rayInMS,
							mc->positions[mc->indices[i + start + 0]],
							mc->positions[mc->indices[i + start + 1]],
							mc->positions[mc->indices[i + start + 2]],
							bc,
							lastDistance
						)
						)
					{
						const Vector3f&
							n0 = mc->normals[mc->indices[i + start + 0]],
							n1 = mc->normals[mc->indices[i + start + 1]],
							n2 = mc->normals[mc->indices[i + start + 2]];
						Vector3f n = (bc.x * n0 + bc.y * n1 + bc.z * n2).normalized();
						//n = mrc->transformMatrix.TransformVector(n);

						if (Dot(n, -rayInMS.direction) <= 0) continue;

						if (mc->uvs && materials[mrc->materialArrayPtr[sbidx]].URPLit.IsAlphaClipping())
						{
							const Vector2f&
								uv0 = mc->uvs[mc->indices[i + start + 0]],
								uv1 = mc->uvs[mc->indices[i + start + 1]],
								uv2 = mc->uvs[mc->indices[i + start + 2]];
							Vector2f uv = bc.x * uv0 + bc.y * uv1 + bc.z * uv2;

							int baseMapIndex = materials[mrc->materialArrayPtr[sbidx]].URPLit.baseMapIndex;
							ColorRGBA c = textures[baseMapIndex].Sample8888(uv);

							if (c.a == 0)
								continue;
						}

						return true;
					}
				}
			}
		}

		return false;
	}

	__host__ __device__ bool IntersectCheck(
		const Ray & rayInWS, 
		const MeshRendererChunk* mrc, const MeshChunk* meshes, const MaterialChunk* materials, const Texture2DChunk* textures
	)
	{
		ASSERT_IS_NOT_NULL(mrc);

		const MeshChunk* mc = meshes + mrc->meshRefIndex;

		Ray rayInMS = rayInWS;
		rayInMS.origin = mrc->transformInverseMatrix.TransformPoint(rayInMS.origin);
		rayInMS.direction = mrc->transformInverseMatrix.TransformVector(rayInMS.direction);

		for (int sbidx = 0; sbidx < mc->submeshCount; sbidx++)
		{
			if (mc->submeshArrayPtr[sbidx].bounds.Intersect(rayInMS))
			{
				ASSERT((mc->submeshArrayPtr[sbidx].topology == eUnityMeshTopology::Triangles));

				int indexCount = mc->submeshArrayPtr[sbidx].indexCount,
					start = mc->submeshArrayPtr[sbidx].indexStart;
				for (int i = 0; i < indexCount; i += 3)
				{
					const Vector3f&
						p0 = mc->positions[mc->indices[i + start + 0]],
						p1 = mc->positions[mc->indices[i + start + 1]],
						p2 = mc->positions[mc->indices[i + start + 2]];
					Vector3f bc;

					if (
						IntersectRayAndTriangle
						(
							rayInMS,
							mc->positions[mc->indices[i + start + 0]],
							mc->positions[mc->indices[i + start + 1]],
							mc->positions[mc->indices[i + start + 2]],
							bc
						)
						)
					{
						const Vector3f&
							n0 = mc->normals[mc->indices[i + start + 0]],
							n1 = mc->normals[mc->indices[i + start + 1]],
							n2 = mc->normals[mc->indices[i + start + 2]];
						Vector3f n = (bc.x * n0 + bc.y * n1 + bc.z * n2).normalized();
						//n = mrc->transformMatrix.TransformVector(n);

						if (Dot(n, -rayInMS.direction) <= 0) continue;

						if (mc->uvs && materials[mrc->materialArrayPtr[sbidx]].URPLit.IsAlphaClipping())
						{
							const Vector2f&
								uv0 = mc->uvs[mc->indices[i + start + 0]],
								uv1 = mc->uvs[mc->indices[i + start + 1]],
								uv2 = mc->uvs[mc->indices[i + start + 2]];
							Vector2f uv = bc.x * uv0 + bc.y * uv1 + bc.z * uv2;

							int baseMapIndex = materials[mrc->materialArrayPtr[sbidx]].URPLit.baseMapIndex;
							ColorRGBA c = textures[baseMapIndex].Sample8888(uv);

							if (c.a == 0)
								continue;
						}

						return true;
					}
				}
			}
		}

		return false;
	}


}
