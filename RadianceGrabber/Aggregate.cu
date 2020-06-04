#include <vector>
#include <algorithm>
#include <cfloat>

#include "Aggregate.h"
#include "Util.h"
#include "Sample.cuh"

namespace RadGrabber
{
	__forceinline__ __host__ __device__ bool IAggregate::IntersectTriangleAndGetGeometric(
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

			isect.normal = mrc->transformMatrix.TransformVector(isect.normal);

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

			isect.isGeometry = 1;
			isect.materialIndex = mrc->materialArrayPtr[sbidx];

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
	}

	__host__ __device__ bool IAggregate::IntersectCheckAndGetGeometric(
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

	__host__ __device__ bool IAggregate::IntersectCheck(
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

	__host__ __device__ Sphere::Sphere(const Vector3f& position, float radius) : position(position), radius(radius)
	{
	}
	__host__ __device__ bool Sphere::Intersect(const Ray& ray, SurfaceIntersection& isect, int threadIndex) const
	{
		Vector3f delta = ray.origin - position;
		float	a = Dot(ray.direction, ray.direction),
			b = Dot(delta, ray.direction),
			c = Dot(delta, delta) - radius * radius;
		float	discrimnant = b * b - a * c;

		if (discrimnant > 0)
		{
			float temp = (-b - sqrtf(discrimnant)) / a;
			if (0 < temp)
			{
				isect.isGeometry = 1;
				isect.materialIndex = 0;
				isect.position = ray.GetPosAt(temp);
				isect.normal = (isect.position - position) / radius;

				return true;
			}

			float temp2 = (-b + sqrtf(discrimnant)) / a;
			if (0 < temp2)
			{
				isect.isGeometry = 1;
				isect.materialIndex = 0;
				isect.position = ray.GetPosAt(temp2);
				isect.normal = (isect.position - position) / radius;

				return true;
			}
		}

		return false;
	}

	__host__ __device__ bool Sphere::IntersectOnly(const Ray& ray, int threadIndex) const
	{
		Vector3f delta = ray.origin - position;
		float	a = Dot(ray.direction, ray.direction),
			b = Dot(delta, ray.direction),
			c = Dot(delta, delta) - radius * radius;
		float	discrimnant = b * b - a * c;

		if (discrimnant > 0)
		{
			float temp = (-b - sqrtf(discrimnant)) / a;
			if (0 < temp)
			{
				return true;
			}

			float temp2 = (-b + sqrtf(discrimnant)) / a;
			if (0 < temp2)
			{
				return true;
			}
		}

		return false;
	}

	__global__ void GetSphere(Sphere* s, const Vector3f position, float radius)
	{
		s = new (s)Sphere(position, radius);
	}
	__host__ Sphere* Sphere::GetSphereDevice(const Vector3f& position, float radius)
	{
		Sphere* allocated = (Sphere*)MAllocDevice(sizeof(Sphere));
		GetSphere << <1, 1 >> > (allocated, position, radius);
		return allocated;
	}

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

	__host__ __device__ bool LinearAggregate::Intersect(const Ray & ray, SurfaceIntersection & isect, int threadIndex) const
	{
		float distance = FLT_MAX, lastDistance = 0.f;
		SurfaceIntersection isectBuffer;
		int submeshIndex = -1;

		for (int i = 0; i < mMRCount; i++)
		{
			if (mMRs[i].boundingBox.Intersect(ray))
			{
				if (IAggregate::IntersectCheckAndGetGeometric(ray, mMRs + i, mMeshs, mMaterials, mTextures, isectBuffer, lastDistance, submeshIndex))
				{
					if (lastDistance < distance)
					{
						isect = isectBuffer;
						isect.isGeometry = 1;
						distance = lastDistance;
					}
				}
			}
		}

		for (int i = 0; i < mSkinnedMRCount; i++)
		{
			if (mSkinnedMRs[i].boundingBox.Intersect(ray))
			{
				if (IAggregate::IntersectCheckAndGetGeometric(ray, (MeshRendererChunk*)(mSkinnedMRs + i), mMeshs, mMaterials, mTextures, isectBuffer, lastDistance, submeshIndex))
				{
					if (lastDistance < distance)
					{
						isect = isectBuffer;
						isect.isGeometry = 1;
						distance = lastDistance;
					}
				}
			}
		}

		for (int i = 0; i < mLightCount; i++)
		{
			if (mLights[i].IntersectRay(ray, isectBuffer, lastDistance))
			{
				if (lastDistance < distance)
				{
					isect = isectBuffer;
					isect.isNotLight = 0;
					isect.materialIndex = -1;
					distance = lastDistance;
				}
			}
		}

		return distance != FLT_MAX;
	}


	__host__ __device__ bool LinearAggregate::IntersectOnly(const Ray & ray, int threadIndex) const
	{
		for (int i = 0; i < mMRCount; i++)
			if (mMRs[i].boundingBox.Intersect(ray))
				if (IntersectCheck(ray, mMRs + i, mMeshs, mMaterials, mTextures))
					return true;

		for (int i = 0; i < mSkinnedMRCount; i++)
			if (mSkinnedMRs[i].boundingBox.Intersect(ray))
				if (IntersectCheck(ray, (MeshRendererChunk*)(mSkinnedMRs + i), mMeshs, mMaterials, mTextures))
					return true;

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
				allocated[idx] = dimin->skinnedMeshBuffer[idx-dimin->meshBufferLen];
		}
	}

	__host__ LinearAggregate* LinearAggregate::GetAggregateDevice(IMultipleInput* hin, IMultipleInput* din, int mutableIndex)
	{
		const FrameMutableInput* hmin = hin->GetMutable(mutableIndex);
		const FrameImmutableInput* himin = hin->GetImmutable();
		int meshCount = himin->meshBufferLen + himin->skinnedMeshBufferLen;
		MeshChunk* meshChunks = (MeshChunk*)MAllocDevice(meshCount * sizeof(MeshChunk));
		SetMeshBuffer <<<1, 1 >>> (meshChunks, meshCount, din);

		LinearAggregate* deviceAggreatePtr = (LinearAggregate*)MAllocDevice(sizeof(LinearAggregate));
		SetAggregate<<< 1, 1 >>>(deviceAggreatePtr, din, mutableIndex, meshCount, meshChunks);
		
		return deviceAggreatePtr;
	}

	__host__ void LinearAggregate::DestroyDeviceAggregate(LinearAggregate* agg)
	{
		LinearAggregate hagg;
		gpuErrchk(cudaMemcpy(&hagg, agg, sizeof(LinearAggregate), cudaMemcpyKind::cudaMemcpyDeviceToHost));

		Log("DestroyDeviceAggregate::1");
		cudaFree(hagg.mMeshs);
		Log("DestroyDeviceAggregate::2");
		cudaFree(agg);
		Log("DestroyDeviceAggregate::3");
	}

	__host__ AcceleratedAggregate::AcceleratedAggregate()
	{
	}

	__host__ __device__ AcceleratedAggregate::AcceleratedAggregate(
		const FrameMutableInput* min, const FrameImmutableInput* imin,
		TransformBVHBuildData* transformNodeGroup, TransformBVHTraversalStack* transformNodeTraversalStack,
		StaticMeshBuildData* staticMeshNodeGroups, StaticMeshKDTreeTraversalStack* staticMeshTraversalStack,
		DynamicMeshBuildData* dynamicMeshNodeGroups, DynamicMeshBVHTraversalStack* dynamicMeshTraversalStack
	) :
		mMRs(min->meshRendererBuffer), mMRCount(min->meshRendererBufferLen), 
		mSkinnedMRs(min->skinnedMeshRendererBuffer), mSkinnedMRCount(min->skinnedMeshRendererBufferLen),
		mLights(min->lightBuffer), mLightCount(min->lightBufferLen),
		mTextures(imin->textureBuffer), mTextureCount (imin->textureBufferLen),
		mMaterials(min->materialBuffer), mMaterialCount (min->materialBufferLen),
		mStaticMeshCount(imin->meshBufferLen), mStaticMeshs(imin->meshBuffer),
		mSkinnedMeshCount(imin->skinnedMeshBufferLen), mSkinnedMeshs(imin->skinnedMeshBuffer),
		mTransformNodeGroup(transformNodeGroup), mTransformNodeTraversalStack(transformNodeTraversalStack),
		mStaticMeshNodeGroups(staticMeshNodeGroups), mStaticMeshTraversalStack(staticMeshTraversalStack),
		mDynamicMeshNodeGroups(dynamicMeshNodeGroups), mDynamicMeshTraversalStack(dynamicMeshTraversalStack)
	{
	}

	__host__ AcceleratedAggregate::~AcceleratedAggregate()
	{
	}

	__host__ __device__ bool AcceleratedAggregate::Intersect(const Ray & rayInWS, SurfaceIntersection & isect, int threadIndex) const
	{
		SurfaceIntersection isectBuffer;
		float distance = FLT_MAX, lastDistance = FLT_MAX;

		if (!TraversalTransformBVH(rayInWS, mMRCount + mSkinnedMRCount + mLightCount, mTransformNodeGroup->transformNodes, mTransformNodeTraversalStack + threadIndex))
			return false;

		for (int stackIndex = 0; stackIndex < mTransformNodeTraversalStack[threadIndex].stackCount; stackIndex++)
		{
			int nodeIndex = mTransformNodeTraversalStack[threadIndex].traversalStack[stackIndex].nodeIndex;
			const TransformNode& transformNode = mTransformNodeGroup->transformNodes[nodeIndex];

			switch (transformNode.kind)
			{
			case TransformItemKind::MeshRenderer:
			{
				MeshRendererChunk* mr = mMRs + transformNode.itemIndex;
				Ray rayInMS =
					Ray(
						mr->transformInverseMatrix.TransformPoint(rayInWS.origin),
						mr->transformInverseMatrix.TransformVector(rayInWS.direction)
					);

				if (TraversalStaticMeshKDTree(
					rayInMS,
					mStaticMeshs + mr->meshRefIndex, mr,
					mMaterials, mTextures,
					mStaticMeshNodeGroups[mr->meshRefIndex].meshNodes, mStaticMeshTraversalStack + threadIndex,
					lastDistance, isectBuffer
				))
				{
					distance = lastDistance;
					isect = isectBuffer;
				}
				break;
			}
			case TransformItemKind::SkinnedMeshRenderer:
			{
				SkinnedMeshRendererChunk* smr = mSkinnedMRs + transformNode.itemIndex;
				Ray rayInMS = 
					Ray(
						smr->transformInverseMatrix.TransformPoint(rayInWS.origin),
						smr->transformInverseMatrix.TransformVector(rayInWS.direction)
					);

				if (
					TraversalDynamicMeshBVH(
						rayInMS,
						mSkinnedMeshs + smr->skinnedMeshRefIndex, (MeshRendererChunk*)smr,
						mMaterials, mTextures,
						mDynamicMeshNodeGroups[smr->skinnedMeshRefIndex].meshNodes, mDynamicMeshTraversalStack + threadIndex,
						lastDistance, isectBuffer
					)
					)
				{
					distance = lastDistance;
					isect = isectBuffer;
				}
				break;
			}
			case TransformItemKind::Light:
			{
				if (mLights[transformNode.itemIndex].IntersectRay(rayInWS, isectBuffer, lastDistance))
				{
					distance = lastDistance;
					isect = isectBuffer;
					isect.isNotLight = 0;
					isect.lightIndex = transformNode.itemIndex;
					mLights[transformNode.itemIndex].Sample(rayInWS, isect, isect.color);
				}
				break;
			}
			}
		}

		return distance != FLT_MAX;
	}

	__host__ __device__ bool AcceleratedAggregate::IntersectOnly(const Ray & rayInWS, int threadIndex) const
	{
		if (!TraversalTransformBVH(rayInWS, mMRCount + mSkinnedMRCount + mLightCount, mTransformNodeGroup->transformNodes, mTransformNodeTraversalStack + threadIndex))
			return false;

		for (int stackIndex = 0; stackIndex < mTransformNodeTraversalStack[threadIndex].stackCount; stackIndex++)
		{
			int nodeIndex = mTransformNodeTraversalStack[threadIndex].traversalStack[stackIndex].nodeIndex;
			const TransformNode& transformNode = mTransformNodeGroup->transformNodes[nodeIndex];

			switch (transformNode.kind)
			{
			case TransformItemKind::MeshRenderer:
			{
				MeshRendererChunk* mr = mMRs + transformNode.itemIndex;
				Ray rayInMS =
					Ray(
						mr->transformInverseMatrix.TransformPoint(rayInWS.origin),
						mr->transformInverseMatrix.TransformVector(rayInWS.direction)
					);

				if (
					TraversalStaticMeshKDTree(
						rayInMS,
						mStaticMeshs + mr->meshRefIndex, mr,
						mMaterials, mTextures,
						mStaticMeshNodeGroups[mr->meshRefIndex].meshNodes, mStaticMeshTraversalStack + threadIndex
					))
					return true;

				break;
			}
			case TransformItemKind::SkinnedMeshRenderer:
			{
				SkinnedMeshRendererChunk* smr = mSkinnedMRs + transformNode.itemIndex;
				Ray rayInMS =
					Ray(
						smr->transformInverseMatrix.TransformPoint(rayInWS.origin),
						smr->transformInverseMatrix.TransformVector(rayInWS.direction)
					);

				if (
					TraversalDynamicMeshBVH(
						rayInMS,
						mSkinnedMeshs + smr->skinnedMeshRefIndex, (MeshRendererChunk*)smr,
						mMaterials, mTextures,
						mDynamicMeshNodeGroups[smr->skinnedMeshRefIndex].meshNodes, mDynamicMeshTraversalStack + threadIndex
					)
					)
					return true;

				break;
			}
			case TransformItemKind::Light:
				if (mLights[transformNode.itemIndex].IntersectRayOnly(rayInWS))
					return true;

				break;
			}
		}
		return false;
	}

	struct TransformBVHBuildSegment
	{
		int depth;
		int itemIndex;
		Bounds bound;

		inline TransformBVHBuildSegment() :
			depth(-1), itemIndex(-1), bound()
		{}
		inline TransformBVHBuildSegment(int depth, int itemIndex, Bounds bound) :
			depth(depth), itemIndex(itemIndex), bound(bound)
		{}
	};

	__host__ bool AcceleratedAggregate::BuildTransformBVH(
		IN int meshRendererLen, IN MeshRendererChunk* meshRenderers,
		IN int skinnedMeshRendererLen, IN SkinnedMeshRendererChunk* skinnedMeshRenderers,
		IN int lightLen, IN LightChunk* lights,
		OUT TransformBVHBuildData* data
	)
	{
		/*
			TODO:: build transform BVH 
		*/
		data->count = data->capacity = meshRendererLen + skinnedMeshRendererLen + lightLen;
		data->transformNodes = (TransformNode*)malloc(sizeof(TransformNode) * data->count);
		
		for (int i = 0; i < data->count; i++)
		{
			data->transformNodes[i].isLeaf = 1;

			if (i < meshRendererLen)
			{
				data->transformNodes[i].kind = TransformItemKind::MeshRenderer;
				data->transformNodes[i].bound = meshRenderers[i].boundingBox;
				data->transformNodes[i].itemIndex = i;
			}
			else if (i < meshRendererLen + skinnedMeshRendererLen)
			{
				data->transformNodes[i].kind = TransformItemKind::SkinnedMeshRenderer;
				data->transformNodes[i].bound = skinnedMeshRenderers[i - meshRendererLen].boundingBox;
				data->transformNodes[i].itemIndex = i - meshRendererLen;
			}
			else
			{
				data->transformNodes[i].kind = TransformItemKind::Light;
				if (!lights[i - meshRendererLen - skinnedMeshRendererLen].GetBoundingBox(data->transformNodes[i].bound))
					data->transformNodes[i].bound = Bounds();
				data->transformNodes[i].itemIndex = i - meshRendererLen - skinnedMeshRendererLen;
			}
		}
			

		//TransformBVHBuildSegment seg(-1, 0, transformCount);
		//std::list<TransformBVHBuildSegment> list(transformCount / 2);
		//list.push_back(seg);

		//int* indices = (int*)alloca(sizeof(int) * seg.primCount);
		//memcpy(indices, c->indices, sizeof(int) * c->indexCount);

		//::

		return true;
	}

	__host__ __device__ bool AcceleratedAggregate::TraversalTransformBVH(
		const Ray& rayInWS,
		int transformNodeCount, const TransformNode* transformNodes, TransformBVHTraversalStack* stack
	)
	{
		stack->stackCount = 0;
		for (int i = 0; i < transformNodeCount; i++)
			if (transformNodes[i].bound.Intersect(rayInWS))
				stack->traversalStack[stack->stackCount++].nodeIndex = i;
		return stack->stackCount > 0;
	}

	struct StaticMeshKDTreeBuildSegment
	{
		int parentNodeIndex;
		int startPrimIndex;
		int primCount;
		int depth;
		MeshChunk* c;
		Vector3f center;

		inline StaticMeshKDTreeBuildSegment()
			: parentNodeIndex(0), startPrimIndex(0), primCount(0), depth(0), c(nullptr)
		{}
		inline StaticMeshKDTreeBuildSegment(int parentNodeIndex, int startPrimIndex, int primCount, int depth, MeshChunk* c)
			: parentNodeIndex(parentNodeIndex), startPrimIndex(startPrimIndex), primCount(primCount), depth(depth), c(c) 
		{}
		
		// TODO:: remove static segment
		static StaticMeshKDTreeBuildSegment seg;
		__host__ static int PrimSortCompareFunc(const void* a, const void* b)
		{
			const int *tia = (const int *)a, *tib = (const int *)b;

			Vector3f	pa = (seg.c->positions[seg.c->indices[*tia * 3 + 0]] + seg.c->positions[seg.c->indices[*tia * 3 + 1]] + seg.c->positions[seg.c->indices[*tia * 3 + 2]]) / 3,
						pb = (seg.c->positions[seg.c->indices[*tib * 3 + 0]] + seg.c->positions[seg.c->indices[*tib * 3 + 1]] + seg.c->positions[seg.c->indices[*tib * 3 + 2]]) / 3;

			if (pa[seg.depth] < pb[seg.depth])
				return -1;
			else if (pa[seg.depth] > pb[seg.depth])
				return 1;
			else
				return 0;
		}
	};

	StaticMeshKDTreeBuildSegment StaticMeshKDTreeBuildSegment::seg = StaticMeshKDTreeBuildSegment();

	__host__ bool AcceleratedAggregate::BuildStaticMeshKDTree(IN MeshChunk* c, OUT StaticMeshBuildData* data)
	{
		StaticMeshKDTreeBuildSegment seg = StaticMeshKDTreeBuildSegment(-1, 0, c->indexCount / 3, 0, c);
		std::vector<StaticMeshKDTreeBuildSegment> list;
		list.push_back(seg);

		int* sortedPrimtives = (int*)alloca(sizeof(int) * seg.primCount);
		for (int i = 0; i < seg.primCount; i++)
			sortedPrimtives[i] = i;

		data->capacity = (c->indexCount / 3) * 2;
		data->meshNodes = (StaticMeshNode*)malloc(sizeof(StaticMeshNode) * data->capacity);

		int startPrimIndex, primCount, nodeCount = 0;

		while (list.size())
		{
			seg = list[list.size()-1];
			list.pop_back();

			if (seg.parentNodeIndex >= 0)
				data->meshNodes[seg.parentNodeIndex].rightChildOffset = nodeCount - seg.parentNodeIndex;

			if (seg.primCount > 2)
			{
				StaticMeshKDTreeBuildSegment::seg = seg;
				qsort(sortedPrimtives + seg.startPrimIndex, seg.primCount, sizeof(int), StaticMeshKDTreeBuildSegment::PrimSortCompareFunc);

				int		rightStartPrimIndex = seg.startPrimIndex + (seg.primCount + 1) / 2, 
						axisIndex = seg.depth % 3; 
				float	minHyperPlane = 
					fmin(
						c->positions[c->indices[sortedPrimtives[rightStartPrimIndex] * 3 + 0]][axisIndex],
						fmin(
							c->positions[c->indices[sortedPrimtives[rightStartPrimIndex] * 3 + 1]][axisIndex],
							c->positions[c->indices[sortedPrimtives[rightStartPrimIndex] * 3 + 2]][axisIndex]
						)
					),
						maxHyperPlane =
					fmax(
						c->positions[c->indices[sortedPrimtives[rightStartPrimIndex - 1] * 3 + 0]][axisIndex],
						fmax(
							c->positions[c->indices[sortedPrimtives[rightStartPrimIndex - 1] * 3 + 1]][axisIndex],
							c->positions[c->indices[sortedPrimtives[rightStartPrimIndex - 1] * 3 + 2]][axisIndex]
						)
					)
					;

				data->meshNodes[nodeCount] = StaticMeshNode((SpaceSplitAxis)(axisIndex + 1), -1, minHyperPlane - EPSILON, maxHyperPlane + EPSILON);

				startPrimIndex = seg.startPrimIndex;
				primCount = seg.primCount;

				seg.depth++;

				// right(far) node insertion
				seg.parentNodeIndex = nodeCount;
				seg.primCount = primCount / 2;
				seg.startPrimIndex = rightStartPrimIndex;
				list.push_back(seg);

				// left(one step) node insertion
				seg.parentNodeIndex = -1;
				seg.primCount = primCount - seg.primCount;
				seg.startPrimIndex = startPrimIndex;
				list.push_back(seg);
			}
			else
			{
				data->meshNodes[nodeCount].reserved = 0;
				data->meshNodes[nodeCount].primitiveIndex1 = sortedPrimtives[seg.startPrimIndex] * 3;
				if (seg.primCount > 1)
					data->meshNodes[nodeCount].primitiveIndex2 = sortedPrimtives[seg.startPrimIndex + 1] * 3;
				else
					data->meshNodes[nodeCount].primitiveIndex2 = INT_MAX;

				if (data->meshNodes[nodeCount].primitiveIndex1 >= c->indexCount)
					Log("sdlkfjslkdjflsk");
				if (data->meshNodes[nodeCount].primitiveIndex2 >= c->indexCount && data->meshNodes[nodeCount].primitiveIndex2 < INT_MAX)
					Log("sdlkfjslkdjflsk");
			}

			nodeCount++;
		}

		data->count = nodeCount;

		return true;
	}

	__host__ __device__ bool AcceleratedAggregate::TraversalStaticMeshKDTree(
		const Ray& rayInMS, 
		const MeshChunk* mc, const MeshRendererChunk* mrc, const MaterialChunk* materials, const Texture2DChunk* textures, 
		const StaticMeshNode* meshNodes, StaticMeshKDTreeTraversalStack* stack, 
		INOUT float& distance, OUT SurfaceIntersection& isect
	)
	{
		ASSERT(stack->stackCapacity);
		ASSERT_IS_FALSE(IsNan(rayInMS.origin));
		ASSERT_IS_FALSE(IsNan(rayInMS.direction));

		stack->traversalStack->bound = mc->aabbInMS;
		stack->traversalStack->itemIndex = 0;

		int stackIndex = 1;
		float lastDistance = distance;

		while (stackIndex > 0)
		{
			--stackIndex;
			StaticMeshKDTreeTraversalSegment curSegment = stack->traversalStack[stackIndex];
			const StaticMeshNode& meshNode = meshNodes[curSegment.itemIndex];

			if (meshNode.reserved)
			{
				const Bounds& b = curSegment.bound;
				const int axis = (int)meshNode.splitAxis - 1;

				Vector3f dir_frac = 1.0f / rayInMS.direction, smp, bgp;
				ASSERT_IS_FALSE(IsNan(dir_frac));

				// 1. calculate bound for bigger center BB
				bgp = b.center + b.extents;
				smp = b.center - b.extents;
				smp[axis] = meshNode.overMinHyperPlaneOffset;

				float t1, t2, t3, t4, t5, t6, tmin, tmax;
				// 2. calc intersection by bigger center BB
				t1 = (smp.x - rayInMS.origin.x) * dir_frac.x;
				t2 = (bgp.x - rayInMS.origin.x) * dir_frac.x;
				t3 = (smp.y - rayInMS.origin.y) * dir_frac.y;
				t4 = (bgp.y - rayInMS.origin.y) * dir_frac.y;
				t5 = (smp.z - rayInMS.origin.z) * dir_frac.z;
				t6 = (bgp.z - rayInMS.origin.z) * dir_frac.z;

				tmin = fmax(fmax(fmin(t1, t2), fmin(t3, t4)), fmin(t5, t6));
				tmax = fmin(fmin(fmax(t1, t2), fmax(t3, t4)), fmax(t5, t6));

				if (tmax >= 0 && tmin <= tmax)
				{
					ASSERT(stackIndex + 1 < stackCapacity);
					StaticMeshKDTreeTraversalSegment& nextSegment = stack->traversalStack[stackIndex];
					nextSegment.itemIndex = curSegment.itemIndex + meshNode.rightChildOffset;
					nextSegment.bound.center = (smp + bgp) / 2;
					nextSegment.bound.extents = Abs(bgp - nextSegment.bound.center);

					stackIndex++;
				}

				// 3. calculate bound for smaller center BB, 
				bgp = b.center + b.extents;
				smp = b.center - b.extents;
				bgp[axis] = meshNode.underMaxHyperPlaneOffset;

				// 4. calc intersection by smaller center BB
				t1 = (smp.x - rayInMS.origin.x) * dir_frac.x;
				t2 = (bgp.x - rayInMS.origin.x) * dir_frac.x;
				t3 = (smp.y - rayInMS.origin.y) * dir_frac.y;
				t4 = (bgp.y - rayInMS.origin.y) * dir_frac.y;
				t5 = (smp.z - rayInMS.origin.z) * dir_frac.z;
				t6 = (bgp.z - rayInMS.origin.z) * dir_frac.z;

				tmin = fmax(fmax(fmin(t1, t2), fmin(t3, t4)), fmin(t5, t6));
				tmax = fmin(fmin(fmax(t1, t2), fmax(t3, t4)), fmax(t5, t6));

				if (tmax >= 0 && tmin <= tmax)
				{
					ASSERT(stackIndex + 1 < stackCapacity);
					StaticMeshKDTreeTraversalSegment& nextSegment = stack->traversalStack[stackIndex];
					nextSegment.itemIndex = curSegment.itemIndex + 1;
					nextSegment.bound.center = (smp + bgp) / 2;
					nextSegment.bound.extents = Abs(bgp - nextSegment.bound.center);

					stackIndex++;
				}
			}
			else
			{
				int primitiveIndex = meshNode.primitiveIndex1;

			ACCEL_AGG_CALC_RAY_TRIANGLE_INTERSECT:

				int i0 = mc->indices[primitiveIndex + 0],
					i1 = mc->indices[primitiveIndex + 1],
					i2 = mc->indices[primitiveIndex + 2];

				Vector3f bc;
				const Vector3f &p0 = mc->positions[i0], &p1 = mc->positions[i1], &p2 = mc->positions[i2];

				if (IntersectRayAndTriangle(rayInMS, p0, p1, p2, lastDistance, bc))
				{
					const Vector3f& n0 = mc->normals[i0], &n1 = mc->normals[i1], &n2 = mc->normals[i2];
					isect.normal = (bc.x * n0 + bc.y * n1 + bc.z * n2).normalized();

					if (Dot(isect.normal, -rayInMS.direction) <= 0) 
						goto ACCEL_AGG_CALC_RAY_TRIANGLE_INTERSECT_NEXT_PRIMITIVE;

					isect.normal = mrc->transformMatrix.TransformVector(isect.normal);

					int sbidx = mc->GetSubmeshIndexFromIndex(i0);

					if (mc->uvs && materials[mrc->materialArrayPtr[sbidx]].URPLit.IsAlphaClipping())
					{
						const Vector2f& uv0 = mc->uvs[i0], uv1 = mc->uvs[i1], uv2 = mc->uvs[i2];
						Vector2f uv = bc.x * uv0 + bc.y * uv1 + bc.z * uv2;

						int baseMapIndex = materials[mrc->materialArrayPtr[sbidx]].URPLit.baseMapIndex;
						ColorRGBA c = textures[baseMapIndex].Sample8888(uv);

						if (c.a == 0)
							goto ACCEL_AGG_CALC_RAY_TRIANGLE_INTERSECT_NEXT_PRIMITIVE;

						isect.uv = uv;
					}

					isect.isGeometry = 1;
					isect.materialIndex = mrc->materialArrayPtr[sbidx];

					isect.position = bc.x * p0 + bc.y * p1 + bc.z * p2;
					isect.position = mrc->transformMatrix.TransformPoint(isect.position);

					const Vector3f& t0 = mc->tangents[i0], t1 = mc->tangents[i1], t2 = mc->tangents[i2];
					isect.tangent = (bc.x * t0 + bc.y * t1 + bc.z * t2).normalized();
					isect.tangent = mrc->transformMatrix.TransformVector(isect.tangent);

					//intserectCheck = true;
					distance = lastDistance;
					return true;
				}

			ACCEL_AGG_CALC_RAY_TRIANGLE_INTERSECT_NEXT_PRIMITIVE:

				if (primitiveIndex == meshNode.primitiveIndex1 && meshNode.primitiveIndex2 < INT_MAX)
				{
					primitiveIndex = meshNode.primitiveIndex2;
					goto ACCEL_AGG_CALC_RAY_TRIANGLE_INTERSECT;
				}
			}

		}

		return false;
	}

	__host__ __device__ bool AcceleratedAggregate::TraversalStaticMeshKDTree(
		const Ray& rayInMS,
		const MeshChunk* mc, const MeshRendererChunk* mrc, const MaterialChunk* materials, const Texture2DChunk* textures,
		const StaticMeshNode* meshNodes, StaticMeshKDTreeTraversalStack* stack
	)
	{
		ASSERT(stack->stackCapacity);
		ASSERT_IS_FALSE(IsNan(rayInMS.origin));
		ASSERT_IS_FALSE(IsNan(rayInMS.direction));

		stack->traversalStack->bound = mc->aabbInMS;
		stack->traversalStack->itemIndex = 0;

		int stackIndex = 1;

		while (stackIndex > 0)
		{
			--stackIndex;
			StaticMeshKDTreeTraversalSegment curSegment = stack->traversalStack[stackIndex];
			const StaticMeshNode& meshNode = meshNodes[curSegment.itemIndex];

			if (meshNode.reserved)
			{
				const Bounds& b = curSegment.bound;
				const int axis = (int)meshNode.splitAxis - 1;

				Vector3f dir_frac = 1.0f / rayInMS.direction, smp, bgp;
				ASSERT_IS_FALSE(IsNan(dir_frac));

				// 1. calculate bound for bigger center BB
				bgp = b.center + b.extents;
				smp = b.center - b.extents;
				smp[axis] = meshNode.overMinHyperPlaneOffset;

				float t1, t2, t3, t4, t5, t6, tmin, tmax;
				// 2. calc intersection by bigger center BB
				t1 = (smp.x - rayInMS.origin.x) * dir_frac.x;
				t2 = (bgp.x - rayInMS.origin.x) * dir_frac.x;
				t3 = (smp.y - rayInMS.origin.y) * dir_frac.y;
				t4 = (bgp.y - rayInMS.origin.y) * dir_frac.y;
				t5 = (smp.z - rayInMS.origin.z) * dir_frac.z;
				t6 = (bgp.z - rayInMS.origin.z) * dir_frac.z;

				tmin = fmax(fmax(fmin(t1, t2), fmin(t3, t4)), fmin(t5, t6));
				tmax = fmin(fmin(fmax(t1, t2), fmax(t3, t4)), fmax(t5, t6));

				if (tmax >= 0 && tmin <= tmax)
				{
					ASSERT(stackIndex + 1 < stackCapacity);
					StaticMeshKDTreeTraversalSegment& nextSegment = stack->traversalStack[stackIndex];
					nextSegment.itemIndex = curSegment.itemIndex + meshNode.rightChildOffset;
					nextSegment.bound.center = (smp + bgp) / 2;
					nextSegment.bound.extents = Abs(bgp - nextSegment.bound.center);

					stackIndex++;
				}

				// 3. calculate bound for smaller center BB, 
				bgp = b.center + b.extents;
				smp = b.center - b.extents;
				bgp[axis] = meshNode.underMaxHyperPlaneOffset;

				// 4. calc intersection by smaller center BB
				t1 = (smp.x - rayInMS.origin.x) * dir_frac.x;
				t2 = (bgp.x - rayInMS.origin.x) * dir_frac.x;
				t3 = (smp.y - rayInMS.origin.y) * dir_frac.y;
				t4 = (bgp.y - rayInMS.origin.y) * dir_frac.y;
				t5 = (smp.z - rayInMS.origin.z) * dir_frac.z;
				t6 = (bgp.z - rayInMS.origin.z) * dir_frac.z;

				tmin = fmax(fmax(fmin(t1, t2), fmin(t3, t4)), fmin(t5, t6));
				tmax = fmin(fmin(fmax(t1, t2), fmax(t3, t4)), fmax(t5, t6));

				if (tmax >= 0 && tmin <= tmax)
				{
					ASSERT(stackIndex + 1 < stackCapacity);
					StaticMeshKDTreeTraversalSegment& nextSegment = stack->traversalStack[stackIndex];
					nextSegment.itemIndex = curSegment.itemIndex + 1;
					nextSegment.bound.center = (smp + bgp) / 2;
					nextSegment.bound.extents = Abs(bgp - nextSegment.bound.center);

					stackIndex++;
				}
			}
			else
			{
				int primitiveIndex = meshNode.primitiveIndex1;

			ACCEL_AGG_CALC_RAY_TRIANGLE_INTERSECTONLY:

				int i0 = mc->indices[primitiveIndex + 0],
					i1 = mc->indices[primitiveIndex + 1],
					i2 = mc->indices[primitiveIndex + 2];

				Vector3f bc;
				const Vector3f &p0 = mc->positions[i0], &p1 = mc->positions[i1], &p2 = mc->positions[i2];

				if (IntersectRayAndTriangle(rayInMS, p0, p1, p2, bc))
				{
					Vector3f normal = (bc.x * mc->normals[i0] + bc.y * mc->normals[i1] + bc.z * mc->normals[i2]).normalized();

					if (Dot(normal, -rayInMS.direction) <= 0) 
						goto ACCEL_AGG_CALC_RAY_TRIANGLE_NEXT_PRIMTIVE;

					int sbidx = mc->GetSubmeshIndexFromIndex(i0);

					if (mc->uvs && materials[mrc->materialArrayPtr[sbidx]].URPLit.IsAlphaClipping())
					{
						const Vector2f& uv0 = mc->uvs[i0], uv1 = mc->uvs[i1], uv2 = mc->uvs[i2];
						Vector2f uv = bc.x * uv0 + bc.y * uv1 + bc.z * uv2;

						int baseMapIndex = materials[mrc->materialArrayPtr[sbidx]].URPLit.baseMapIndex;
						ColorRGBA c = textures[baseMapIndex].Sample8888(uv);

						if (c.a == 0)
							goto ACCEL_AGG_CALC_RAY_TRIANGLE_NEXT_PRIMTIVE;
					}

					return true;
				}

			ACCEL_AGG_CALC_RAY_TRIANGLE_NEXT_PRIMTIVE:

				if (primitiveIndex == meshNode.primitiveIndex1 && meshNode.primitiveIndex2 < INT_MAX)
				{
					primitiveIndex = meshNode.primitiveIndex2;
					goto ACCEL_AGG_CALC_RAY_TRIANGLE_INTERSECTONLY;
				}
			}
		}

		return false;
	}

	__host__ bool AcceleratedAggregate::BuildDynamicMeshBVH(IN MeshChunk* c, OUT DynamicMeshBuildData* data)
	{


		return true;
	}

	__host__ bool AcceleratedAggregate::UpdateDynamicMeshBVH(IN MeshChunk* c, OUT DynamicMeshNode** nodeArray)
	{
		return false;
	}

	__host__ __device__ bool AcceleratedAggregate::TraversalDynamicMeshBVH(
		const Ray& rayInMS,
		const MeshChunk* mc, const MeshRendererChunk* mrc,
		const MaterialChunk* materials, const Texture2DChunk* textures,
		const DynamicMeshNode* meshNodes, DynamicMeshBVHTraversalStack* stack,
		INOUT float& distance, OUT SurfaceIntersection& isect
	)
	{
		return false;
	}
	__host__ __device__ bool AcceleratedAggregate::TraversalDynamicMeshBVH(
		const Ray& rayInMS,
		const MeshChunk* mc, const MeshRendererChunk* mrc,
		const MaterialChunk* materials, const Texture2DChunk* textures,
		const DynamicMeshNode* meshNodes, DynamicMeshBVHTraversalStack* stack
	)
	{
		return false;
	}

	__host__ AcceleratedAggregate* AcceleratedAggregate::GetAggregateHost(const FrameMutableInput* hmin, const FrameImmutableInput* himin, int threadCount)
	{
		int transformStackSize = hmin->meshRendererBufferLen + hmin->skinnedMeshRendererBufferLen + hmin->lightBufferLen,
			staticMeshStackSize = 0,
			dynamicMeshStackSize = 0;

		for (int i = 0; i < himin->meshBufferLen; i++)
			staticMeshStackSize = max(himin->meshBuffer[i].indexCount / 3, staticMeshStackSize);

		for (int i = 0; i < himin->skinnedMeshBufferLen; i++)
			dynamicMeshStackSize = max(himin->skinnedMeshBuffer[i].indexCount / 3, dynamicMeshStackSize);

		TransformBVHBuildData* transformBuild = (TransformBVHBuildData*)malloc(sizeof(TransformBVHBuildData));
		ASSERT(
			AcceleratedAggregate::BuildTransformBVH(
				hmin->meshRendererBufferLen, hmin->meshRendererBuffer,
				hmin->skinnedMeshRendererBufferLen, hmin->skinnedMeshRendererBuffer,
				hmin->lightBufferLen, hmin->lightBuffer,
				transformBuild
			)
		);

		TransformBVHTraversalStack* transformStack = (TransformBVHTraversalStack*)malloc(sizeof(TransformBVHTraversalStack) * threadCount);
		for (int i = 0; i < threadCount; i++)
		{
			transformStack[i].stackCapacity = transformStackSize;
			transformStack[i].traversalStack = (TransformBVHTraversalSegment*)malloc(sizeof(TransformBVHTraversalSegment));
		}

		StaticMeshBuildData* hostStaticGroupBuild = (StaticMeshBuildData*)malloc(sizeof(StaticMeshBuildData) * himin->meshBufferLen);
		for (int i = 0; i < himin->meshBufferLen; i++)
			ASSERT(
				AcceleratedAggregate::BuildStaticMeshKDTree(
					himin->meshBuffer + i,
					staticMeshMinPrim,
					hostStaticGroupBuild + i
				)
			);
		StaticMeshKDTreeTraversalStack* staticMeshStack = (StaticMeshKDTreeTraversalStack*)malloc(sizeof(StaticMeshKDTreeTraversalStack) * threadCount);
		for (int i = 0; i < threadCount; i++)
		{
			staticMeshStack[i].stackCapacity = staticMeshStackSize;
			staticMeshStack[i].traversalStack = (StaticMeshKDTreeTraversalSegment*)malloc(sizeof(StaticMeshKDTreeTraversalSegment));
		}

		DynamicMeshBuildData* hostDynamicGroupBuild = (DynamicMeshBuildData*)malloc(sizeof(DynamicMeshBuildData) * himin->skinnedMeshBufferLen);
		for (int i = 0; i < himin->skinnedMeshBufferLen; i++)
			ASSERT(
				AcceleratedAggregate::BuildDynamicMeshBVH(
					himin->skinnedMeshBuffer + i,
					dynamicMeshMinPrim,
					hostDynamicGroupBuild + i
				)
			);
		DynamicMeshBVHTraversalStack* dynamicMeshStack = (DynamicMeshBVHTraversalStack*)malloc(sizeof(DynamicMeshBVHTraversalStack));
		for (int i = 0; i < threadCount; i++)
		{
			dynamicMeshStack[i].stackCapacity = dynamicMeshStackSize;
			dynamicMeshStack[i].traversalStack = (DynamicMeshBVHTraversalSegment*)malloc(sizeof(DynamicMeshBVHTraversalSegment));
		}

		return new AcceleratedAggregate(hmin, himin, transformBuild, transformStack, hostStaticGroupBuild, staticMeshStack, hostDynamicGroupBuild, dynamicMeshStack);
	}

	__global__ void SetAccelAggregate(
		AcceleratedAggregate* allocated,
		IMultipleInput* din, int mutableIndex,
		TransformBVHBuildData* transformNodeGroup, TransformBVHTraversalStack* transformNodeTraversalStack,
		StaticMeshBuildData* staticMeshNodeGroups, StaticMeshKDTreeTraversalStack* staticMeshTraversalStack,
		DynamicMeshBuildData* dynamicMeshNodeGroups, DynamicMeshBVHTraversalStack* dynamicMeshTraversalStack
	)
	{
		if (threadIdx.x == 0)
		{
			allocated = 
				new (allocated)AcceleratedAggregate(
					din->GetMutable(mutableIndex), din->GetImmutable(), 
					transformNodeGroup, transformNodeTraversalStack, 
					staticMeshNodeGroups, staticMeshTraversalStack, 
					dynamicMeshNodeGroups, dynamicMeshTraversalStack
				);
		}
	}

	__host__ AcceleratedAggregate* AcceleratedAggregate::GetAggregateDevice(IMultipleInput* hin, IMultipleInput* din, int mutableIndex, int threadCount)
	{
		const FrameImmutableInput* himin = hin->GetImmutable();
		const FrameMutableInput* hmin = hin->GetMutable(mutableIndex);

		int transformStackSize = hmin->meshRendererBufferLen + hmin->skinnedMeshRendererBufferLen + hmin->lightBufferLen, 
			staticMeshStackSize = 0,
			dynamicMeshStackSize = 0;

		for (int i = 0; i < himin->meshBufferLen; i++)
			staticMeshStackSize = max(himin->meshBuffer[i].indexCount / 3, staticMeshStackSize);

		for (int i = 0; i < himin->skinnedMeshBufferLen; i++)
			dynamicMeshStackSize = max(himin->skinnedMeshBuffer[i].indexCount / 3, dynamicMeshStackSize);

		// transform nodes
		TransformBVHBuildData*	hostTransformBuild = (TransformBVHBuildData*)malloc(sizeof(TransformBVHBuildData));
		AcceleratedAggregate::BuildTransformBVH(
			hmin->meshRendererBufferLen, hmin->meshRendererBuffer,
			hmin->skinnedMeshRendererBufferLen, hmin->skinnedMeshRendererBuffer,
			hmin->lightBufferLen, hmin->lightBuffer,
			hostTransformBuild
		);

		TransformNode* transformNodes = hostTransformBuild->transformNodes;
		hostTransformBuild->transformNodes = (TransformNode*)MAllocDevice(sizeof(TransformNode) * hostTransformBuild->capacity);
		gpuErrchk(cudaMemcpy(hostTransformBuild->transformNodes, transformNodes, sizeof(TransformNode) * hostTransformBuild->capacity, cudaMemcpyKind::cudaMemcpyHostToDevice));
		free(transformNodes);

		TransformBVHBuildData* deviceTransformBuild = (TransformBVHBuildData*)MAllocDevice(sizeof(TransformBVHBuildData));
		gpuErrchk(cudaMemcpy(deviceTransformBuild, hostTransformBuild, sizeof(TransformBVHBuildData), cudaMemcpyKind::cudaMemcpyHostToDevice));
		free(hostTransformBuild);

		// transform stack
		TransformBVHTraversalStack* hostTransformStack = (TransformBVHTraversalStack*)malloc(sizeof(TransformBVHTraversalStack) * threadCount);
		for (int i = 0; i < threadCount; i++)
		{
			hostTransformStack[i].stackCapacity = transformStackSize;
			hostTransformStack[i].traversalStack = (TransformBVHTraversalSegment*)MAllocDevice(sizeof(TransformBVHTraversalSegment) * transformStackSize);
		}

		TransformBVHTraversalStack* deviceTransformStack = (TransformBVHTraversalStack*)MAllocDevice(sizeof(TransformBVHTraversalStack)* threadCount);
		gpuErrchk(cudaMemcpy(deviceTransformStack, hostTransformStack, sizeof(TransformBVHTraversalStack) * threadCount, cudaMemcpyKind::cudaMemcpyHostToDevice));
		free(hostTransformStack);

		// static mesh nodes 
		StaticMeshBuildData* hostStaticGroupBuild = (StaticMeshBuildData*)malloc(sizeof(StaticMeshBuildData) * himin->meshBufferLen);
		for (int i = 0; i < himin->meshBufferLen; i++)
		{
			AcceleratedAggregate::BuildStaticMeshKDTree(
				himin->meshBuffer + i,
				hostStaticGroupBuild + i
			);

			StaticMeshNode* nodes = hostStaticGroupBuild[i].meshNodes;
			hostStaticGroupBuild[i].meshNodes = (StaticMeshNode*)MAllocDevice(sizeof(StaticMeshNode) * hostStaticGroupBuild[i].count);
			gpuErrchk(cudaMemcpy(hostStaticGroupBuild[i].meshNodes, nodes, sizeof(StaticMeshNode) * hostStaticGroupBuild[i].count, cudaMemcpyKind::cudaMemcpyHostToDevice));
			free(nodes);
		}
		StaticMeshBuildData* deviceStaticGroupBuild = (StaticMeshBuildData*)MAllocDevice(sizeof(StaticMeshBuildData) * himin->meshBufferLen);
		gpuErrchk(cudaMemcpy(deviceStaticGroupBuild, hostStaticGroupBuild, sizeof(StaticMeshBuildData) * himin->meshBufferLen, cudaMemcpyKind::cudaMemcpyHostToDevice));
		free(hostStaticGroupBuild);

		// static mesh stack
		StaticMeshKDTreeTraversalStack* hostStaticMeshStack = (StaticMeshKDTreeTraversalStack*)malloc(sizeof(StaticMeshKDTreeTraversalStack) * threadCount);
		for (int i = 0; i < threadCount; i++)
		{
			hostStaticMeshStack[i].stackCapacity = staticMeshStackSize;
			hostStaticMeshStack[i].traversalStack = (StaticMeshKDTreeTraversalSegment*)MAllocDevice(sizeof(StaticMeshKDTreeTraversalSegment) * staticMeshStackSize);
		}

		StaticMeshKDTreeTraversalStack* deviceStaticMeshStack = (StaticMeshKDTreeTraversalStack*)MAllocDevice(sizeof(StaticMeshKDTreeTraversalStack) * threadCount);
		gpuErrchk(cudaMemcpy(deviceStaticMeshStack, hostStaticMeshStack, sizeof(StaticMeshKDTreeTraversalStack) * threadCount, cudaMemcpyKind::cudaMemcpyHostToDevice));
		free(hostStaticMeshStack);

		// dynamic mesh nodes
		DynamicMeshBuildData* hostDynamicGroupBuild = (DynamicMeshBuildData*)malloc(sizeof(DynamicMeshBuildData) * himin->skinnedMeshBufferLen);
		for (int i = 0; i < himin->skinnedMeshBufferLen; i++)
		{
			AcceleratedAggregate::BuildDynamicMeshBVH(
				himin->skinnedMeshBuffer + i,
				hostDynamicGroupBuild + i
			);

			DynamicMeshNode* nodes = hostDynamicGroupBuild[i].meshNodes;
			hostDynamicGroupBuild[i].meshNodes = (DynamicMeshNode*)MAllocDevice(sizeof(DynamicMeshNode) * hostDynamicGroupBuild[i].count);;
			gpuErrchk(cudaMemcpy(hostDynamicGroupBuild[i].meshNodes, nodes, sizeof(DynamicMeshNode) * hostDynamicGroupBuild[i].count, cudaMemcpyKind::cudaMemcpyHostToDevice));
			free(nodes);
		}
		DynamicMeshBuildData* deviceDynamicGroupBuild = (DynamicMeshBuildData*)MAllocDevice(sizeof(DynamicMeshBuildData) * himin->skinnedMeshBufferLen);
		gpuErrchk(cudaMemcpy(deviceDynamicGroupBuild, hostDynamicGroupBuild, sizeof(DynamicMeshBuildData) * himin->skinnedMeshBufferLen, cudaMemcpyKind::cudaMemcpyHostToDevice));
		free(hostDynamicGroupBuild);

		// dynamic mesh stack
		DynamicMeshBVHTraversalStack* hostDynamicMeshStack = (DynamicMeshBVHTraversalStack*)malloc(sizeof(DynamicMeshBVHTraversalStack) * threadCount);
		for (int i = 0; i < threadCount; i++)
		{
			hostDynamicMeshStack[i].stackCapacity = dynamicMeshStackSize;
			hostDynamicMeshStack[i].traversalStack = (DynamicMeshBVHTraversalSegment*)MAllocDevice(sizeof(DynamicMeshBVHTraversalSegment) * dynamicMeshStackSize);
		}

		DynamicMeshBVHTraversalStack* deviceDynamicMeshStack = (DynamicMeshBVHTraversalStack*)MAllocDevice(sizeof(DynamicMeshBVHTraversalStack) * threadCount);
		gpuErrchk(cudaMemcpy(deviceDynamicMeshStack, hostDynamicMeshStack, sizeof(DynamicMeshBVHTraversalStack) * threadCount, cudaMemcpyKind::cudaMemcpyHostToDevice));
		free(hostDynamicMeshStack);

		AcceleratedAggregate* deviceAllocated = (AcceleratedAggregate*)MAllocDevice(sizeof(AcceleratedAggregate));
		SetAccelAggregate<<<1, 1>>>(
			deviceAllocated,
			din, mutableIndex,
			deviceTransformBuild, deviceTransformStack,
			deviceStaticGroupBuild, deviceStaticMeshStack,
			deviceDynamicGroupBuild, deviceDynamicMeshStack
		);

		return deviceAllocated;
	}

	__host__ void AcceleratedAggregate::DestroyDeviceAggregate(AcceleratedAggregate* agg, int threadCount)
	{
		AcceleratedAggregate hagg;
		gpuErrchk(cudaMemcpy(&hagg, agg, sizeof(AcceleratedAggregate), cudaMemcpyKind::cudaMemcpyDeviceToHost));

		TransformBVHBuildData hostTransformBuild;
		gpuErrchk(cudaMemcpy(&hostTransformBuild, hagg.mTransformNodeGroup, sizeof(TransformBVHBuildData), cudaMemcpyKind::cudaMemcpyHostToDevice));

		gpuErrchk(cudaFree(hostTransformBuild.transformNodes));
		gpuErrchk(cudaFree(hagg.mTransformNodeGroup));

		for (int i = 0; i < threadCount; i++)
		{
			TransformBVHTraversalStack hstack;
			gpuErrchk(cudaMemcpy(&hstack, hagg.mTransformNodeTraversalStack + i, sizeof(TransformBVHTraversalStack), cudaMemcpyKind::cudaMemcpyDeviceToHost));
			gpuErrchk(cudaFree(hstack.traversalStack));
		}
		gpuErrchk(cudaFree(hagg.mTransformNodeTraversalStack));

		StaticMeshBuildData hostStaticGroupBuild;
		for (int i = 0; i < hagg.mStaticMeshCount; i++)
		{
			gpuErrchk(cudaMemcpy(&hostStaticGroupBuild, hagg.mStaticMeshNodeGroups + i, sizeof(StaticMeshBuildData), cudaMemcpyKind::cudaMemcpyHostToDevice));
			gpuErrchk(cudaFree(hostStaticGroupBuild.meshNodes));
		}
		gpuErrchk(cudaFree(hagg.mStaticMeshNodeGroups));

		for (int i = 0; i < threadCount; i++)
		{
			StaticMeshKDTreeTraversalStack hostStaticMeshStack;
			gpuErrchk(cudaMemcpy(&hostStaticMeshStack, hagg.mStaticMeshTraversalStack + i, sizeof(StaticMeshKDTreeTraversalStack), cudaMemcpyKind::cudaMemcpyDeviceToHost));
			gpuErrchk(cudaFree(hostStaticMeshStack.traversalStack));
		}
		gpuErrchk(cudaFree(hagg.mStaticMeshTraversalStack));

		DynamicMeshBuildData hostDynamicGroupBuild;
		for (int i = 0; i < hagg.mSkinnedMeshCount; i++)
		{
			gpuErrchk(cudaMemcpy(&hostDynamicGroupBuild, hagg.mDynamicMeshNodeGroups + i, sizeof(DynamicMeshBuildData), cudaMemcpyKind::cudaMemcpyHostToDevice));
			gpuErrchk(cudaFree(hostDynamicGroupBuild.meshNodes));
		}
		gpuErrchk(cudaFree(hagg.mDynamicMeshNodeGroups));

		for (int i = 0; i < threadCount; i++)
		{
			DynamicMeshBVHTraversalStack hostDynamicMeshStack;
			gpuErrchk(cudaMemcpy(&hostDynamicMeshStack, hagg.mDynamicMeshTraversalStack + i, sizeof(DynamicMeshBVHTraversalStack), cudaMemcpyKind::cudaMemcpyDeviceToHost));
			gpuErrchk(cudaFree(hostDynamicMeshStack.traversalStack));
		}
		gpuErrchk(cudaFree(hagg.mDynamicMeshTraversalStack));
	}
}
