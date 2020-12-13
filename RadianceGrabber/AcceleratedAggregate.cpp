#include <vector>
#include <list>
#include <algorithm>
#include <cfloat>

#include "AcceleratedAggregate.h"
#include "AcceleratedAggregateInternal.h"
#include "Util.h"
#include "Sample.cuh"


#define ITERATIVE_COST_COUNT 1


namespace RadGrabber
{
	__host__ AcceleratedAggregate::AcceleratedAggregate()
	{
	}

	__host__ __device__ AcceleratedAggregate::AcceleratedAggregate(
		const FrameMutableInput* min, const FrameImmutableInput* imin, int countOfStack,
		TransformBVHBuildData* transformNodeGroup, TransformBVHTraversalStack* transformNodeTraversalStack,
		StaticMeshBuildData* staticMeshNodeGroups, StaticMeshKDTreeTraversalStack* staticMeshTraversalStack,
		DynamicMeshBuildData* dynamicMeshNodeGroups, DynamicMeshBVHTraversalStack* dynamicMeshTraversalStack
	) :
		mMRs(min->meshRendererBuffer), mMRCount(min->meshRendererBufferLen),
		mSkinnedMRs(min->skinnedMeshRendererBuffer), mSkinnedMRCount(min->skinnedMeshRendererBufferLen),
		mLights(min->lightBuffer), mLightCount(min->lightBufferLen),
		mTextures(imin->textureBuffer), mTextureCount(imin->textureBufferLen),
		mMaterials(min->materialBuffer), mMaterialCount(min->materialBufferLen),
		mStaticMeshCount(imin->meshBufferLen), mStaticMeshs(imin->meshBuffer),
		mSkinnedMeshCount(imin->skinnedMeshBufferLen), mSkinnedMeshs(imin->skinnedMeshBuffer),
		mTransformNodeGroup(transformNodeGroup), mTransformNodeTraversalStack(transformNodeTraversalStack),
		mStaticMeshNodeGroups(staticMeshNodeGroups), mStaticMeshTraversalStack(staticMeshTraversalStack),
		mDynamicMeshNodeGroups(dynamicMeshNodeGroups), mDynamicMeshTraversalStack(dynamicMeshTraversalStack),
		countOfStack(countOfStack)
	{
	}

	__host__ AcceleratedAggregate::~AcceleratedAggregate()
	{
	}

	__host__ __device__ bool AcceleratedAggregate::Intersect(const Ray & rayInWS, SurfaceIntersection & isect, int threadIndex, uint flag) const
	{
		isect.isHit = 0;
		float distance = FLT_MAX;

		if (!TraversalTransformBVH(rayInWS, mTransformNodeGroup->transformNodes, mTransformNodeTraversalStack + threadIndex))
			return false;

		for (int intsectBoundIndex = 0; intsectBoundIndex < mTransformNodeTraversalStack[threadIndex].listCount; intsectBoundIndex++)
		{
			const TransformNode transformNode = mTransformNodeGroup->transformNodes[mTransformNodeTraversalStack[threadIndex].nodeIndexList[intsectBoundIndex]];

			switch (transformNode.kind)
			{
			case TransformItemKind::MeshRenderer:
			{
				if ((flag & (uint)AggregateItem::StaticMesh)) 
					continue;

				TraversalStaticMeshKDTree(
					rayInWS,
					Ray(
						mMRs[transformNode.itemIndex].transformInverseMatrix.TransformPoint(rayInWS.origin),
						mMRs[transformNode.itemIndex].transformInverseMatrix.TransformVector(rayInWS.direction)
					),
					0, FLT_MAX,
					mStaticMeshs + mMRs[transformNode.itemIndex].meshRefIndex, mMRs + transformNode.itemIndex,
					mMaterials, mTextures,
					mStaticMeshNodeGroups[mMRs[transformNode.itemIndex].meshRefIndex].meshNodes, mStaticMeshTraversalStack + threadIndex,
					distance, isect
				);
				break;
			}
			case TransformItemKind::SkinnedMeshRenderer:
			{
				if ((flag & (uint)AggregateItem::DynamicMesh))
					continue;

				TraversalDynamicMeshBVH(
					Ray(
						mSkinnedMRs[transformNode.itemIndex].transformInverseMatrix.TransformPoint(rayInWS.origin),
						mSkinnedMRs[transformNode.itemIndex].transformInverseMatrix.TransformVector(rayInWS.direction)
					),
					0, FLT_MAX,
					mSkinnedMeshs + mSkinnedMRs[transformNode.itemIndex].skinnedMeshRefIndex, (MeshRendererChunk*)(mSkinnedMRs + transformNode.itemIndex),
					mMaterials, mTextures,
					mDynamicMeshNodeGroups[mSkinnedMRs[transformNode.itemIndex].skinnedMeshRefIndex].meshNodes, mDynamicMeshTraversalStack + threadIndex,
					distance, isect
				);
				break;
			}
			case TransformItemKind::Light:
			{
				if ((flag & (uint)AggregateItem::Light))
					continue;

				if (mLights[transformNode.itemIndex].IntersectRay(rayInWS, isect, distance))
				{
					isect.isHit = 1;
					isect.isGeometry = 0;
					isect.itemIndex = transformNode.itemIndex;
					//mLights[transformNode.itemIndex].Sample(rayInWS, isect, isect.color);
				}
				break;
			}
			}
		}

		return isect.isHit;
	}

	__host__ __device__ bool AcceleratedAggregate::Intersect(const Ray& rayInWS, float minDistance, float maxDistance, int threadIndex, uint flag) const
	{
		float distance = FLT_MAX;

		if (!TraversalTransformBVH(rayInWS, mTransformNodeGroup->transformNodes, mTransformNodeTraversalStack + threadIndex))
			return false;

		for (int intsectBoundIndex = 0; intsectBoundIndex < mTransformNodeTraversalStack[threadIndex].listCount; intsectBoundIndex++)
		{
			const TransformNode transformNode = mTransformNodeGroup->transformNodes[mTransformNodeTraversalStack[threadIndex].nodeIndexList[intsectBoundIndex]];

			switch (transformNode.kind)
			{
			case TransformItemKind::MeshRenderer:
			{
				if ((flag & (uint)AggregateItem::StaticMesh))
					continue;

				TraversalStaticMeshKDTree(
					rayInWS,
					Ray(
						mMRs[transformNode.itemIndex].transformInverseMatrix.TransformPoint(rayInWS.origin),
						mMRs[transformNode.itemIndex].transformInverseMatrix.TransformVector(rayInWS.direction)
					),
					minDistance, maxDistance,
					mStaticMeshs + mMRs[transformNode.itemIndex].meshRefIndex, mMRs + transformNode.itemIndex,
					mMaterials, mTextures,
					mStaticMeshNodeGroups[mMRs[transformNode.itemIndex].meshRefIndex].meshNodes, mStaticMeshTraversalStack + threadIndex,
					distance
				);
				break;
			}
			case TransformItemKind::SkinnedMeshRenderer:
			{
				if ((flag & (uint)AggregateItem::DynamicMesh))
					continue;

				TraversalDynamicMeshBVH(
					Ray(
						mSkinnedMRs[transformNode.itemIndex].transformInverseMatrix.TransformPoint(rayInWS.origin),
						mSkinnedMRs[transformNode.itemIndex].transformInverseMatrix.TransformVector(rayInWS.direction)
					),
					0, FLT_MAX,
					mSkinnedMeshs + mSkinnedMRs[transformNode.itemIndex].skinnedMeshRefIndex, (MeshRendererChunk*)(mSkinnedMRs + transformNode.itemIndex),
					mMaterials, mTextures,
					mDynamicMeshNodeGroups[mSkinnedMRs[transformNode.itemIndex].skinnedMeshRefIndex].meshNodes, mDynamicMeshTraversalStack + threadIndex,
					distance
				);
				break;
			}
			case TransformItemKind::Light:
			{
				if ((flag & (uint)AggregateItem::Light))
					continue;

				if (mLights[transformNode.itemIndex].IntersectRay(rayInWS, minDistance, maxDistance, distance))
				{
					//mLights[transformNode.itemIndex].Sample(rayInWS, isect, isect.color);
				}
				break;
			}
			}
		}

		return distance != FLT_MAX;
	}


	__host__ __device__ bool AcceleratedAggregate::IntersectOnly(const Ray & rayInWS, int threadIndex, uint flag) const
	{
		if (!TraversalTransformBVH(rayInWS, mTransformNodeGroup->transformNodes, mTransformNodeTraversalStack + threadIndex))
			return false;

		for (int intsectBoundIndex = 0; intsectBoundIndex < mTransformNodeTraversalStack[threadIndex].listCount; intsectBoundIndex++)
		{
			int nodeIndex = mTransformNodeTraversalStack[threadIndex].nodeIndexList[intsectBoundIndex];
			const TransformNode& transformNode = mTransformNodeGroup->transformNodes[nodeIndex];

			switch (transformNode.kind)
			{
			case TransformItemKind::MeshRenderer:
			{
				if ((flag & (uint)AggregateItem::StaticMesh))
					continue;

				MeshRendererChunk* mr = mMRs + transformNode.itemIndex;
				Ray rayInMS =
					Ray(
						mr->transformInverseMatrix.TransformPoint(rayInWS.origin),
						mr->transformInverseMatrix.TransformVector(rayInWS.direction)
					);

				if (
					TraversalStaticMeshKDTree(
						rayInWS, rayInMS,
						mStaticMeshs + mr->meshRefIndex, mr,
						mMaterials, mTextures,
						mStaticMeshNodeGroups[mr->meshRefIndex].meshNodes, mStaticMeshTraversalStack + threadIndex
					))
					return true;

				break;
			}
			case TransformItemKind::SkinnedMeshRenderer:
			{
				if ((flag & (uint)AggregateItem::DynamicMesh))
					continue;

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
				if ((flag & (uint)AggregateItem::Light))
					continue;

				if (mLights[transformNode.itemIndex].IntersectRayOnly(rayInWS))
					return true;

				break;
			}
		}
		return false;
	}

	__host__ __device__ bool AcceleratedAggregate::IterativeIntersect(const Ray & rayInWS, Ray& rayInMS, SurfaceIntersection & isect, int threadIndex, int& iterateCount, AATraversalSegment& travSeg, uint flag) const
	{
		if (travSeg.isLowerTransform)
			goto ITERATIVE_INTERSECT_LOWER;

	ITERATIVE_INTERSECT_UPPER:

		if (
			!TraversalTransformBVH(
				rayInWS,
				mTransformNodeGroup->transformNodes, mTransformNodeTraversalStack + threadIndex,
				iterateCount, travSeg
			)
			)
			return false;

		travSeg.isLowerTransform = 1;
		travSeg.initilaized = 0;
		travSeg.lastDistance = FLT_MAX;
		travSeg.findPrimitive = 0;
		isect.isHit = 0;

		if (mTransformNodeTraversalStack[threadIndex].listCount == 0)
			return true;
		if (iterateCount <= 0)
			return false;

	ITERATIVE_INTERSECT_LOWER:

		while (travSeg.upperStackIndex < mTransformNodeTraversalStack[threadIndex].listCount)
		{
			const TransformNode& transformNode = mTransformNodeGroup->transformNodes[mTransformNodeTraversalStack[threadIndex].nodeIndexList[travSeg.upperStackIndex]];

			switch (transformNode.kind)
			{
			case TransformItemKind::MeshRenderer:
			{
				if ((flag & (uint)AggregateItem::StaticMesh))
					continue;

				if (!travSeg.initilaized)
				{
					rayInMS =
						Ray(
							mMRs[transformNode.itemIndex].transformInverseMatrix.TransformPoint(rayInWS.origin),
							mMRs[transformNode.itemIndex].transformInverseMatrix.TransformVector(rayInWS.direction)
						);
				}

				if (
					TraversalStaticMeshKDTree(
						rayInWS, rayInMS,
						0, FLT_MAX,
						mStaticMeshs + mMRs[transformNode.itemIndex].meshRefIndex, &mMRs[transformNode.itemIndex],
						mMaterials, mTextures,
						iterateCount, travSeg,
						mStaticMeshNodeGroups[mMRs[transformNode.itemIndex].meshRefIndex].meshNodes, mStaticMeshTraversalStack + threadIndex,
						isect
					)
					)
				{
					travSeg.initilaized = 0;
				}
				break;
			}
			case TransformItemKind::SkinnedMeshRenderer:
			{
				if ((flag & (uint)AggregateItem::DynamicMesh))
					continue;

				if (!travSeg.initilaized)
				{
					rayInMS =
						Ray(
							mSkinnedMRs[transformNode.itemIndex].transformInverseMatrix.TransformPoint(rayInWS.origin),
							mSkinnedMRs[transformNode.itemIndex].transformInverseMatrix.TransformVector(rayInWS.direction)
						);
				}

				if (
					TraversalDynamicMeshBVH(
						rayInMS,
						0, FLT_MAX,
						mSkinnedMeshs + mSkinnedMRs[transformNode.itemIndex].skinnedMeshRefIndex, (MeshRendererChunk*)&mSkinnedMRs[transformNode.itemIndex],
						mMaterials, mTextures,
						iterateCount, travSeg,
						mDynamicMeshNodeGroups[mSkinnedMRs[transformNode.itemIndex].skinnedMeshRefIndex].meshNodes, mDynamicMeshTraversalStack + threadIndex,
						isect
					)
					)
				{
					travSeg.initilaized = 0;
				}
				break;
			}
			case TransformItemKind::Light:
			{
				if ((flag & (uint)AggregateItem::Light))
					continue;

				if (mLights[transformNode.itemIndex].IntersectRay(rayInWS, isect, travSeg.lastDistance))
				{
					travSeg.findPrimitive = 1;
					isect.isGeometry = 0;
					isect.itemIndex = transformNode.itemIndex;
					//mLights[transformNode.itemIndex].Sample(rayInWS, isect, isect.color);
				}
				break;
			}
			}

			if (iterateCount <= 0)
				return false;

			++travSeg.upperStackIndex;
		}
		;

		return travSeg.upperStackIndex == mTransformNodeTraversalStack[threadIndex].listCount;
	}
	__host__ __device__ bool AcceleratedAggregate::IterativeIntersectOnly(const Ray & rayInWS, Ray& rayInMS, int threadIndex, int& iterateCount, AATraversalSegment& travSeg, uint flag) const
	{
		if (travSeg.isLowerTransform)
			goto ITERATIVE_INTERSECT_ONLY_LOWER;

	ITERATIVE_INTERSECT_ONLY_UPPER:
		if (
			!TraversalTransformBVH(
				rayInWS,
				mTransformNodeGroup->transformNodes, mTransformNodeTraversalStack + threadIndex,
				iterateCount, travSeg
			)
			)
			return false;

		travSeg.findPrimitive = 0;
		if (mTransformNodeTraversalStack[threadIndex].listCount == 0)
			return true;

		travSeg.isLowerTransform = 1;
		travSeg.initilaized = 0;
		travSeg.findPrimitive = 0;
		travSeg.upperStackIndex = 0;
		travSeg.lastDistance = 0;

		if (iterateCount <= 0)
			return false;

	ITERATIVE_INTERSECT_ONLY_LOWER:

		do
		{
			const TransformNode& transformNode = mTransformNodeGroup->transformNodes[mTransformNodeTraversalStack[threadIndex].nodeIndexList[travSeg.upperStackIndex]];

			switch (transformNode.kind)
			{
			case TransformItemKind::MeshRenderer:
			{
				if ((flag & (uint)AggregateItem::StaticMesh))
					continue;

				if (!travSeg.initilaized)
				{
					rayInMS =
						Ray(
							mMRs[transformNode.itemIndex].transformInverseMatrix.TransformPoint(rayInWS.origin),
							mMRs[transformNode.itemIndex].transformInverseMatrix.TransformVector(rayInWS.direction)
						);
				}

				if (
					TraversalStaticMeshKDTree(
						rayInWS, rayInMS,
						0, FLT_MAX,
						mStaticMeshs + mMRs[transformNode.itemIndex].meshRefIndex, &mMRs[transformNode.itemIndex],
						mMaterials, mTextures,
						iterateCount, travSeg,
						mStaticMeshNodeGroups[mMRs[transformNode.itemIndex].meshRefIndex].meshNodes, mStaticMeshTraversalStack + threadIndex
					)
					)
				{
					travSeg.initilaized = 0;
				}
				break;
			}
			case TransformItemKind::SkinnedMeshRenderer:
			{
				if ((flag & (uint)AggregateItem::DynamicMesh))
					continue;

				if (!travSeg.initilaized)
				{
					rayInMS =
						Ray(
							mSkinnedMRs[transformNode.itemIndex].transformInverseMatrix.TransformPoint(rayInWS.origin),
							mSkinnedMRs[transformNode.itemIndex].transformInverseMatrix.TransformVector(rayInWS.direction)
						);
				}

				if (
					TraversalDynamicMeshBVH(
						rayInMS,
						0, FLT_MAX,
						mSkinnedMeshs + mSkinnedMRs[transformNode.itemIndex].skinnedMeshRefIndex, (MeshRendererChunk*)&mSkinnedMRs[transformNode.itemIndex],
						mMaterials, mTextures,
						iterateCount, travSeg,
						mDynamicMeshNodeGroups[mSkinnedMRs[transformNode.itemIndex].skinnedMeshRefIndex].meshNodes, mDynamicMeshTraversalStack + threadIndex
					)
					)
				{
					travSeg.initilaized = 0;
				}
				break;
			}
			case TransformItemKind::Light:
			{
				if ((flag & (uint)AggregateItem::Light))
					continue;

				if (mLights[transformNode.itemIndex].IntersectRayOnly(rayInWS))
				{
					travSeg.findPrimitive = 1;
					//mLights[transformNode.itemIndex].Sample(rayInWS, isect, isect.color);
				}
				break;
			}
			}

			if (iterateCount <= 0)
				return false;

		} while (++travSeg.upperStackIndex < mTransformNodeTraversalStack[threadIndex].listCount);

		return travSeg.lastDistance != FLT_MAX;
	}

	bool LinearBVHBuild(
		IN int meshRendererLen, IN MeshRendererChunk* meshRenderers,
		IN int skinnedMeshRendererLen, IN SkinnedMeshRendererChunk* skinnedMeshRenderers,
		IN int lightLen, IN LightChunk* lights,
		OUT TransformBVHBuildData* data)
	{
		data->count = data->capacity = meshRendererLen + skinnedMeshRendererLen + lightLen;
		data->transformNodes = (TransformNode*)malloc(sizeof(TransformNode) * data->count);

		for (int i = 0; i < data->count; i++)
		{
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

		return true;
	}

	__forceinline__ __host__ TransformRef* GetTransform(
		IN int itemIndex,
		IN int meshRendererLen, IN MeshRendererChunk* meshRenderers,
		IN int skinnedMeshRendererLen, IN SkinnedMeshRendererChunk* skinnedMeshRenderers,
		IN int lightLen, IN LightChunk* lights
	)
	{
		if (itemIndex < meshRendererLen)
		{
			return (TransformRef*)(meshRenderers + itemIndex);
		}
		else if (itemIndex < meshRendererLen + skinnedMeshRendererLen)
		{
			return (TransformRef*)(skinnedMeshRenderers + (itemIndex - meshRendererLen));
		}
		else if (itemIndex < meshRendererLen + skinnedMeshRendererLen + lightLen)
		{
			return (TransformRef*)(skinnedMeshRenderers + (itemIndex - meshRendererLen - lightLen));
		}

		return nullptr;
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
		BuildTransformBVH(
			hmin->meshRendererBufferLen, hmin->meshRendererBuffer,
			hmin->skinnedMeshRendererBufferLen, hmin->skinnedMeshRendererBuffer,
			hmin->lightBufferLen, hmin->lightBuffer,
			transformBuild
		);

		TransformBVHTraversalStack* transformStack = (TransformBVHTraversalStack*)malloc(sizeof(TransformBVHTraversalStack) * threadCount);
		for (int i = 0; i < threadCount; i++)
		{
			transformStack[i].listCapacity = transformStack[i].stackCapacity = transformStackSize;
			transformStack[i].traversalStack = (int*)malloc(sizeof(int) * transformStack[i].stackCapacity);
			transformStack[i].nodeIndexList = (int*)malloc(sizeof(int) * transformStack[i].listCapacity);
		}

		StaticMeshBuildData* hostStaticGroupBuild = (StaticMeshBuildData*)malloc(sizeof(StaticMeshBuildData) * himin->meshBufferLen);
		for (int i = 0; i < himin->meshBufferLen; i++)
			BuildStaticMeshKDTree(
				himin->meshBuffer + i,
				hostStaticGroupBuild + i
			);
		StaticMeshKDTreeTraversalStack* staticMeshStack = (StaticMeshKDTreeTraversalStack*)malloc(sizeof(StaticMeshKDTreeTraversalStack) * threadCount);
		for (int i = 0; i < threadCount; i++)
		{
			staticMeshStack[i].stackCapacity = staticMeshStackSize;
			staticMeshStack[i].traversalStack = (StaticMeshKDTreeTraversalSegment*)malloc(sizeof(StaticMeshKDTreeTraversalSegment));
		}

		DynamicMeshBuildData* hostDynamicGroupBuild = (DynamicMeshBuildData*)malloc(sizeof(DynamicMeshBuildData) * himin->skinnedMeshBufferLen);
		for (int i = 0; i < himin->skinnedMeshBufferLen; i++)
			BuildDynamicMeshBVH(
				himin->skinnedMeshBuffer + i,
				hostDynamicGroupBuild + i
			);
		DynamicMeshBVHTraversalStack* dynamicMeshStack = (DynamicMeshBVHTraversalStack*)malloc(sizeof(DynamicMeshBVHTraversalStack));
		for (int i = 0; i < threadCount; i++)
		{
			dynamicMeshStack[i].stackCapacity = dynamicMeshStackSize;
			dynamicMeshStack[i].traversalStack = (DynamicMeshBVHTraversalSegment*)malloc(sizeof(DynamicMeshBVHTraversalSegment));
		}

		return new AcceleratedAggregate(hmin, himin, threadCount, transformBuild, transformStack, hostStaticGroupBuild, staticMeshStack, hostDynamicGroupBuild, dynamicMeshStack);
	}

	__global__ void SetAccelAggregate(
		AcceleratedAggregate* allocated,
		IMultipleInput* din, int mutableIndex,
		TransformBVHBuildData* transformNodeGroup, TransformBVHTraversalStack* transformNodeTraversalStack,
		StaticMeshBuildData* staticMeshNodeGroups, StaticMeshKDTreeTraversalStack* staticMeshTraversalStack,
		DynamicMeshBuildData* dynamicMeshNodeGroups, DynamicMeshBVHTraversalStack* dynamicMeshTraversalStack,
		int threadCount
	)
	{
		new (allocated + mutableIndex)AcceleratedAggregate(
			din->GetMutable(mutableIndex), din->GetImmutable(), threadCount,
			transformNodeGroup, transformNodeTraversalStack,
			staticMeshNodeGroups, staticMeshTraversalStack,
			dynamicMeshNodeGroups, dynamicMeshTraversalStack
		);
	}

	__host__ void AcceleratedAggregate::GetAggregateDevice(int aagCount, AcceleratedAggregate* aag, IMultipleInput* hin, IMultipleInput* din, int stackCount)
	{
		const FrameImmutableInput* himin = hin->GetImmutable();
		int staticMeshStackSize = 0, dynamicMeshStackSize = 0, transformStackSize = 0;

		for (int i = hin->GetStartIndex(); i < hin->GetCount(); i++)
		{
			const FrameMutableInput* hmin = hin->GetMutable(i);
			transformStackSize = max(transformStackSize, hmin->meshRendererBufferLen + hmin->skinnedMeshRendererBufferLen + hmin->lightBufferLen);
		}
		for (int i = 0; i < himin->meshBufferLen; i++)
			staticMeshStackSize = max(himin->meshBuffer[i].indexCount / 3, staticMeshStackSize);
		for (int i = 0; i < himin->skinnedMeshBufferLen; i++)
			dynamicMeshStackSize = max(himin->skinnedMeshBuffer[i].indexCount / 3, dynamicMeshStackSize);

		// static mesh nodes 
		StaticMeshBuildData* hostStaticGroupBuild = (StaticMeshBuildData*)malloc(sizeof(StaticMeshBuildData) * himin->meshBufferLen);
		for (int i = 0; i < himin->meshBufferLen; i++)
		{
			BuildStaticMeshKDTree(
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
		StaticMeshKDTreeTraversalStack* hostStaticMeshStack = (StaticMeshKDTreeTraversalStack*)malloc(sizeof(StaticMeshKDTreeTraversalStack) * stackCount);
		StaticMeshKDTreeTraversalSegment* deviceStackSegments = (StaticMeshKDTreeTraversalSegment*)MAllocDevice(sizeof(StaticMeshKDTreeTraversalSegment) * staticMeshStackSize * stackCount);
		for (int i = 0; i < stackCount; i++)
		{
			hostStaticMeshStack[i].stackCapacity = staticMeshStackSize;
			hostStaticMeshStack[i].traversalStack = deviceStackSegments + i * staticMeshStackSize;
		}

		StaticMeshKDTreeTraversalStack* deviceStaticMeshStack = (StaticMeshKDTreeTraversalStack*)MAllocDevice(sizeof(StaticMeshKDTreeTraversalStack) * stackCount);
		gpuErrchk(cudaMemcpy(deviceStaticMeshStack, hostStaticMeshStack, sizeof(StaticMeshKDTreeTraversalStack) * stackCount, cudaMemcpyKind::cudaMemcpyHostToDevice));
		free(hostStaticMeshStack);

		// dynamic mesh nodes
		DynamicMeshBuildData* hostDynamicGroupBuild = (DynamicMeshBuildData*)malloc(sizeof(DynamicMeshBuildData) * himin->skinnedMeshBufferLen);
		for (int i = 0; i < himin->skinnedMeshBufferLen; i++)
		{
			BuildDynamicMeshBVH(
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
		DynamicMeshBVHTraversalStack* hostDynamicMeshStack = (DynamicMeshBVHTraversalStack*)malloc(sizeof(DynamicMeshBVHTraversalStack) * stackCount);
		DynamicMeshBVHTraversalSegment* deviceDynamicMeshStackSegments = (DynamicMeshBVHTraversalSegment*)MAllocDevice(sizeof(DynamicMeshBVHTraversalSegment) * dynamicMeshStackSize * stackCount);
		for (int i = 0; i < stackCount; i++)
		{
			hostDynamicMeshStack[i].stackCapacity = dynamicMeshStackSize;
			hostDynamicMeshStack[i].traversalStack = deviceDynamicMeshStackSegments + i * dynamicMeshStackSize;
		}

		DynamicMeshBVHTraversalStack* deviceDynamicMeshStack = (DynamicMeshBVHTraversalStack*)MAllocDevice(sizeof(DynamicMeshBVHTraversalStack) * stackCount);
		gpuErrchk(cudaMemcpy(deviceDynamicMeshStack, hostDynamicMeshStack, sizeof(DynamicMeshBVHTraversalStack) * stackCount, cudaMemcpyKind::cudaMemcpyHostToDevice));
		free(hostDynamicMeshStack);

		// transform stack
		TransformBVHTraversalStack* hostTransformStack = (TransformBVHTraversalStack*)malloc(sizeof(TransformBVHTraversalStack) * stackCount);
		int	*deviceTraversalStack = (int*)MAllocDevice(sizeof(int) * transformStackSize * stackCount),
			*deviceNodeIndexList = (int*)MAllocDevice(sizeof(int) * transformStackSize * stackCount);
		for (int i = 0; i < stackCount; i++)
		{
			hostTransformStack[i].listCapacity = hostTransformStack[i].stackCapacity = transformStackSize;
			hostTransformStack[i].traversalStack = deviceTraversalStack + i * transformStackSize;
			hostTransformStack[i].nodeIndexList = deviceNodeIndexList + i * transformStackSize;
		}

		TransformBVHTraversalStack* deviceTransformStack = (TransformBVHTraversalStack*)MAllocDevice(sizeof(TransformBVHTraversalStack) * stackCount);
		gpuErrchk(cudaMemcpy(deviceTransformStack, hostTransformStack, sizeof(TransformBVHTraversalStack) * stackCount, cudaMemcpyKind::cudaMemcpyHostToDevice));
		free(hostTransformStack);

		TransformBVHBuildData* hostTransformBuild = (TransformBVHBuildData*)malloc(sizeof(TransformBVHBuildData));
		TransformBVHBuildData* deviceTransformBuild = (TransformBVHBuildData*)MAllocDevice(sizeof(TransformBVHBuildData) * aagCount);

		for (int i = hin->GetStartIndex(); i < hin->GetCount(); i++)
		{
			const FrameMutableInput* hmin = hin->GetMutable(i);

			Log(
				"%.4f, %.4f, %.4f\n", 
				hmin->meshRendererBuffer[hmin->meshRendererBufferLen - 1].position.x,
				hmin->meshRendererBuffer[hmin->meshRendererBufferLen - 1].position.y,
				hmin->meshRendererBuffer[hmin->meshRendererBufferLen - 1].position.z
			);

			// transform nodes
			BuildTransformBVH(
				hmin->meshRendererBufferLen, hmin->meshRendererBuffer,
				hmin->skinnedMeshRendererBufferLen, hmin->skinnedMeshRendererBuffer,
				hmin->lightBufferLen, hmin->lightBuffer,
				hostTransformBuild
			);

			TransformNode* transformNodes = hostTransformBuild->transformNodes;
			hostTransformBuild->transformNodes = (TransformNode*)MAllocDevice(sizeof(TransformNode) * hostTransformBuild->capacity);
			gpuErrchk(cudaMemcpy(hostTransformBuild->transformNodes, transformNodes, sizeof(TransformNode) * hostTransformBuild->capacity, cudaMemcpyKind::cudaMemcpyHostToDevice));
			free(transformNodes);

			gpuErrchk(cudaMemcpy(deviceTransformBuild + i, hostTransformBuild, sizeof(TransformBVHBuildData), cudaMemcpyKind::cudaMemcpyHostToDevice));

			SetAccelAggregate << <1, 1 >> > (
				aag,
				din, i,
				deviceTransformBuild + i, deviceTransformStack,
				deviceStaticGroupBuild, deviceStaticMeshStack,
				deviceDynamicGroupBuild, deviceDynamicMeshStack,
				stackCount
				);
		}

		free(hostTransformBuild);
	}
	__host__ void AcceleratedAggregate::DestroyDeviceAggregate(int aagCount, AcceleratedAggregate* dagg)
	{
		AcceleratedAggregate hagg;
		gpuErrchk(cudaMemcpy(&hagg, dagg, sizeof(AcceleratedAggregate), cudaMemcpyKind::cudaMemcpyDeviceToHost));

		TransformBVHBuildData hostTransformBuild;
		gpuErrchk(cudaMemcpy(&hostTransformBuild, hagg.mTransformNodeGroup, sizeof(TransformBVHBuildData), cudaMemcpyKind::cudaMemcpyHostToDevice));

		gpuErrchk(cudaFree(hostTransformBuild.transformNodes));
		gpuErrchk(cudaFree(hagg.mTransformNodeGroup));

		TransformBVHTraversalStack firstHostStack;
		gpuErrchk(cudaMemcpy(&firstHostStack, hagg.mTransformNodeTraversalStack, sizeof(TransformBVHTraversalStack), cudaMemcpyKind::cudaMemcpyDeviceToHost));
		gpuErrchk(cudaFree(firstHostStack.nodeIndexList));
		gpuErrchk(cudaFree(firstHostStack.traversalStack));

		gpuErrchk(cudaFree(hagg.mTransformNodeTraversalStack));

		StaticMeshBuildData hostStaticGroupBuild;
		for (int i = 0; i < hagg.mStaticMeshCount; i++)
		{
			gpuErrchk(cudaMemcpy(&hostStaticGroupBuild, hagg.mStaticMeshNodeGroups + i, sizeof(StaticMeshBuildData), cudaMemcpyKind::cudaMemcpyHostToDevice));
			gpuErrchk(cudaFree(hostStaticGroupBuild.meshNodes));
		}
		gpuErrchk(cudaFree(hagg.mStaticMeshNodeGroups));

		StaticMeshKDTreeTraversalStack firstHostStaticMeshStack;
		gpuErrchk(cudaMemcpy(&firstHostStaticMeshStack, hagg.mStaticMeshTraversalStack, sizeof(StaticMeshKDTreeTraversalStack), cudaMemcpyKind::cudaMemcpyDeviceToHost));
		gpuErrchk(cudaFree(firstHostStaticMeshStack.traversalStack));
		gpuErrchk(cudaFree(hagg.mStaticMeshTraversalStack));

		DynamicMeshBuildData hostDynamicGroupBuild;
		for (int i = 0; i < hagg.mSkinnedMeshCount; i++)
		{
			gpuErrchk(cudaMemcpy(&hostDynamicGroupBuild, hagg.mDynamicMeshNodeGroups + i, sizeof(DynamicMeshBuildData), cudaMemcpyKind::cudaMemcpyHostToDevice));
			gpuErrchk(cudaFree(hostDynamicGroupBuild.meshNodes));
		}
		gpuErrchk(cudaFree(hagg.mDynamicMeshNodeGroups));

		DynamicMeshBVHTraversalStack firstHostDynamicMeshStack;
		gpuErrchk(cudaMemcpy(&firstHostDynamicMeshStack, hagg.mDynamicMeshTraversalStack, sizeof(DynamicMeshBVHTraversalStack), cudaMemcpyKind::cudaMemcpyDeviceToHost));
		gpuErrchk(cudaFree(firstHostDynamicMeshStack.traversalStack));
		gpuErrchk(cudaFree(hagg.mDynamicMeshTraversalStack));

		gpuErrchk(cudaFree(dagg));
	}

	int LeafMeshNodeCounter(StaticMeshNode* node)
	{
		if (node->splitAxis == SpaceSplitAxis::None)
		{
			if (node->primitiveIndex2 < INT_MAX)
				return 1;
			else
				return 2;
		}
		else
			return LeafMeshNodeCounter(node + 1) + LeafMeshNodeCounter(node + node->rightChildOffset);
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
		BuildTransformBVH(
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
		int	*deviceTraversalStack = (int*)MAllocDevice(sizeof(int) * transformStackSize * threadCount),
			*deviceNodeIndexList = (int*)MAllocDevice(sizeof(int) * transformStackSize * threadCount);
		for (int i = 0; i < threadCount; i++)
		{
			hostTransformStack[i].listCapacity = hostTransformStack[i].stackCapacity = transformStackSize;
			hostTransformStack[i].traversalStack = deviceTraversalStack + i * transformStackSize;
			hostTransformStack[i].nodeIndexList = deviceNodeIndexList + i * transformStackSize;
		}

		TransformBVHTraversalStack* deviceTransformStack = (TransformBVHTraversalStack*)MAllocDevice(sizeof(TransformBVHTraversalStack)* threadCount);
		gpuErrchk(cudaMemcpy(deviceTransformStack, hostTransformStack, sizeof(TransformBVHTraversalStack) * threadCount, cudaMemcpyKind::cudaMemcpyHostToDevice));
		free(hostTransformStack);

		// static mesh nodes 
		StaticMeshBuildData* hostStaticGroupBuild = (StaticMeshBuildData*)malloc(sizeof(StaticMeshBuildData) * himin->meshBufferLen);
		for (int i = 0; i < himin->meshBufferLen; i++)
		{
			BuildStaticMeshKDTree(
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
		StaticMeshKDTreeTraversalSegment* deviceStackSegments = (StaticMeshKDTreeTraversalSegment*)MAllocDevice(sizeof(StaticMeshKDTreeTraversalSegment) * staticMeshStackSize * threadCount);
		for (int i = 0; i < threadCount; i++)
		{
			hostStaticMeshStack[i].stackCapacity = staticMeshStackSize;
			hostStaticMeshStack[i].traversalStack = deviceStackSegments + i * staticMeshStackSize;
		}

		StaticMeshKDTreeTraversalStack* deviceStaticMeshStack = (StaticMeshKDTreeTraversalStack*)MAllocDevice(sizeof(StaticMeshKDTreeTraversalStack) * threadCount);
		gpuErrchk(cudaMemcpy(deviceStaticMeshStack, hostStaticMeshStack, sizeof(StaticMeshKDTreeTraversalStack) * threadCount, cudaMemcpyKind::cudaMemcpyHostToDevice));
		free(hostStaticMeshStack);

		// dynamic mesh nodes
		DynamicMeshBuildData* hostDynamicGroupBuild = (DynamicMeshBuildData*)malloc(sizeof(DynamicMeshBuildData) * himin->skinnedMeshBufferLen);
		for (int i = 0; i < himin->skinnedMeshBufferLen; i++)
		{
			BuildDynamicMeshBVH(
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
		DynamicMeshBVHTraversalSegment* deviceDynamicMeshStackSegments = (DynamicMeshBVHTraversalSegment*)MAllocDevice(sizeof(DynamicMeshBVHTraversalSegment) * dynamicMeshStackSize * threadCount);
		for (int i = 0; i < threadCount; i++)
		{
			hostDynamicMeshStack[i].stackCapacity = dynamicMeshStackSize;
			hostDynamicMeshStack[i].traversalStack = deviceDynamicMeshStackSegments + i * dynamicMeshStackSize;
		}

		DynamicMeshBVHTraversalStack* deviceDynamicMeshStack = (DynamicMeshBVHTraversalStack*)MAllocDevice(sizeof(DynamicMeshBVHTraversalStack) * threadCount);
		gpuErrchk(cudaMemcpy(deviceDynamicMeshStack, hostDynamicMeshStack, sizeof(DynamicMeshBVHTraversalStack) * threadCount, cudaMemcpyKind::cudaMemcpyHostToDevice));
		free(hostDynamicMeshStack);

		AcceleratedAggregate* deviceAllocated = (AcceleratedAggregate*)MAllocDevice(sizeof(AcceleratedAggregate));
		SetAccelAggregate << <1, 1 >> > (
			deviceAllocated,
			din, mutableIndex,
			deviceTransformBuild, deviceTransformStack,
			deviceStaticGroupBuild, deviceStaticMeshStack,
			deviceDynamicGroupBuild, deviceDynamicMeshStack,
			threadCount
			);

		return deviceAllocated;
	}

	__host__ void AcceleratedAggregate::DestroyDeviceAggregate(AcceleratedAggregate* dagg)
	{
		AcceleratedAggregate hagg;
		gpuErrchk(cudaMemcpy(&hagg, dagg, sizeof(AcceleratedAggregate), cudaMemcpyKind::cudaMemcpyDeviceToHost));

		TransformBVHBuildData hostTransformBuild;
		gpuErrchk(cudaMemcpy(&hostTransformBuild, hagg.mTransformNodeGroup, sizeof(TransformBVHBuildData), cudaMemcpyKind::cudaMemcpyHostToDevice));

		gpuErrchk(cudaFree(hostTransformBuild.transformNodes));
		gpuErrchk(cudaFree(hagg.mTransformNodeGroup));

		TransformBVHTraversalStack firstHostStack;
		gpuErrchk(cudaMemcpy(&firstHostStack, hagg.mTransformNodeTraversalStack, sizeof(TransformBVHTraversalStack), cudaMemcpyKind::cudaMemcpyDeviceToHost));
		gpuErrchk(cudaFree(firstHostStack.nodeIndexList));
		gpuErrchk(cudaFree(firstHostStack.traversalStack));

		gpuErrchk(cudaFree(hagg.mTransformNodeTraversalStack));

		StaticMeshBuildData hostStaticGroupBuild;
		for (int i = 0; i < hagg.mStaticMeshCount; i++)
		{
			gpuErrchk(cudaMemcpy(&hostStaticGroupBuild, hagg.mStaticMeshNodeGroups + i, sizeof(StaticMeshBuildData), cudaMemcpyKind::cudaMemcpyHostToDevice));
			gpuErrchk(cudaFree(hostStaticGroupBuild.meshNodes));
		}
		gpuErrchk(cudaFree(hagg.mStaticMeshNodeGroups));

		StaticMeshKDTreeTraversalStack firstHostStaticMeshStack;
		gpuErrchk(cudaMemcpy(&firstHostStaticMeshStack, hagg.mStaticMeshTraversalStack, sizeof(StaticMeshKDTreeTraversalStack), cudaMemcpyKind::cudaMemcpyDeviceToHost));
		gpuErrchk(cudaFree(firstHostStaticMeshStack.traversalStack));
		gpuErrchk(cudaFree(hagg.mStaticMeshTraversalStack));

		DynamicMeshBuildData hostDynamicGroupBuild;
		for (int i = 0; i < hagg.mSkinnedMeshCount; i++)
		{
			gpuErrchk(cudaMemcpy(&hostDynamicGroupBuild, hagg.mDynamicMeshNodeGroups + i, sizeof(DynamicMeshBuildData), cudaMemcpyKind::cudaMemcpyHostToDevice));
			gpuErrchk(cudaFree(hostDynamicGroupBuild.meshNodes));
		}
		gpuErrchk(cudaFree(hagg.mDynamicMeshNodeGroups));

		DynamicMeshBVHTraversalStack firstHostDynamicMeshStack;
		gpuErrchk(cudaMemcpy(&firstHostDynamicMeshStack, hagg.mDynamicMeshTraversalStack, sizeof(DynamicMeshBVHTraversalStack), cudaMemcpyKind::cudaMemcpyDeviceToHost));
		gpuErrchk(cudaFree(firstHostDynamicMeshStack.traversalStack));
		gpuErrchk(cudaFree(hagg.mDynamicMeshTraversalStack));

		gpuErrchk(cudaFree(dagg));
	}
}
