#include <vector>
#include <list>
#include <algorithm>
#include <cfloat>

#include "AcceleratedAggregateInternal.h"
#include "Util.h"

namespace RadGrabber
{
	// TODO:: AAC Construction
	__host__ bool BuildTransformBVH(
		IN int meshRendererLen, IN MeshRendererChunk* meshRenderers,
		IN int skinnedMeshRendererLen, IN SkinnedMeshRendererChunk* skinnedMeshRenderers,
		IN int lightLen, IN LightChunk* lights,
		OUT TransformBVHBuildData* data
	)
	{
		//return LinearBVHBuild(meshRendererLen, meshRenderers, skinnedMeshRendererLen, skinnedMeshRenderers, lightLen, lights, data);

		int transformLen = meshRendererLen + skinnedMeshRendererLen + lightLen,
			nodeCount = 0;
		TransformBVHBuildSegment* segArray = (TransformBVHBuildSegment*)alloca(sizeof(TransformBVHBuildSegment) * transformLen * 2);
		TransformBVHBuildSegment seg, nseg;

		for (int i = 0, offset = 0; i < transformLen; i++)
		{
		BUILD_TRANSFORM_BVH_NEXT_LIGHT:
			if (i + offset >= transformLen) continue;

			int index;
			if (i < meshRendererLen)
			{
				index = i + offset;
				seg.kind = TransformItemKind::MeshRenderer;
				seg.bound = meshRenderers[index].boundingBox;
				seg.itemIndex = index;
			}
			else if (i < meshRendererLen + skinnedMeshRendererLen)
			{
				index = i + offset - meshRendererLen;
				seg.kind = TransformItemKind::SkinnedMeshRenderer;
				seg.bound = skinnedMeshRenderers[index].boundingBox;
				seg.itemIndex = index;
			}
			else if (i < meshRendererLen + skinnedMeshRendererLen + lightLen)
			{
				index = i + offset - meshRendererLen - skinnedMeshRendererLen;
				seg.kind = TransformItemKind::Light;
				if (!lights[index].GetBoundingBox(seg.bound))
				{
					offset++;
					goto BUILD_TRANSFORM_BVH_NEXT_LIGHT;
				}
				seg.itemIndex = index;
			}

			seg.parentIndex = -1;
			segArray[nodeCount++] = seg;
		}

		Vector3f delta;
		float sqrBestDistance, sqrDistance;
		int itemIndex = -1, processedNodeIndex = -1;

		while (true)
		{
			itemIndex = -1;
			sqrBestDistance = FLT_MAX;

			for (processedNodeIndex++; processedNodeIndex < nodeCount; processedNodeIndex++)
				if (!segArray[processedNodeIndex].processed)
					break;

			if (processedNodeIndex == nodeCount)
				break;

			for (int i = processedNodeIndex + 1; i < nodeCount; i++)
			{
				if (segArray[i].processed)
					continue;

				delta = segArray[processedNodeIndex].bound.center - segArray[i].bound.center;
				sqrDistance = delta.sqrMagnitude();

				if (sqrDistance < sqrBestDistance)
				{
					sqrBestDistance = sqrDistance;
					itemIndex = i;
				}
			}

			if (itemIndex < 0)
				break;

			segArray[processedNodeIndex].processed = 1;
			segArray[itemIndex].processed = 1;

			segArray[nodeCount].parentIndex = -1;
			segArray[nodeCount].processed = 0;
			segArray[nodeCount].isNotInternal = 0;
			segArray[nodeCount].leftChildOffset = nodeCount - itemIndex;
			segArray[nodeCount].rightChildOffset = nodeCount - processedNodeIndex;
			Union(segArray[processedNodeIndex].bound, segArray[itemIndex].bound, segArray[nodeCount].bound);

			nodeCount++;
		}

		int segmentIndex = nodeCount - 1, dataIndex = 0;
		data->count = nodeCount;
		data->capacity = nodeCount;
		data->transformNodes = (TransformNode*)malloc(sizeof(TransformNode) * nodeCount);

		int* stack = (int*)alloca(sizeof(int) * nodeCount), stackIndex = 1;
		stack[0] = nodeCount - 1;

		while (stackIndex > 0)
		{
			segmentIndex = stack[--stackIndex];

			data->transformNodes[dataIndex].bound = segArray[segmentIndex].bound;
			data->transformNodes[dataIndex].isNotInternal = segArray[segmentIndex].isNotInternal;

			if (segArray[segmentIndex].parentIndex >= 0)
				data->transformNodes[segArray[segmentIndex].parentIndex].rightChildOffset = dataIndex - segArray[segmentIndex].parentIndex;

			if (segArray[segmentIndex].isNotInternal)
			{
				data->transformNodes[dataIndex].itemIndex = segArray[segmentIndex].itemIndex;
			}
			else
			{
				data->transformNodes[dataIndex].rightChildOffset = 0;

				if (segArray[segmentIndex].rightChildOffset)
					segArray[segmentIndex - segArray[segmentIndex].rightChildOffset].parentIndex = dataIndex;

				stack[stackIndex++] = segmentIndex - segArray[segmentIndex].rightChildOffset;
				stack[stackIndex++] = segmentIndex - segArray[segmentIndex].leftChildOffset;
			}

			dataIndex++;
		}

		return true;
	}

	bool LinearTraversal(
		const Ray& rayInWS,
		const TransformNode* transformNodes, TransformBVHTraversalStack* stack
	)
	{
		int stackCount = 0;
		for (int i = 0; i < stack->stackCapacity; i++)
			if (transformNodes[i].bound.Intersect(rayInWS))
				stack->traversalStack[stackCount++] = i;
		return stackCount > 0;
	}

	struct StaticMeshKDTreeBuildSegment
	{
		bool ignoreAppendSegment;
		int parentNodeIndex;
		int startPrimIndex;
		int primCount;
		int depth;
		MeshChunk* c;
		Vector3f center;

		inline StaticMeshKDTreeBuildSegment()
			: parentNodeIndex(0), startPrimIndex(0), primCount(0), depth(0), c(nullptr), ignoreAppendSegment(false)
		{}
		inline StaticMeshKDTreeBuildSegment(int parentNodeIndex, int startPrimIndex, int primCount, int depth, MeshChunk* c)
			: parentNodeIndex(parentNodeIndex), startPrimIndex(startPrimIndex), primCount(primCount), depth(depth), c(c), ignoreAppendSegment(false)
		{}
		inline StaticMeshKDTreeBuildSegment(int parentNodeIndex, int startPrimIndex, int primCount, int depth, MeshChunk* c, bool ignoreAppendSegment)
			: parentNodeIndex(parentNodeIndex), startPrimIndex(startPrimIndex), primCount(primCount), depth(depth), c(c), ignoreAppendSegment(ignoreAppendSegment)
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

	__host__ bool BuildStaticMeshKDTree(IN MeshChunk* c, OUT StaticMeshBuildData* data)
	{
		StaticMeshKDTreeBuildSegment seg = StaticMeshKDTreeBuildSegment(-1, 0, c->indexCount / 3, 0, c);
		std::vector<StaticMeshKDTreeBuildSegment> list;
		list.reserve(seg.primCount);

		int* sortedPrimtives = (int*)alloca(sizeof(int) * seg.primCount);
		for (int i = 0; i < seg.primCount; i++)
			sortedPrimtives[i] = i;

		data->capacity = (c->indexCount / 3) * 2;
		data->meshNodes = (StaticMeshNode*)malloc(sizeof(StaticMeshNode) * data->capacity);

		int startPrimIndex, primCount, nodeCount = 0;

		if (c->submeshCount == 1)
		{
			list.push_back(seg);
		}
		else if (c->submeshCount > 1)
		{
			for (int i = 0; i < c->submeshCount; i++)
			{
				seg = StaticMeshKDTreeBuildSegment(-1, c->submeshArrayPtr[i].indexStart / 3, c->submeshArrayPtr[i].indexCount / 3, 0, c, false);			
				list.push_back(seg);
			}

			std::vector<int> checkedIndices;
			checkedIndices.reserve(data->capacity);
			checkedIndices.resize(c->submeshCount);

			for (int remainNodeCount = c->submeshCount, lastNodeIndex = 0; remainNodeCount >= 2; )
			{
				int targetIndex0 = -1, targetIndex1 = -1;
				for (int i = 0; i < checkedIndices.size(); i++)
				{
					if (!checkedIndices[i])
					{
						if (targetIndex0 < 0)
							targetIndex0 = i;
						else if (targetIndex1 < 0)
						{
							targetIndex1 = i;
							break;
						}
					}
				}

				if (targetIndex1 < 0)
				{
					lastNodeIndex = 0;
					continue;
				}

				list[targetIndex1].parentNodeIndex = targetIndex1 - lastNodeIndex + 1;
				checkedIndices[targetIndex0] = checkedIndices[targetIndex1] = 1;

				seg = StaticMeshKDTreeBuildSegment(
					-1,
					min(list[targetIndex0].startPrimIndex, list[targetIndex1].startPrimIndex),
					list[targetIndex0].primCount + list[targetIndex1].primCount,
					list[targetIndex0].depth - 1,
					c,
					true
				);
				list.insert(list.begin() + lastNodeIndex, seg);
				checkedIndices.insert(checkedIndices.begin() + lastNodeIndex, 0);

				lastNodeIndex = targetIndex1 + 1 + 1;
				remainNodeCount--;
			}

			for (int i = 0; i < list.size(); i++)
				if (list[i].parentNodeIndex >= 0)
					list[i].parentNodeIndex = i - list[i].parentNodeIndex;

			std::reverse(std::begin(list), std::end(list));
		}

		while (list.size())
		{
			seg = list[list.size() - 1];
			list.pop_back();

			if (seg.parentNodeIndex >= 0)
				data->meshNodes[seg.parentNodeIndex].rightChildOffset = nodeCount - seg.parentNodeIndex;

			if (seg.primCount > 2)
			{
				StaticMeshKDTreeBuildSegment::seg = seg;
				qsort(sortedPrimtives + seg.startPrimIndex, seg.primCount, sizeof(int), StaticMeshKDTreeBuildSegment::PrimSortCompareFunc);

				int		rightStartPrimIndex = seg.startPrimIndex + (seg.primCount + 1) / 2,
						axisIndex = ((seg.depth % 3) + 3) % 3;
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

				if (!seg.ignoreAppendSegment)
				{
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
			}
			else
			{
				data->meshNodes[nodeCount].reserved = 0;
				data->meshNodes[nodeCount].primitiveIndex1 = sortedPrimtives[seg.startPrimIndex] * 3;
				if (seg.primCount > 1)
					data->meshNodes[nodeCount].primitiveIndex2 = sortedPrimtives[seg.startPrimIndex + 1] * 3;
				else
					data->meshNodes[nodeCount].primitiveIndex2 = INT_MAX;
			}

			nodeCount++;
		}

		data->count = nodeCount;

		return true;
	}

	__host__ bool BuildDynamicMeshBVH(IN MeshChunk* c, OUT DynamicMeshBuildData* data)
	{


		return true;
	}

	__host__ bool UpdateDynamicMeshBVH(IN MeshChunk* c, OUT DynamicMeshNode** nodeArray)
	{
		return false;
	}


	__host__ __device__ bool TraversalStaticMeshKDTree(
		const Ray& rayInWS, const Ray& rayInMS,
		float minDistance, float maxDistance,
		const MeshChunk* mc, const MeshRendererChunk* mrc, const MaterialChunk* materials, const Texture2DChunk* textures,
		const StaticMeshNode* meshNodes, StaticMeshKDTreeTraversalStack* stack,
		INOUT float& distance, OUT SurfaceIntersection& isect
	)
	{
		ASSERT(stack->stackCapacity);
		ASSERT_IS_FALSE(IsNan(rayInMS.origin));
		ASSERT_IS_FALSE(IsNan(rayInMS.direction));

		int stackIndex = 1, isHit = 0;
		stack->traversalStack->bound = mc->aabbInMS;
		stack->traversalStack->itemIndex = 0;

		while (stackIndex > 0)
		{
			StaticMeshKDTreeTraversalSegment curSegment = stack->traversalStack[--stackIndex];
			const StaticMeshNode& meshNode = meshNodes[curSegment.itemIndex];

			if (meshNode.reserved)
			{
				const int axis = (int)meshNode.splitAxis - 1;

				Vector3f dir_frac = 1.0f / rayInMS.direction, smp, bgp;
				ASSERT_IS_FALSE(IsNan(dir_frac));

				// 1. calculate bound for bigger center BB
				bgp = curSegment.bound.center + curSegment.bound.extents;
				smp = curSegment.bound.center - curSegment.bound.extents;
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
					ASSERT(stackIndex + 1 < stack->stackCapacity);
					StaticMeshKDTreeTraversalSegment& nextSegment = stack->traversalStack[stackIndex];
					nextSegment.itemIndex = curSegment.itemIndex + meshNode.rightChildOffset;
					nextSegment.bound.center = (smp + bgp) / 2;
					nextSegment.bound.extents = Abs(bgp - nextSegment.bound.center);

					stackIndex++;
				}

				// 3. calculate bound for smaller center BB, 
				bgp = curSegment.bound.center + curSegment.bound.extents;
				smp = curSegment.bound.center - curSegment.bound.extents;
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
					ASSERT(stackIndex + 1 < stack->stackCapacity);
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

			ACCEL_AGG_CALC_RAY_TRIANGLE:
				int i0 = mc->indices[primitiveIndex + 0],
					i1 = mc->indices[primitiveIndex + 2],
					i2 = mc->indices[primitiveIndex + 1];

				Vector3f bc;
				float lastDistance = distance;
				if (
					IntersectRayAndTriangle(rayInMS, mc->positions[i0], mc->positions[i1], mc->positions[i2], lastDistance, bc) &&
					minDistance <= lastDistance && lastDistance < maxDistance
					)
				{
					isect.normal = (bc.x * mc->normals[i0] + bc.y * mc->normals[i1] + bc.z * mc->normals[i2]).normalized();

					if (Dot(isect.normal, -rayInMS.direction) <= 0)
						goto ACCEL_AGG_CALC_RAY_TRIANGLE_NEXT_PRIMTIVE;

					volatile int sbidx = mc->GetSubmeshIndexFromIndexOfIndex(primitiveIndex);

					if (mc->uvs && materials[mrc->materialArrayPtr[sbidx]].URPLit.IsAlphaClipping())
					{
						Vector2f uv = bc.x * mc->uvs[i0] + bc.y * mc->uvs[i1] + bc.z * mc->uvs[i2];
						ColorRGBA c = materials[mrc->materialArrayPtr[sbidx]].URPLit.SampleAlbedo(textures, uv);

						if (c.a == 0)
							goto ACCEL_AGG_CALC_RAY_TRIANGLE_NEXT_PRIMTIVE;

						isect.uv = uv;
					}

					distance = lastDistance;

					isect.normal = mrc->transformMatrix.TransformVector(isect.normal).normalized();

					isect.isHit = isHit = 1;
					isect.isGeometry = 1;
					isect.itemIndex = mrc->materialArrayPtr[sbidx];

					isect.position = bc.x * mc->positions[i0] + bc.y * mc->positions[i1] + bc.z * mc->positions[i2];
					isect.position = mrc->transformMatrix.TransformPoint(isect.position);

					isect.tangent = (bc.x * mc->tangents[i0] + bc.y * mc->tangents[i1] + bc.z * mc->tangents[i2]);
					isect.tangent = mrc->transformMatrix.TransformVector(isect.tangent).normalized();
				}

			ACCEL_AGG_CALC_RAY_TRIANGLE_NEXT_PRIMTIVE:

				if (primitiveIndex == meshNode.primitiveIndex1 && meshNode.primitiveIndex2 < INT_MAX)
				{
					primitiveIndex = meshNode.primitiveIndex2;
					goto ACCEL_AGG_CALC_RAY_TRIANGLE;
				}
			}
		}

		return isHit;
	}

	__host__ __device__ bool TraversalStaticMeshKDTree(
		const Ray& rayInWS, const Ray& rayInMS,
		float minDistance, float maxDistance,
		const MeshChunk* mc, const MeshRendererChunk* mrc, const MaterialChunk* materials, const Texture2DChunk* textures,
		const StaticMeshNode* meshNodes, StaticMeshKDTreeTraversalStack* stack,
		INOUT float& distance
	)
	{
		ASSERT(stack->stackCapacity);
		ASSERT_IS_FALSE(IsNan(rayInMS.origin));
		ASSERT_IS_FALSE(IsNan(rayInMS.direction));

		int isHit = 0;
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
					ASSERT(stackIndex + 1 < stack->stackCapacity);
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
					ASSERT(stackIndex + 1 < stack->stackCapacity);
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
				float lastDistance = distance;
				if (
					IntersectRayAndTriangle(rayInMS, mc->positions[i0], mc->positions[i1], mc->positions[i2], lastDistance, bc) &&
					minDistance <= lastDistance && lastDistance < maxDistance
					)
				{
					Vector3f normal = (bc.x * mc->normals[i0] + bc.y * mc->normals[i1] + bc.z * mc->normals[i2]).normalized();

					if (Dot(normal, -rayInMS.direction) <= 0)
						goto ACCEL_AGG_CALC_RAY_TRIANGLE_NEXT_PRIMTIVE_INTERSECTONLY;

					int sbidx = mc->GetSubmeshIndexFromIndexOfIndex(primitiveIndex);

					if (mc->uvs && materials[mrc->materialArrayPtr[sbidx]].URPLit.IsAlphaClipping())
					{
						Vector2f uv = bc.x * mc->uvs[i0] + bc.y * mc->uvs[i1] + bc.z * mc->uvs[i2];
						ColorRGBA c = materials[mrc->materialArrayPtr[sbidx]].URPLit.SampleAlbedo(textures, uv);

						if (c.a == 0)
							goto ACCEL_AGG_CALC_RAY_TRIANGLE_NEXT_PRIMTIVE_INTERSECTONLY;
					}

					distance = lastDistance;
					isHit = 1;
				}

			ACCEL_AGG_CALC_RAY_TRIANGLE_NEXT_PRIMTIVE_INTERSECTONLY:

				if (primitiveIndex == meshNode.primitiveIndex1 && meshNode.primitiveIndex2 < INT_MAX)
				{
					primitiveIndex = meshNode.primitiveIndex2;
					goto ACCEL_AGG_CALC_RAY_TRIANGLE_INTERSECTONLY;
				}
			}
		}

		return isHit;
	}

	__host__ __device__ bool TraversalStaticMeshKDTree(
		const Ray& rayInWS, const Ray& rayInMS,
		const MeshChunk* mc, const MeshRendererChunk* mrc, const MaterialChunk* materials, const Texture2DChunk* textures,
		const StaticMeshNode* meshNodes, StaticMeshKDTreeTraversalStack* stack
	)
	{
		ASSERT(stack->stackCapacity);
		ASSERT_IS_FALSE(IsNan(rayInMS.origin));
		ASSERT_IS_FALSE(IsNan(rayInMS.direction));

		int isHit = 0;
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
					ASSERT(stackIndex + 1 < stack->stackCapacity);
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
					ASSERT(stackIndex + 1 < stack->stackCapacity);
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

			TRAVERSAL_STATIC_KDTREE_INTERSECT_ONLY_TRIANGLE:

				int i0 = mc->indices[primitiveIndex + 0],
					i1 = mc->indices[primitiveIndex + 1],
					i2 = mc->indices[primitiveIndex + 2];

				Vector3f bc;
				const Vector3f &p0 = mc->positions[i0], &p1 = mc->positions[i1], &p2 = mc->positions[i2];
				if (IntersectRayAndTriangle(rayInMS, mc->positions[i0], mc->positions[i1], mc->positions[i2], bc))
				{
					Vector3f normal = (bc.x * mc->normals[i0] + bc.y * mc->normals[i1] + bc.z * mc->normals[i2]).normalized();

					if (Dot(normal, -rayInMS.direction) <= 0)
						goto TRAVERSAL_STATIC_KDTREE_INTERSECT_ONLY_NEXT_PRIMITIVE;

					int sbidx = mc->GetSubmeshIndexFromIndexOfIndex(primitiveIndex);

					if (mc->uvs && materials[mrc->materialArrayPtr[sbidx]].URPLit.IsAlphaClipping())
					{
						Vector2f uv = bc.x * mc->uvs[i0] + bc.y * mc->uvs[i1] + bc.z * mc->uvs[i2];
						ColorRGBA c = materials[mrc->materialArrayPtr[sbidx]].URPLit.SampleAlbedo(textures, uv);

						if (c.a == 0)
							goto TRAVERSAL_STATIC_KDTREE_INTERSECT_ONLY_NEXT_PRIMITIVE;
					}

					isHit = 1;
				}

			TRAVERSAL_STATIC_KDTREE_INTERSECT_ONLY_NEXT_PRIMITIVE:

				if (primitiveIndex == meshNode.primitiveIndex1 && meshNode.primitiveIndex2 < INT_MAX)
				{
					primitiveIndex = meshNode.primitiveIndex2;
					goto TRAVERSAL_STATIC_KDTREE_INTERSECT_ONLY_TRIANGLE;
				}
			}
		}

		return isHit;
	}

	__host__ __device__ bool TraversalStaticMeshKDTree(
		const Ray& rayInWS, const Ray& rayInMS,
		float minDistance, float maxDistance,
		const MeshChunk* mc, const MeshRendererChunk* mrc, const MaterialChunk* materials, const Texture2DChunk* textures,
		int& iterateCount, struct AATraversalSegment& seg,
		const StaticMeshNode* meshNodes, StaticMeshKDTreeTraversalStack* stack,
		OUT SurfaceIntersection& isect
	)
	{
		ASSERT(stack->stackCapacity);
		ASSERT_IS_FALSE(IsNan(rayInMS.origin));
		ASSERT_IS_FALSE(IsNan(rayInMS.direction));

		if (!seg.initilaized)
		{
			seg.initilaized = 1;

			seg.lowerStackCount = 1;
			stack->traversalStack[0].bound = mc->aabbInMS;
			stack->traversalStack[0].itemIndex = 0;
		}

		while (seg.lowerStackCount > 0 && iterateCount > 0)
		{
			iterateCount -= ITERATIVE_COST_COUNT;
			StaticMeshKDTreeTraversalSegment curSegment = stack->traversalStack[--seg.lowerStackCount];
			const StaticMeshNode& meshNode = meshNodes[curSegment.itemIndex];

			if (meshNode.reserved)
			{
				const int axis = (int)meshNode.splitAxis - 1;

				Vector3f dir_frac = 1.0f / rayInMS.direction, smp, bgp;
				ASSERT_IS_FALSE(IsNan(dir_frac));

				// 1. calculate bound for bigger center BB
				bgp = curSegment.bound.center + curSegment.bound.extents;
				smp = curSegment.bound.center - curSegment.bound.extents;
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
					ASSERT(seg.lowerStackCount + 1 < stack->stackCapacity);
					StaticMeshKDTreeTraversalSegment& nextSegment = stack->traversalStack[seg.lowerStackCount];
					nextSegment.itemIndex = curSegment.itemIndex + meshNode.rightChildOffset;
					nextSegment.bound.center = (smp + bgp) / 2;
					nextSegment.bound.extents = Abs(bgp - nextSegment.bound.center);

					seg.lowerStackCount++;
				}

				// 3. calculate bound for smaller center BB, 
				bgp = curSegment.bound.center + curSegment.bound.extents;
				smp = curSegment.bound.center - curSegment.bound.extents;
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
					ASSERT(seg.lowerStackCount + 1 < stack->stackCapacity);
					StaticMeshKDTreeTraversalSegment& nextSegment = stack->traversalStack[seg.lowerStackCount];
					nextSegment.itemIndex = curSegment.itemIndex + 1;
					nextSegment.bound.center = (smp + bgp) / 2;
					nextSegment.bound.extents = Abs(bgp - nextSegment.bound.center);

					seg.lowerStackCount++;
				}
			}
			else
			{
				int primitiveIndex = meshNode.primitiveIndex1;

			ITERATIVE_ACCEL_AGG_CALC_RAY_TRIANGLE:

				int i0 = mc->indices[primitiveIndex + 0],
					i1 = mc->indices[primitiveIndex + 1],
					i2 = mc->indices[primitiveIndex + 2];

				Vector3f bc;
				float lastDistance = seg.lastDistance;
				if (
					IntersectRayAndTriangle(rayInMS, mc->positions[i0], mc->positions[i1], mc->positions[i2], lastDistance, bc) &&
					minDistance <= lastDistance && lastDistance < maxDistance
					)
				{
					isect.normal = (bc.x * mc->normals[i0] + bc.y * mc->normals[i1] + bc.z * mc->normals[i2]).normalized();

					if (Dot(isect.normal, -rayInMS.direction) <= 0)
						goto ITERATIVE_ACCEL_AGG_CALC_RAY_TRIANGLE_NEXT_PRIMTIVE;

					volatile int sbidx = mc->GetSubmeshIndexFromIndexOfIndex(primitiveIndex);

					if (mc->uvs && materials[mrc->materialArrayPtr[sbidx]].URPLit.IsAlphaClipping())
					{
						Vector2f uv = bc.x * mc->uvs[i0] + bc.y * mc->uvs[i1] + bc.z * mc->uvs[i2];
						ColorRGBA c = materials[mrc->materialArrayPtr[sbidx]].URPLit.SampleAlbedo(textures, uv);

						if (c.a == 0)
							goto ITERATIVE_ACCEL_AGG_CALC_RAY_TRIANGLE_NEXT_PRIMTIVE;

						isect.uv = uv;
					}

					isect.normal = mrc->transformMatrix.TransformVector(isect.normal).normalized();

					isect.isHit = 1;
					isect.isGeometry = 1;
					isect.itemIndex = mrc->materialArrayPtr[sbidx];

					isect.position = bc.x * mc->positions[i0] + bc.y * mc->positions[i1] + bc.z * mc->positions[i2];
					isect.position = mrc->transformMatrix.TransformPoint(isect.position);

					isect.tangent = (bc.x * mc->tangents[i0] + bc.y * mc->tangents[i1] + bc.z * mc->tangents[i2]);
					isect.tangent = mrc->transformMatrix.TransformVector(isect.tangent).normalized();

					seg.lastDistance = lastDistance;
					seg.findPrimitive = 1;
				}

			ITERATIVE_ACCEL_AGG_CALC_RAY_TRIANGLE_NEXT_PRIMTIVE:

				if (primitiveIndex == meshNode.primitiveIndex1 && meshNode.primitiveIndex2 < INT_MAX)
				{
					primitiveIndex = meshNode.primitiveIndex2;
					goto ITERATIVE_ACCEL_AGG_CALC_RAY_TRIANGLE;
				}
			}

		}

		return seg.lowerStackCount <= 0;
	}

	__host__ __device__ bool TraversalStaticMeshKDTree(
		const Ray& rayInWS, const Ray& rayInMS,
		float minDistance, float maxDistance,
		const MeshChunk* mc, const MeshRendererChunk* mrc, const MaterialChunk* materials, const Texture2DChunk* textures,
		int& iterateCount, struct AATraversalSegment& seg,
		const StaticMeshNode* meshNodes, StaticMeshKDTreeTraversalStack* stack
	)
	{
		ASSERT(stack->stackCapacity);
		ASSERT_IS_FALSE(IsNan(rayInMS.origin));
		ASSERT_IS_FALSE(IsNan(rayInMS.direction));

		if (!seg.initilaized)
		{
			seg.initilaized = 1;
			seg.lowerStackCount = 1;

			stack->traversalStack[0].bound = mc->aabbInMS;
			stack->traversalStack[0].itemIndex = 0;
		}

		while (seg.lowerStackCount > 0 && iterateCount > 0)
		{
			--iterateCount;
			StaticMeshKDTreeTraversalSegment curSegment = stack->traversalStack[--seg.lowerStackCount];
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
					ASSERT(seg.lowerStackCount + 1 < stack->stackCapacity);
					StaticMeshKDTreeTraversalSegment& nextSegment = stack->traversalStack[seg.lowerStackCount];
					nextSegment.itemIndex = curSegment.itemIndex + meshNode.rightChildOffset;
					nextSegment.bound.center = (smp + bgp) / 2;
					nextSegment.bound.extents = Abs(bgp - nextSegment.bound.center);

					seg.lowerStackCount++;
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
					ASSERT(seg.lowerStackCount + 1 < stack->stackCapacity);
					StaticMeshKDTreeTraversalSegment& nextSegment = stack->traversalStack[seg.lowerStackCount];
					nextSegment.itemIndex = curSegment.itemIndex + 1;
					nextSegment.bound.center = (smp + bgp) / 2;
					nextSegment.bound.extents = Abs(bgp - nextSegment.bound.center);

					seg.lowerStackCount++;
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

				float lastDistance = seg.lastDistance;
				if (
					IntersectRayAndTriangle(rayInMS, mc->positions[i0], mc->positions[i1], mc->positions[i2], lastDistance, bc) &&
					minDistance <= lastDistance && lastDistance < maxDistance
					)
				{
					Vector3f normal = (bc.x * mc->normals[i0] + bc.y * mc->normals[i1] + bc.z * mc->normals[i2]).normalized();

					if (Dot(normal, -rayInMS.direction) <= 0)
						goto ACCEL_AGG_CALC_RAY_TRIANGLE_NEXT_PRIMTIVE;

					int sbidx = mc->GetSubmeshIndexFromIndexOfIndex(primitiveIndex);

					if (mc->uvs && materials[mrc->materialArrayPtr[sbidx]].URPLit.IsAlphaClipping())
					{
						Vector2f uv = bc.x * mc->uvs[i0] + bc.y * mc->uvs[i1] + bc.z * mc->uvs[i2];
						ColorRGBA c = materials[mrc->materialArrayPtr[sbidx]].URPLit.SampleAlbedo(textures, uv);

						if (c.a == 0)
							goto ACCEL_AGG_CALC_RAY_TRIANGLE_NEXT_PRIMTIVE;
					}

					seg.findPrimitive = 1;
					seg.lastDistance = lastDistance;
				}

			ACCEL_AGG_CALC_RAY_TRIANGLE_NEXT_PRIMTIVE:

				if (primitiveIndex == meshNode.primitiveIndex1 && meshNode.primitiveIndex2 < INT_MAX)
				{
					primitiveIndex = meshNode.primitiveIndex2;
					goto ACCEL_AGG_CALC_RAY_TRIANGLE_INTERSECTONLY;
				}
			}
		}

		return seg.lowerStackCount <= 0;
	}

}
