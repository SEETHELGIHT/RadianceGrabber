#include "Marshal.h"

#pragma once

namespace RadGrabber
{
	enum class GeometryKind
	{
		StaticMesh		= 0,
		SkinnedMesh		= 1,
		AreaLight		= 2,
	};

	struct BVHInternalNode
	{
		Bounds bounds;
		union
		{
			struct
			{
				GeometryKind geomKind	: 2;
				int32 leafMeshIndex		: 30;
			};
			struct
			{
				int32 splitAxis			: 2;
				int32 leftIndex			: 31;
				int32 rightIndex		: 31;
			};
		};
	};

	class BVH
	{
	public:
		void BuildBVH(MeshRendererChunk* mrs, int mrCount, SkinnedMeshRendererChunk* smrs, int smrCount, LightChunk* lights, int lightCount);

	private:
		void RecursiveBuild();

	private:
		BVHInternalNode* mRoot;
	};
}
