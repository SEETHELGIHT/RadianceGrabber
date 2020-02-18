#pragma once

#include "DataTypes.cuh"
#include "Define.h"

namespace RadGrabber
{
	struct BVHInternalNode
	{
		Bounds bounds;
		union
		{
			struct
			{
				int32 leafMeshIndex;
			};
			struct
			{
				int32 splitAxis			: 2;
				int32 leftIndex			: 31;
				int32 rightIndex		: 31;
			};
		};
	};
}
