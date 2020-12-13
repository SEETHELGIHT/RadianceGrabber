#include <vector_types.h>

#pragma once

namespace RadGrabber
{
	struct RequestOption;

	namespace Utility
	{
		struct TestProjectConfig
		{
			void RefreshValues();

			int testMode;
			char* frameRequestPath;
			char* multiFrameRequestPath;
		};

		struct FramePathTestConfig 
		{
			void Set(RequestOption& opt, dim3& threadCount, dim3& blockCount);
			void RefreshValues();

			bool imageConfigIgnore;
			int maxSamplingCount;
			int pathMaxDepth;
			int threadIterateCount;
			int imageWidth;
			int imageHeight;
			int pixelContent;
			int incrementalMode;
			dim3 threadCount;
			dim3 blockCount;
			int startIndex;
			int endCount;
		};
	}
}
