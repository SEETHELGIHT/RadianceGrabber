#pragma once

namespace RadGrabber
{
	namespace Utility
	{
		class ConfigUtility
		{
		public:
			static void RefreshValues();

			static int GetMode();

			static bool IsImageConfigExist();
			static int GetMaxSamplingCount();
			static int GetPathMaxDepth();
			static int GetImageWidth();
			static int GetImageHeight();

		private:
			static int testMode;

			static bool imageConfigExist;
			static int maxSamplingCount;
			static int pathMaxDepth;
			static int imageWidth;
			static int imageHeight;

		};
	}
}
