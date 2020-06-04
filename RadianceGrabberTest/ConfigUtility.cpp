#include "ConfigUtility.h"
#include <cstdio>

namespace RadGrabber
{
	namespace Utility
	{
		int ConfigUtility::testMode = 0;
		bool ConfigUtility::imageConfigExist = 0;
		int ConfigUtility::maxSamplingCount = 0;
		int ConfigUtility::pathMaxDepth = 0;
		int ConfigUtility::imageWidth = 0;
		int ConfigUtility::imageHeight = 0;

		void ConfigUtility::RefreshValues()
		{
			FILE* imageConfigFile;
			fopen_s(&imageConfigFile, "./image.ini", "rt");
			{
				int buf;
				fscanf_s(imageConfigFile, "%d", &buf);
				ConfigUtility::testMode = buf;

				fscanf_s(imageConfigFile, "%d", &buf);
				ConfigUtility::imageConfigExist = buf != 0;

				if (ConfigUtility::imageConfigExist)
				{
					fscanf_s(imageConfigFile, "%d", &ConfigUtility::maxSamplingCount);
					fscanf_s(imageConfigFile, "%d", &ConfigUtility::pathMaxDepth);
					fscanf_s(imageConfigFile, "%d", &ConfigUtility::imageWidth);
					fscanf_s(imageConfigFile, "%d", &ConfigUtility::imageHeight);
				}
				else
				{
					ConfigUtility::maxSamplingCount = 0;
					ConfigUtility::pathMaxDepth = 0;
					ConfigUtility::imageWidth = 0;
					ConfigUtility::imageHeight = 0;
				}
			}
			fclose(imageConfigFile);
		}

		int ConfigUtility::GetMode()
		{
			return ConfigUtility::testMode;
		}
		bool ConfigUtility::IsImageConfigExist()
		{
			return ConfigUtility::imageConfigExist;
		}
		int ConfigUtility::GetMaxSamplingCount()
		{
			return ConfigUtility::maxSamplingCount;
		}
		int ConfigUtility::GetPathMaxDepth()
		{
			return ConfigUtility::pathMaxDepth;
		}
		int ConfigUtility::GetImageWidth()
		{
			return ConfigUtility::imageWidth;
		}
		int ConfigUtility::GetImageHeight()
		{
			return ConfigUtility::imageHeight;
		}
	}
}