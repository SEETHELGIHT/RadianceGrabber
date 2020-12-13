#define _CRT_SECURE_NO_WARNINGS
#include <cstdio>

#include "ConfigUtility.h"
#include "Marshal.cuh"
#include "DeviceConfig.h"

namespace RadGrabber
{
	namespace Utility
	{
		void TestProjectConfig::RefreshValues()
		{
			FILE* configFile;
			fopen_s(&configFile, "./project.ini", "rt");
			if (!configFile)
			{
				fopen_s(&configFile, "./RadianceGrabberTest/project.ini", "rt");
				assert(configFile);
			}
			
			{
				int buf;
				char pathBuffer[1025];
				
				fscanf_s(configFile, "%d\n", &buf);
				testMode = buf;

				fscanf_s(configFile, "%s", pathBuffer);
				int len = strlen(pathBuffer);
				frameRequestPath = (char*)malloc(sizeof(char) * (len + 1));
				strcpy_s(frameRequestPath, 1024, pathBuffer);

				fscanf_s(configFile, "%s", pathBuffer);
				len = strlen(pathBuffer);
				multiFrameRequestPath = (char*)malloc(sizeof(char) * (len + 1));
				strcpy_s(multiFrameRequestPath, 1024, pathBuffer);
			}
			fclose(configFile);
		}

		void FramePathTestConfig::RefreshValues()
		{
			FILE* imageConfigFile;
			fopen_s(&imageConfigFile, "./frame_path.ini", "rt");
			if (!imageConfigFile)
			{
				fopen_s(&imageConfigFile, "./RadianceGrabberTest/frame_path.ini", "rt");
				assert(imageConfigFile);
			}
			{
				int buf;
				fscanf_s(imageConfigFile, "%d", &buf);
				imageConfigIgnore = buf == 0;
				fscanf_s(imageConfigFile, "%d", &maxSamplingCount);
				fscanf_s(imageConfigFile, "%d", &pathMaxDepth);
				fscanf_s(imageConfigFile, "%d", &threadIterateCount);
				fscanf_s(imageConfigFile, "%d", &imageWidth);
				fscanf_s(imageConfigFile, "%d", &imageHeight);
				fscanf_s(imageConfigFile, "%d", &pixelContent);
				fscanf_s(imageConfigFile, "%d", &incrementalMode);
				fscanf_s(imageConfigFile, "%d", &threadCount.x);
				fscanf_s(imageConfigFile, "%d", &threadCount.y);
				fscanf_s(imageConfigFile, "%d", &threadCount.z);
				fscanf_s(imageConfigFile, "%d", &blockCount.x);
				fscanf_s(imageConfigFile, "%d", &blockCount.y);
				fscanf_s(imageConfigFile, "%d", &blockCount.z);
				fscanf_s(imageConfigFile, "%d", &startIndex);
				fscanf_s(imageConfigFile, "%d", &endCount);
			}
			fclose(imageConfigFile);
		}

		void FramePathTestConfig::Set(RequestOption& opt, dim3& threadCount, dim3& blockCount)
		{
			opt.maxSamplingCount = this->maxSamplingCount;
			opt.maxDepth = this->pathMaxDepth;
			opt.threadIterateCount = this->threadIterateCount;
			opt.resultImageResolution.x = this->imageWidth;
			opt.resultImageResolution.y = this->imageHeight;

			threadCount = this->threadCount;
			blockCount = this->blockCount;
		}
	}
}