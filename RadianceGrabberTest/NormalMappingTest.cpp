#include <cuda_runtime_api.h>
#include <cstdio>
#include <chrono>
#include <direct.h>
#include <csignal>
#include <thread>
#include <algorithm>

#define RADIANCEGRABBER_REMOVE_LOG
#include "Util.h"
#include "ColorTarget.h"
#include "Aggregate.h"
#include "integrator.h"
#include "Marshal.cuh"
#include "Sample.cuh"
#include "DataTypes.cuh"
#include "DeviceConfig.h"
#include "Unity/RenderAPI.h"
#include "Pipeline.h"

namespace RadGrabber
{
	namespace Test
	{
		FrameRequest* hreq;

		int NormalMappingTest()
		{
			char buf[129] = "SampleScene";

			char buf2[257];
			sprintf_s(buf2, "../UnityExampleProjects/URPForRT/%s.framerequest", buf);
			FILE* fp;
			fopen_s(&fp, buf2, "rb");
			hreq = new FrameRequest();
			LoadFrameRequest(fp, &hreq, malloc);
			fclose(fp);

			for (int i = 0; i < hreq->input.in.textureBufferLen; i++)
			{
				Texture2DChunk& c = hreq->input.in.textureBuffer[i];
				char buf3[257];
				int v;
				sprintf_s(buf3, 256, "./%s_%d.ppm", buf, i);
				FILE* fp = nullptr;
				fopen_s(&fp, buf3, "wt");

				fprintf(fp, "P3\n%d %d 255\n", c.size.x, c.size.y);
				ColorRGBA32* pp = reinterpret_cast<ColorRGBA32*>(hreq->input.in.textureBuffer[i].pixelPtr);
				for (int iw = 0; iw < c.size.x; iw++, fputs("\n", fp))
					for (int ih = 0; ih < c.size.y; ih++)
						fprintf(
							fp, 
							"%d %d %d ", 
							(v = (int)(pp[iw + ih * c.size.x].r * 255) < 256)? v: 255, 
							(v = (int)(pp[iw + ih * c.size.x].g * 255) < 256)? v: 255, 
							(v = (int)(pp[iw + ih * c.size.x].b * 255) < 256)? v: 255
						);

				fclose(fp);
			}

			return 0;
		}
	}
}
