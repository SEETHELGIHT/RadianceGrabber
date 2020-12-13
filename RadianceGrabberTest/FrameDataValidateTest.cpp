#include <cuda_runtime_api.h>
#define _CRT_SECURE_NO_WARNINGS
#include <cstdio>
#include <chrono>

#include "Util.h"
#include "ColorTarget.h"
#include "LinearAggregate.h"
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
		int FrameDataValidateTest()
		{
			FILE* fp = nullptr;
			fopen_s(&fp, "./SimpleScene.framerequest", "rb");
			FrameRequest* req = new FrameRequest();
			LoadFrameRequest(fp, &req, malloc);
			fclose(fp);

			LinearAggregate* aggr = LinearAggregate::GetAggregateHost(req->input.GetMutable(0), req->input.GetImmutable());
			Ray ray(Vector3f(9.20683670f, 5.45162678f, 3.55255318f), Vector3f(-0.624369442, 0.614192069, 0.482629150));
			SurfaceIntersection isect;
			if (aggr->Intersect(ray, isect, 0))
			{
				Log("Intersected!");
			}

			return 0;
		}
	}
}
