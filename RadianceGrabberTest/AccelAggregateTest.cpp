#include <random>

#include "Marshal.cuh"
#include "AcceleratedAggregate.h"
#include "ConfigUtility.h"
#include "Pipeline.h"

namespace RadGrabber
{
	namespace Test
	{
		int AccelAggregateIntersectTest()
		{
			int equalCount = 0, diffrCount = 0, result;
			int mode, iterateCount = 1;
			printf("Interactive :: 0, Iterate And Count :: 1, 0 or 1 ? ");
			scanf("%d", &mode);
			if (mode)
			{
				printf("Iterate Count ? ");
				scanf("%d", &iterateCount);
			}

			Utility::TestProjectConfig config;
			config.RefreshValues();

			FILE* fp = nullptr;
			fopen_s(&fp, config.frameRequestPath, "rb");
			FrameRequest* req = new FrameRequest();
			LoadFrameRequest(fp, &req, malloc);
			fclose(fp);

			auto min = req->input.GetMutable(0);
			auto imin = req->input.GetImmutable();

			AcceleratedAggregate* agg = AcceleratedAggregate::GetAggregateHost(req->input.GetMutable(0), req->input.GetImmutable(), 1);

			std::default_random_engine generator;
			std::uniform_real_distribution<float> distribution(0, 1);

			int maxIterateCount = 30, currentiterateCount = maxIterateCount;
			AATraversalSegment seg;
			InitTraversalSegment(seg);
			int width = 0, height = 0;
			float step = 0.125f;

			for (int i = 0, j = 0; i < iterateCount; mode == 1 ? i++ : i = 0, j++)
			{
				currentiterateCount = maxIterateCount;

				Vector2f v = Vector2f(distribution(generator), distribution(generator));
				v = Vector2f((float)width * step, (float)height * step);
				if (!mode) std::cout << "(" << v.x << "," << v.y << ")" << std::endl;

				Ray r;
				SurfaceIntersection isect, isect2;
				memset(&isect, 0, sizeof(SurfaceIntersection));
				GetRay(req->input.in.cameraBuffer[req->input.in.selectedCameraIndex].projectionInverseMatrix, req->input.in.cameraBuffer[req->input.in.selectedCameraIndex].cameraInverseMatrix, v, r);

				if (!mode) printf("ray(origin=(%.2f, %.2f, %.2f), dir=(%.3f, %.3f, %.3f))\n", r.origin.x, r.origin.y, r.origin.z, r.direction.x, r.direction.y, r.direction.z);

				if (result = agg->Intersect(r, isect, 0))
				{
					if (!mode) printf("normal :: hit!, isect.isGeometry :: %d, isect.itemIndex :: %d\n", isect.isGeometry, isect.itemIndex);
				}
				else
				{
					if (!mode) printf("normal :: missed!@!@!@#@\n");
				}

				Ray rayInMS;
				seg.isLowerTransform = 0;
				seg.initilaized = 0;
				seg.lastDistance = FLT_MAX;
				while (!agg->IterativeIntersect(r, rayInMS, isect2, 0, currentiterateCount, seg))
				{
					currentiterateCount = maxIterateCount;
					if (!mode) printf("iterative :: reset iterate count\n");
				}

				if (isect2.isHit)
				{
					if (!mode) printf("iterative :: hit!, isect.isGeometry :: %d, isect.itemIndex :: %d\n", isect2.isGeometry, isect2.itemIndex);
				}
				else
				{
					if (!mode) printf("iterative :: missed!@!@!@#@\n");
				}

				if (!mode)
				{
					char c = getchar();

					switch (c)
					{
					case 'W':
						height++;
					case 'w':
						height++;
						break;
					case 'S':
						height--;
					case 's':
						height--;
						break;
					case 'D':
						width++;
					case 'd':
						width++;
						break;
					case 'A':
						width--;
					case 'a':
						width--;
						break;
					case 'Q':
						width *= 4;
						height *= 4;
						step *= 0.25f;
						break;
					case 'q':
						width *= 2;
						height *= 2;
						step *= 0.5f;
						break;
					case 'E':
						width /= 4;
						height /= 4;
						step *= 4.f;
						break;
					case 'e':
						width /= 2;
						height /= 2;
						step *= 2.f;
						break;
					}
				}
					

				if (isect.isHit == isect2.isHit && isect.isGeometry == isect2.isGeometry && isect.itemIndex == isect2.itemIndex)
					equalCount++;
				else
					diffrCount++;
			}

			printf("equal :: %d, different :: %d, %.6lf\n", equalCount, diffrCount, (double)diffrCount / (equalCount + diffrCount));

			return 0;
		}
	}
}