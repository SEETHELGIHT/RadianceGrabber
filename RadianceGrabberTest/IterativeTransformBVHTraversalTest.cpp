#include <random>

#include "Marshal.cuh"

#include "AcceleratedAggregate.h"
#include "AcceleratedAggregateInternal.h"
#include "ConfigUtility.h"
#include "Pipeline.h"

namespace RadGrabber
{
	namespace Test
	{
		int IterativeTransformBVHBuildTest()
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

			std::default_random_engine generator;
			std::uniform_real_distribution<float> distribution(0, 1);

			TransformBVHBuildData data;
			TransformBVHTraversalStack stack;
			stack.listCapacity = stack.stackCapacity = min->meshRendererBufferLen + min->skinnedMeshRendererBufferLen + min->lightBufferLen;
			stack.listCount = 0;
			stack.nodeIndexList = (int*)malloc(sizeof(int)*stack.listCapacity);
			stack.traversalStack = (int*)malloc(sizeof(int)*stack.stackCapacity);

			RadGrabber::BuildTransformBVH(
				min->meshRendererBufferLen, min->meshRendererBuffer,
				min->skinnedMeshRendererBufferLen, min->skinnedMeshRendererBuffer,
				min->lightBufferLen, min->lightBuffer,
				&data
			);

			int currentiterateCount = 30, maxIterateCount = 30;
			AATraversalSegment seg;
			InitTraversalSegment(seg);

			for (int i = 0; i < iterateCount; mode == 1 ? i++ : i = 0)
			{
				currentiterateCount = maxIterateCount;

				Vector2f v = Vector2f(distribution(generator), distribution(generator));
				if (!mode) std::cout << "(" << v.x << "," << v.y << ")" << std::endl;

				Ray r;
				GetRay(req->input.in.cameraBuffer[req->input.in.selectedCameraIndex].projectionInverseMatrix, req->input.in.cameraBuffer[req->input.in.selectedCameraIndex].cameraInverseMatrix, v, r);
				if (!mode) printf("ray(origin=(%.2f, %.2f, %.2f), dir=(%.3f, %.3f, %.3f))\n", r.origin.x, r.origin.y, r.origin.z, r.direction.x, r.direction.y, r.direction.z);

				if (result = RadGrabber::TraversalTransformBVH(r, data.transformNodes, &stack))
				{
					if (!mode) printf("normal traversal :: hit!\n");

					for (int i = 0; i < stack.listCount; i++)
					{
						if (data.transformNodes[stack.nodeIndexList[i]].isNotInternal)
						{
							if (!mode) printf("normal traversal :: item kind::%d ", data.transformNodes[stack.nodeIndexList[i]].kind);
							if (!mode) printf(", item Index::%d\n", data.transformNodes[stack.nodeIndexList[i]].itemIndex);
						}
						else
						{
							if (!mode) printf("normal traversal :: fuck \n");
						}
					}
				}
				else
				{
					if (!mode) printf("normal traversal :: missed!@!@!@#@\n");
				}

				while (!RadGrabber::TraversalTransformBVH(r, data.transformNodes, &stack, currentiterateCount, seg));

				if (stack.listCount > 0)
				{
					if (!mode) printf("iterative traversal :: hit!\n");

					for (int i = 0; i < stack.listCount; i++)
					{
						if (data.transformNodes[stack.nodeIndexList[i]].isNotInternal)
						{
							if (!mode) printf("iterative traversal :: item kind::%d ", data.transformNodes[stack.nodeIndexList[i]].kind);
							if (!mode) printf(", item Index::%d\n", data.transformNodes[stack.nodeIndexList[i]].itemIndex);
						}
						else
						{
							if (!mode) printf("normal traversal :: fuck \n");
						}
					}
				}
				else
				{
					if (!mode) printf("iterative traversal :: missed!@!@!@#@\n");
				}

				if (!mode) getchar();

				if (result == stack.listCount > 0)
					equalCount++;
				else
					diffrCount++;
			}

			printf("equal :: %d, different :: %d, %.6lf\n", equalCount, diffrCount, (double)diffrCount / (equalCount + diffrCount));

			return 0;
		}
	}
}
