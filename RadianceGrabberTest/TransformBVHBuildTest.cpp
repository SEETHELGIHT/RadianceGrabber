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
		int TransformBVHBuildTest()
		{
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

			while (true)
			{
				Vector2f v = Vector2f(distribution(generator), distribution(generator));
				std::cout << "(" << v.x << "," << v.y << ")" << std::endl;

				Ray r;
				GetRay(req->input.in.cameraBuffer[req->input.in.selectedCameraIndex].projectionInverseMatrix, req->input.in.cameraBuffer[req->input.in.selectedCameraIndex].cameraInverseMatrix, v, r);
				printf("ray(origin=(%.2f, %.2f, %.2f), dir=(%.3f, %.3f, %.3f))\n", r.origin.x, r.origin.y, r.origin.z, r.direction.x, r.direction.y, r.direction.z);

				if (RadGrabber::TraversalTransformBVH(r, data.transformNodes, &stack))
				{
					printf("hit!\n");

					for (int i = 0; i < stack.listCount; i++)
					{
						if (data.transformNodes[stack.nodeIndexList[i]].isNotInternal)
						{
							printf("item kind::%d ", data.transformNodes[stack.nodeIndexList[i]].kind);
							printf(", item Index::%d\n", data.transformNodes[stack.nodeIndexList[i]].itemIndex);
						}
						else
						{
							printf("fuck \n");
						}
					}
				}
				else
				{
					printf("missed!@!@!@#@\n");
				}

				getchar();
			}

			return 0;
		}
	}
}
