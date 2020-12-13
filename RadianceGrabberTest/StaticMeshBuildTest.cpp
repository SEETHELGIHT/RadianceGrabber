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
		int StaticMeshBuildTest()
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

			StaticMeshBuildData data;
			StaticMeshKDTreeTraversalStack stack;
			stack.stackCapacity = 100;
			stack.traversalStack = (StaticMeshKDTreeTraversalSegment*)malloc(sizeof(StaticMeshKDTreeTraversalSegment)*stack.stackCapacity);

			
			MeshRendererChunk* mrc = req->input.in.meshRendererBuffer;
			MeshChunk* mc = req->input.in.meshBuffer + mrc->meshRefIndex;
			RadGrabber::BuildStaticMeshKDTree(mc, &data);

			while (true)
			{
				Vector2f v = Vector2f(distribution(generator), distribution(generator));
				std::cout << "(" << v.x << "," << v.y << ")" << std::endl;

				Ray r, r2;
				GetRay(req->input.in.cameraBuffer[req->input.in.selectedCameraIndex].projectionInverseMatrix, req->input.in.cameraBuffer[req->input.in.selectedCameraIndex].cameraInverseMatrix, v, r);
				printf("ray(origin=(%.2f, %.2f, %.2f), dir=(%.3f, %.3f, %.3f))\n", r.origin.x, r.origin.y, r.origin.z, r.direction.x, r.direction.y, r.direction.z);

				r2.origin = mrc->transformInverseMatrix.TransformPoint(r.origin);
				r2.direction = mrc->transformInverseMatrix.TransformPoint(r.direction);

				if (RadGrabber::TraversalStaticMeshKDTree(r, r2, mc, mrc, req->input.in.materialBuffer, req->input.in.textureBuffer, data.meshNodes, &stack))
				{
					printf("hit!\n");
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
