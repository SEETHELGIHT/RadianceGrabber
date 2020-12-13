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
		int IterativeStaticMeshTraversalTest()
		{
			int equalCount = 0, diffrCount = 0;
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

			StaticMeshBuildData data;
			StaticMeshKDTreeTraversalStack stack;
			stack.stackCapacity = 100;
			stack.traversalStack = (StaticMeshKDTreeTraversalSegment*)malloc(sizeof(StaticMeshKDTreeTraversalSegment)*stack.stackCapacity);

			int maxIterateCount = 1000, currentiterateCount = maxIterateCount;
			AATraversalSegment seg;
			InitTraversalSegment(seg);

			for (int i = 0; i < iterateCount; mode == 1? i++: i=0)
			{
				MeshRendererChunk* mrc = req->input.in.meshRendererBuffer;
				MeshChunk* mc = req->input.in.meshBuffer + mrc->meshRefIndex;
				RadGrabber::BuildStaticMeshKDTree(mc, &data);

				float distance = FLT_MAX;

				Vector2f v = Vector2f(distribution(generator), distribution(generator));

				if (!mode)
					std::cout << "(" << v.x << "," << v.y << ")" << std::endl;

				SurfaceIntersection isect;
				Ray r, r2;
				GetRay(req->input.in.cameraBuffer[req->input.in.selectedCameraIndex].projectionInverseMatrix, req->input.in.cameraBuffer[req->input.in.selectedCameraIndex].cameraInverseMatrix, v, r);

				r2.origin = mrc->transformInverseMatrix.TransformPoint(r.origin);
				r2.direction = mrc->transformInverseMatrix.TransformVector(r.direction);

				//r = Ray(Vector3f(1.641098, 2.436149, 1.318705), Vector3f(-0.028390, -0.021538, -0.035073));
				//r = Ray(Vector3f(1.427106, 9.233054, 11.542421), Vector3f(-0.117944, -0.107950, -0.120150));
				//r = Ray(Vector3f(4.145553, 4.506527, 2.986211), Vector3f(-0.058972, -0.053975, -0.060075));
				
				if (!mode)
					printf("ray(origin=(%.6f, %.6f, %.6f), dir=(%.6f, %.6f, %.6f))\n", r.origin.x, r.origin.y, r.origin.z, r.direction.x, r.direction.y, r.direction.z);

				int result;
				if (result = RadGrabber::TraversalStaticMeshKDTree(r, r2, 0, FLT_MAX, mc, mrc, req->input.in.materialBuffer, req->input.in.textureBuffer, data.meshNodes, &stack, distance, isect))
				{
					if (!mode)
						printf("normal :: hit!\n");
				}
				else
				{
					if (!mode)
						printf("normal :: missed!@!@!@#@\n");
				}

				distance = FLT_MAX;
				seg.lastDistance = FLT_MAX;
				isect.isHit = 0;
				seg.findPrimitive = 0;

				while (!RadGrabber::TraversalStaticMeshKDTree(r, r2, 0, FLT_MAX, mc, mrc, req->input.in.materialBuffer, req->input.in.textureBuffer, currentiterateCount, seg, data.meshNodes, &stack, isect))
				{
					//printf("itertae count update :: %d\n", maxIterateCount);
					currentiterateCount = maxIterateCount;
				}

				if (isect.isHit)
				{
					if (!mode)
						printf("iterative :: hit!\n");
				}
				else
				{
					if (!mode)
						printf("iterative :: missed!@!@!@#@\n");
				}

				if (!!result == isect.isHit)
					equalCount++;
				else
					diffrCount++;

				InitTraversalSegment(seg);

				if (!mode)
					getchar();
			}

			printf("equal :: %d, different :: %d, %.6lf\n", equalCount, diffrCount, (double)diffrCount / (equalCount + diffrCount));

			return 0;
		}
	}
}
