#include "DataTypes.cuh"
#include "Util.h"
#include "Sample.cuh"

namespace RadGrabber
{
	namespace Test
	{
		__global__ void RayIntersectTestKernel(int *p)
		{
			Ray r = Ray(Vector3f(-10, 0, 0), Vector3f(1, 0, 0));
			Vector3f p0 = Vector3f(0, -1, -1), p1(0, 1, 0), p2(0, 0, 1);

			float distance = FLT_MAX;
			Vector3f bc;

			p[0] = RadGrabber::IntersectRayAndTriangle(r, p0, p1, p2, distance, bc);

			Bounds b = Bounds(Vector3f::Zero(), Vector3f::One() / 2);
			float t = FLT_MAX;

			p[1] = b.Intersect(r, t);
		}

		int RayIntersect()
		{
			Ray r = Ray(Vector3f(-10, 0, 0), Vector3f(1, 0, 0));

			{
				Vector3f p0 = Vector3f(0, -1, -1), p1(0, 1, 0), p2(0, 0, 1);

				float distance = FLT_MAX;
				Vector3f bc;

				printf("IntersectRayAndTriangle return %d\n", IntersectRayAndTriangle(r, p0, p1, p2, distance, bc));
			}

			{
				Bounds b = Bounds(Vector3f::Zero(), Vector3f::One() / 2);
				float t;

				printf("Bounds::Intersect() return %d\n", b.Intersect(r, t));
			}

			{
				Quaternion q;
				q = Quaternion(0.f, 0.7f, 0.f, 0.7f);
				Rotate(q, r.direction);
				printf("%.1f, %.1f, %.1f\n", r.direction.x, r.direction.y, r.direction.z);
				q = Quaternion(0.7f, 0.f, 0.f, 0.7f);
				Rotate(q, r.direction);
				printf("%.1f, %.1f, %.1f\n", r.direction.x, r.direction.y, r.direction.z);
			}

			int *dp = (int*)MAllocDevice(sizeof(int) * 2);
			RayIntersectTestKernel << <1, 1 >> > (dp);
			int hp[2];
			gpuErrchk(cudaMemcpy(hp, dp, sizeof(int) * 2, cudaMemcpyKind::cudaMemcpyDeviceToHost));
			gpuErrchk(cudaDeviceSynchronize());

			printf("%d, %d\n", hp[0], hp[1]);

			printf("IntersectRayAndTriangle return %d\n", hp[0]);
			printf("Bounds::Intersect() return %d\n", hp[1]);

			return 0;
		}
	}
}

