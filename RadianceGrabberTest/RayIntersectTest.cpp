#include "DataTypes.cuh"
#include "Sample.cuh"

namespace RadGrabber
{
	namespace Test
	{
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
				q = Quaternion(0, 0.7, 0, 0.7);
				Rotate(q, r.direction);
				printf("%.1f, %.1f, %.1f\n", r.direction, r.direction.y, r.direction.z);
				q = Quaternion(0.7, 0, 0, 0.7);
				Rotate(q, r.direction);
				printf("%.1f, %.1f, %.1f\n", r.direction, r.direction.y, r.direction.z);
			}

			return 0;
		}
	}
}

