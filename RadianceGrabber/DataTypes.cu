#include "DataTypes.cuh"

namespace RadGrabber
{
	__host__ float RandomFloat(float min, float max)
	{
		// this  function assumes max > min, you may want 
		// more robust error checking for a non-debug build
		assert(max > min);
		float random = ((float)rand()) / (float)RAND_MAX;

		// generate (in your case) a float between 0 and (4.5-.78)
		// then add .78, giving you a float between .78 and 4.5
		float range = max - min;
		return (random*range) + min;
	}

	__device__ __host__ void Quaternion::Rotate(Vector3f& p) const
	{
		float num = x * 2;
		float num2 = y * 2;
		float num3 = z * 2;
		float num4 = x * num;
		float num5 = y * num2;
		float num6 = z * num3;
		float num7 = x * num2;
		float num8 = x * num3;
		float num9 = y * num3;
		float num10 = w * num;
		float num11 = w * num2;
		float num12 = w * num3;

		Vector3f result;
		result.x = (1 - (num5 + num6)) * p.x + (num7 - num12) * p.y + (num8 + num11) * p.z;
		result.y = (num7 + num12) * p.x + (1 - (num4 + num6)) * p.y + (num9 - num10) * p.z;
		result.z = (num8 - num11) * p.x + (num9 + num10) * p.y + (1 - (num4 + num5)) * p.z;
		p = result;
	}
}
