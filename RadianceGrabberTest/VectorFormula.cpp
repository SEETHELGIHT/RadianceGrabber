#include <time.h>

#include "DataTypes.cuh"
#include "Sample.cuh"

namespace RadGrabber
{
	namespace Test
	{
		int VectorFormulaTest()
		{
			Vector3f
				normal = Vector3f(0, 1, 0),
				incidant = -Vector3f(1, 1, 1).normalized(),
				reflected = Reflect(incidant, normal);

			printf("normal(%.2f, %.2f, %.2f)\n", normal.x, normal.y, normal.z);
			printf("incidant(%.2f, %.2f, %.2f)\n", incidant.x, incidant.y, incidant.z);
			printf("reflected(%.2f, %.2f, %.2f)\n", reflected.x, reflected.y, reflected.z);

			return 0;
		}
	}
}