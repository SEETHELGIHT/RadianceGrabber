#include <time.h>

#include "DataTypes.cuh"
#include "Sample.cuh"

namespace RadGrabber
{
	namespace Test
	{

		int UniformRandomSampleTest()
		{
			Vector3f up = Vector3f(0, 0, 1);
			float cos = 0, maxcos = 0, mincos = FLT_MAX;
			srand(time(nullptr));
			float cosWidth = 0;
			while (true)
			{
				cos = 0; maxcos = 0; mincos = FLT_MAX;

				printf("cosWidth::");
				scanf("%f", &cosWidth);

				for (int count = 0; count < 1000000; count++)
				{
					Vector2f rv = Vector2f((float)rand() / RAND_MAX, (float)rand() / RAND_MAX);
					Vector3f v = UniformSampleConeInFrame(up, rv, cosWidth);
					float	vcos = Dot(v, up),
							vacos = acos(vcos);

					maxcos = fmax(vcos, maxcos);
					mincos = fmin(vcos, mincos);
					cos = cos * count / (count + 1) + vcos / (count + 1);
				}

				printf("avg::%.2f, min::%.2f, max::%.2f\n", cos, mincos, maxcos);
			}

			return 0;
		}
	}
}