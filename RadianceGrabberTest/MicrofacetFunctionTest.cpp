#include <time.h>
#include <random>

#include "Microfacet.h"
#include "Sample.cuh"

namespace RadGrabber
{
	namespace Test
	{
		int MicrofacetFunctionTest()
		{
			const int maxFuncs = 3;
			void (*funcs[maxFuncs])() = {
				[]() 
				{
					puts("==>> roughness to alpha");
					{
						int count = 10;
						for (int i = 0; i <= count; i++)
						{
							float roughness = (float)i / count;
							printf("%.2f --> %.2f\n", roughness, TrowbridgeReitzRoughnessToAlpha(roughness));
						}
					}
				},
				[]()
				{
					puts("==>> Sample Wh(a=0.5)");

					while (getchar() != 'q')
					{
						while (getchar() != '\n');

						std::default_random_engine generator;
						std::uniform_real_distribution<float> distribution(0, 1);

						Vector3f v = Vector3f(-1.f, -1.f, 1.f).normalized(), n = Vector3f(0.f, 0.f, 1.f);
						Vector3f wh = Vector3f::Zero(), wh2 = Vector3f::Zero();

						const int maxCount = 100000;
						float alpha = TrowbridgeReitzRoughnessToAlpha(0.5f);
						for (int i = 0; i < maxCount; i++)
						{
							Vector2f u = Vector2f(distribution(generator), distribution(generator));
							Vector3f sampledWh = TrowbridgeReitzSampleWh(v, u, alpha, alpha);
							wh += sampledWh;
							wh2 += sampledWh * sampledWh;
						}

						wh /= maxCount;
						wh2 /= maxCount;
						Vector3f varWh = wh2 - wh * wh;

						printf("v(%.2f, %.2f, %.2f), ", v.x, v.y, v.z);
						printf("n(%.2f, %.2f, %.2f), ", n.x, n.y, n.z);
						printf("alpha:%.2f ", alpha);
						puts("");
						printf("wh : mean(%.2f, %.2f, %.2f), var(%.2f, %.2f, %.2f)", wh.x, wh.y, wh.z, varWh.x, varWh.y, varWh.z);
						puts("");

					}
				},
				[]()
				{
				}
			};

			for (int i = 0; i < maxFuncs; i++, puts(""))
				funcs[i]();

			return 0;
		}
	}
}
