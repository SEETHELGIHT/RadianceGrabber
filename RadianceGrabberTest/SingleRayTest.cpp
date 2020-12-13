#include <random>

#include "Marshal.cuh"
#include "AcceleratedAggregate.h"
#include "ConfigUtility.h"
#include "Pipeline.h"

namespace RadGrabber
{
	namespace Test
	{
		int SingleRayTest()
		{
			std::random_device rd; std::mt19937 mersenne(rd()); // Create a mersenne twister, seeded using the random device

			int result;

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

			int width = 0, height = 0, 
				confirmMode = 0, 
				randomMode = 0,
				interactMode = 0;
			float step = 0.125f;

			printf("Is Confirm Mode?");
			scanf("%d", &confirmMode);
			while ('\n' != getchar());

			printf("Is Random Mode?");
			scanf("%d", &randomMode);
			while ('\n' != getchar());

			while (true)
			{
				Vector2f v;
				if (randomMode)
					v = Vector2f(distribution(generator), distribution(generator));
				else
					v = Vector2f((float)width * step, (float)height * step);
				std::cout << "(" << v.x << "," << v.y << ")" << std::endl;

				ColorRGB luminance = ColorRGB::Zero(), throughput = ColorRGB::One();
				Ray r;
				SurfaceIntersection isect;
				memset(&isect, 0, sizeof(SurfaceIntersection));
				GetRay(
					req->input.in.cameraBuffer[req->input.in.selectedCameraIndex].projectionInverseMatrix, 
					req->input.in.cameraBuffer[req->input.in.selectedCameraIndex].cameraInverseMatrix, 
					v, r
				);

				for (int rayBoundCount = 0; ; rayBoundCount++)
				{
					printf("%dth ray(origin=(%.2f, %.2f, %.2f), dir=(%.3f, %.3f, %.3f))\n", rayBoundCount, r.origin.x, r.origin.y, r.origin.z, r.direction.x, r.direction.y, r.direction.z);
					if (result = agg->Intersect(r, isect, 0))
					{
						printf("normal :: hit!, isect.isGeometry :: %d, isect.itemIndex :: %d\n", isect.isGeometry, isect.itemIndex);

						if (isect.isGeometry)
						{
							char c = 'y';
							if (confirmMode)
							{
								printf("Geometry HIT! Continue next ray? (Y or Others) : ");
								c = getchar();
							}

							if (tolower(c) == 'y')
							{
								Vector3f dir;
								MaterialChunk mat = min->materialBuffer[isect.itemIndex];
								ColorRGBA albedoAndAlpha = mat.URPLit.SampleAlbedoAndAlpha(imin->textureBuffer, isect.uv);
								std::uniform_real_distribution<float> urd;
								Vector2f randomSample = Vector2f(urd(mersenne), urd(mersenne));

								ColorRGB bxdf;
								float pdf;

								if (isect.isGeometry)
								{
									if (!interactMode)
									{
										MicrofacetInteract(
											mat.URPLit,
											albedoAndAlpha, isect, r.direction, randomSample,
											mat.URPLit.SampleMetallic(imin->textureBuffer, isect.uv), mat.URPLit.SampleSmoothness(imin->textureBuffer, isect.uv),
											bxdf, dir, pdf
										);

										if (pdf)
											throughput = throughput * bxdf * Abs(Dot(dir, isect.normal)) / pdf;
										if (mat.URPLit.IsEmission())
											luminance = luminance + throughput * bxdf;
									}
									else
									{
										CoarseInteract(
											mat.URPLit,
											albedoAndAlpha, isect, r.direction, randomSample,
											mat.URPLit.SampleMetallic(imin->textureBuffer, isect.uv), mat.URPLit.SampleSmoothness(imin->textureBuffer, isect.uv),
											bxdf, dir, pdf
										);

										throughput = throughput * bxdf;
									}

									printf("bxdf : (%.3f, %.3f, %.3f), pdf : %.3f\n", bxdf.r, bxdf.g, bxdf.b, pdf);
								}
								else
								{
									LightChunk& l = min->lightBuffer[isect.itemIndex];
									ColorRGB emittedLight;

									l.GetLightInteract(isect, r, emittedLight, pdf);

									if (pdf)
										luminance = luminance + throughput * emittedLight / pdf;

									printf("light : (%.3f, %.3f, %.3f), pdf : %.3f\n", emittedLight.r, emittedLight.g, emittedLight.b, pdf);
									break;
								}

								r.origin = isect.position;
								r.direction = dir;
							}
							else
								break;

							printf("luminance : (%.3f, %.3f, %.3f), throughput : (%.3f, %.3f, %.3f)\n", luminance.r, luminance.g, luminance.b, throughput.r, throughput.g, throughput.b);

							if (confirmMode)
								while ('\n' != getchar());
						}
						else
						{
							printf("Light HIT! Terminated..\n");

							LightChunk& l = min->lightBuffer[isect.itemIndex];
							ColorRGB emittedLight;
							float pdf;

							l.GetLightInteract(
								isect,
								r,
								emittedLight,
								pdf
							);

							if (pdf)
								luminance = luminance + throughput * emittedLight / pdf;

							break;
						}
					}
					else
					{
						printf("normal :: missed!@!@!@#@\n");
						break;
					}
				}

				printf("end of ray, ");
				printf("luminance : (%.3f, %.3f, %.3f), throughput : (%.3f, %.3f, %.3f)\n", luminance.r, luminance.g, luminance.b, throughput.r, throughput.g, throughput.b);

				ADJUST_RAY:

				std::cout << "Ray Position :: " << "(" << width << "*" << step << "=" << width * step << ", " << height << "*" << step << "=" << height * step << ")" << std::endl;

				char c = getchar();
				while ('\n' != getchar());

				switch (c)
				{
				case 'W':
					height++;
				case 'w':
					height++;
					goto ADJUST_RAY;
					break;
				case 'S':
					height--;
				case 's':
					height--;
					goto ADJUST_RAY;
					break;
				case 'D':
					width++;
				case 'd':
					width++;
					goto ADJUST_RAY;
					break;
				case 'A':
					width--;
				case 'a':
					width--;
					goto ADJUST_RAY;
					break;
				case 'Q':
					width *= 4;
					height *= 4;
					step *= 0.25f;
					goto ADJUST_RAY;
					break;
				case 'q':
					width *= 2;
					height *= 2;
					step *= 0.5f;
					goto ADJUST_RAY;
					break;
				case 'E':
					width /= 4;
					height /= 4;
					step *= 4.f;
					goto ADJUST_RAY;
					break;
				case 'e':
					width /= 2;
					height /= 2;
					step *= 2.f;
					goto ADJUST_RAY;
					break;
				}

			}


			return 0;
		}
	}
}