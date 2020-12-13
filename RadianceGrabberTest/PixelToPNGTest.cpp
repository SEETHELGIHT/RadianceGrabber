#include "image.h"

namespace RadGrabber
{
	namespace Test
	{
		int PixelToPNGTest()
		{
			char buf[256], buf2[1024];
			const char* format = "./1234/dd_%s";
			int w = 4, h = 4;
			ColorRGBA* rgba = (ColorRGBA*)malloc(sizeof(ColorRGBA) * w * h);

			for (int k = 0; k < 4; k++)
			{
				sprintf(buf, "%03d", k);
				sprintf(buf2, format, buf);

				for (int i = 0; i < h; i++)
					for (int j = 0; j < w; j++)
						rgba[i * w + j] = ColorRGBA((float)((i + 1 + k) % w) / w, (float)((j + 1 + k + h/2) % h) / h, 1, 1);

				ImageWrite(buf2, ".png", rgba, 4, 4);
			}
			
			free(rgba);

			return 0;
		}
	}
}
