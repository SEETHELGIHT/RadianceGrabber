#include "image.h"

namespace RadGrabber
{
	namespace Test
	{
		int PNGToMJPGTest()
		{
			VideoWrite("./temp/%s", ".png", "./SimpleScene.avi", 256, 256, 60, 113);
			VideoWrite2("./temp/%s.png", "./ddd.avi", 60.0, 168, 101, 269);

			return 0;
		}
	}
}
