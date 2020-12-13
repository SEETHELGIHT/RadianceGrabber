#define _CRT_SECURE_NO_WARNINGS

#include "AAC.h"
#include "geometry.h"
#include <string>
#include <cstdlib>
#include <iostream>
#include <vector>
using namespace std;

int main(int argc, char* argv[]) {
	Model m;
	char ch[100], file[] = "conference.mdl"; // "sponza.mdl", "fairy.mdl", "buddha.mdl", "ep2.mdl", "san-miguel.mdl"
	AAC<long long> b;
	m.readFromFile(file);
	if (m.numTri >= 1 << 20) {
		sprintf(ch, "%s-HQ.bin", file);
		AAC<long long> bvhLong;
		bvhLong.prepare(&m, ch, 20, 0.1);
		bvhLong.build();
		bvhLong.freeMem();
		sprintf(ch, "%s-fast.bin", file);
		bvhLong.prepare(&m, ch, 4, 0.2);
		bvhLong.build();
		bvhLong.freeMem();
	} else
	{
		AAC<int> bvhInt;
		sprintf(ch, "%s-HQ.bin", file);
		bvhInt.prepare(&m, ch, 20, 0.1);
		bvhInt.build();
		bvhInt.freeMem();
		sprintf(ch, "%s-fast.bin", file);
		bvhInt.prepare(&m, ch, 4, 0.2);
		bvhInt.build();
		bvhInt.freeMem();
	}
	return 0;
}
