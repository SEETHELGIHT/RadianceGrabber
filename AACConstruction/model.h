#pragma once
#ifndef MODEL_H
#define MODEL_H

#include "stdio.h"
#include "geometry.h"
#include <iostream>
using namespace std;

struct Model {
	int numVer, numTri;
	Point3D *ver, *bary;
	int *tri;

	void readFromFile(const char *file) {
		FILE *f;
		f = fopen(file, "r");
		fscanf(f, "%d%d", &numVer, &numTri);
		int i;
		ver = new Point3D[numVer];
		tri = new int[numTri*3];
		bary = new Point3D[numTri];
		for (i=0; i<numVer; i++) {
			fscanf(f, "%f%f%f", &ver[i][0], &ver[i][1], &ver[i][2]);
		}
		for (i=0; i<numTri; i++) {
			fscanf(f, "%d%d%d", &tri[i*3], &tri[i*3+1], &tri[i*3+2]);
			tri[i*3] --; tri[i*3+1] --; tri[i*3+2] --;
			bary[i][0] = (ver[tri[i*3]][0] + ver[tri[i*3+1]][0] + ver[tri[i*3+2]][0]) / 3;
			bary[i][1] = (ver[tri[i*3]][1] + ver[tri[i*3+1]][1] + ver[tri[i*3+2]][1]) / 3;
			bary[i][2] = (ver[tri[i*3]][2] + ver[tri[i*3+1]][2] + ver[tri[i*3+2]][2]) / 3;
		}
		fclose(f);
	}

	void freeMem() {
		delete[] ver;
		delete[] tri;
		delete[] bary;
	}
};

#endif
