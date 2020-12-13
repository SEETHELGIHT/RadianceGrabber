#pragma once
#ifndef AAC_H
#define AAC_H

#include "geometry.h"
#include "model.h"
#include "assert.h"
#include <string>
using namespace std;
#define epsi 1e-8
#define sqr(a) (a)*(a)
#define set(a,b) memmove((a), (b), sizeof(BoundingBox))
#define setPoint(a,b) memmove((a), (b), sizeof(Point3D))
#define SurA(a) ((a[0].max-a[0].min)*(a[1].max-a[1].min)+(a[0].max-a[0].min)*(a[2].max-a[2].min)+(a[1].max-a[1].min)*(a[2].max-a[2].min))
const float infi = 1e10;

void inline update(BoundingBox &a, Point3D &b) {
	if (b[0] < a[0].min) a[0].min = b[0];
	if (b[1] < a[1].min) a[1].min = b[1];
	if (b[2] < a[2].min) a[2].min = b[2];
	if (b[0] > a[0].max) a[0].max = b[0];
	if (b[1] > a[1].max) a[1].max = b[1];
	if (b[2] > a[2].max) a[2].max = b[2];
}

void inline update(BoundingBox &a, BoundingBox &b) {
	if (b[0].min < a[0].min) a[0].min = b[0].min;
	if (b[1].min < a[1].min) a[1].min = b[1].min;
	if (b[2].min < a[2].min) a[2].min = b[2].min;
	if (b[0].max > a[0].max) a[0].max = b[0].max;
	if (b[1].max > a[1].max) a[1].max = b[1].max;
	if (b[2].max > a[2].max) a[2].max = b[2].max;
}

void inline update(BoundingBox &a, BoundingBox &b, BoundingBox &c) {
	a[0].min = (b[0].min < c[0].min) ? b[0].min : c[0].min;
	a[0].max = (b[0].max > c[0].max) ? b[0].max : c[0].max;
	a[1].min = (b[1].min < c[1].min) ? b[1].min : c[1].min;
	a[1].max = (b[1].max > c[1].max) ? b[1].max : c[1].max;
	a[2].min = (b[2].min < c[2].min) ? b[2].min : c[2].min;
	a[2].max = (b[2].max > c[2].max) ? b[2].max : c[2].max;
}

// Interval operations
struct Interval
{
	float min, max;
	Interval() { min = -(max = -1e10); }
	float range() { return max-min; }
	float mid() { return (max+min)/2; }
	void twice() { min *= 2; max *= 2; }
	int check(float x) { return min<=x && x<=max; }
	float update(Interval a) { return a.max < min ? max-a.max : a.min > max ? a.min-min : max-min; }
};

typedef Interval BoundingBox[3];

template<class T>
struct AAC {
	int minSize;
	double alpha;
	int table[100];
	char filename[100];
	struct Thread  // A single thread. In order to parallel, just start multiple threads.
	{
		float **area, *minArea;
		int *label, *minPos, *minLabel;
		int size;
		void init(int size) {
			this->size = size;
			label = new int [size];
			minArea = new float [size];
			minPos = new int [size];
			minLabel = new int [size];
			area = new float* [size];
			for (int i=0; i<size; i++)
				area[i] = new float [size];
		}
		~Thread() {
			delete[] label;
			delete[] minArea;
			delete[] minPos;
			delete[] minLabel;
			for (int i=0; i<size; i++)
				delete[] area[i];
			delete[] area;
		}
	};
	Model *model;

	Point3D *bary;
	int *set, *setTmp;
	T *code;
	int MortonDigit;

	int *lChild, *rChild, *nodeNum, *triNum;
	int curNode;
	BoundingBox *AABB, box;
	bool *isLeaf;
	double *cost, *surA;

	// Data input.
	void prepare(Model *modelInput, const char *filename, int delta, double alpha) {
		this->minSize = delta/2; this->alpha = alpha;
		for (int i=0; i<100; i++)
			table[i] = int(delta*pow(float(i)/delta, 0.5-alpha)/2-epsi)+1;
		memmove(this->filename, filename, strlen(filename)+1);
		model = modelInput;
		int i;
		Point3D *ver = model -> ver;
		int *tri = model -> tri;
		MortonDigit = 9 + ((model -> numTri >> 18) > 0);
		if (model -> numTri >= 1 << 20) MortonDigit ++;
		if (model -> numTri >= 1 << 22 && delta > 10) MortonDigit ++;
		cout << MortonDigit << endl;
		Point3D p1, p2, p3;
		lChild = new int [model -> numTri * 2];
		rChild = new int [model -> numTri * 2];
		AABB = new BoundingBox [model -> numTri * 2];
		triNum = new int [model -> numTri * 2];
		bary = new Point3D [model -> numTri * 2];
		for (i=0; i<model -> numTri; i++) {
			setPoint(p1, ver[tri[i*3]]  );
			setPoint(p2, ver[tri[i*3+1]]);
			setPoint(p3, ver[tri[i*3+2]]);
			AABB[i][0].min = min3(p1[0], p2[0], p3[0]);
			AABB[i][1].min = min3(p1[1], p2[1], p3[1]);
			AABB[i][2].min = min3(p1[2], p2[2], p3[2]);
			AABB[i][0].max = max3(p1[0], p2[0], p3[0]);
			AABB[i][1].max = max3(p1[1], p2[1], p3[1]);
			AABB[i][2].max = max3(p1[2], p2[2], p3[2]);
			bary[i][0] = (AABB[i][0].min + AABB[i][0].max)/2;
			bary[i][1] = (AABB[i][1].min + AABB[i][1].max)/2;
			bary[i][2] = (AABB[i][2].min + AABB[i][2].max)/2;
			triNum[i] = 1;
			update(box, bary[i]);
		}
		code = new T [model -> numTri];
		set = new int [model -> numTri];
		setTmp = new int [model -> numTri];
		cost = new double [model -> numTri * 2];
		surA = new double [model -> numTri * 2];
		isLeaf = new bool [model -> numTri * 2];
		nodeNum = new int [model -> numTri * 2];
	}

	// Main process
	void build() {
		curNode = model->numTri;
		Thread th;
		th.init(4*int(minSize*pow(model->numTri/2.0/minSize, 0.5-alpha/2)+1e-5));
		radixSort();
		int finalLen = 0;
		threadBuild(&th, 0, 0, model->numTri, MortonDigit*3-1, finalLen);
		AAClomerate(&th, 0, finalLen, 1, finalLen);

		pruneTree();
		printTreeBIN();
	}

	//Release memory.
	void freeMem() {
		delete[] set;
		delete[] code;
		delete[] setTmp;
		delete[] nodeNum;
		delete[] triNum;
		delete[] lChild;
		delete[] rChild;
		delete[] AABB;
		delete[] isLeaf;
		delete[] cost;
	}

	//Generate Morton Code.
	void radixSort() {
		T *mapping = new T [1<<MortonDigit];
		for (int i=0; i<(1<<MortonDigit); i++) {
			mapping[i] = 0;
			for (int j=0; j<MortonDigit; j++)
				mapping[i] += ((T)(i&(1<<j))>0) << (j*3);
		}
		double cutLen[] = { (1<<MortonDigit) / (box[0].max-box[0].min + 1e-8), 
							(1<<MortonDigit) / (box[1].max-box[1].min + 1e-8), 
							(1<<MortonDigit) / (box[2].max-box[2].min + 1e-8) };
		for (int i=0; i<model->numTri; i++) {
			code[i] = (mapping[int((bary[i][0]-box[0].min) * cutLen[0])] << 2) + 
					  (mapping[int((bary[i][1]-box[1].min) * cutLen[1])] << 1) + 
					  (mapping[int((bary[i][2]-box[2].min) * cutLen[2])]);
		}
		int totBuc = ( 1 << ((MortonDigit * 3 + 1) >> 1) ), digit = (MortonDigit * 3 + 1) >> 1;
		int maskLow = totBuc - 1, maskHigh = (totBuc - 1) << digit;
		int *cntHigh = new int [totBuc + 1], *cntLow = new int [totBuc + 1];
		for (int i=0; i<totBuc; i++)
			cntHigh[i] = cntLow[i] = 0;
		for (int i=0; i<model->numTri; i++)
			cntLow[(code[i] & maskLow) + 1] ++, cntHigh[((code[i] & maskHigh)>>digit) + 1] ++;
		for (int i=1; i<totBuc; i++)
			cntLow[i] += cntLow[i-1], cntHigh[i] += cntHigh[i-1];
		for (int i=0; i<model->numTri; i++)
			setTmp[cntLow[(code[i] & maskLow)] ++] = i;
		for (int i=0; i<model->numTri; i++)
			set[cntHigh[((code[setTmp[i]] & maskHigh)>>digit)] ++] = setTmp[i];
	}

	//A single agglomerative clustering thread.
	void threadBuild(Thread *th, int start, int startTri, int endTri, int digit, int &finalLen) {
		if (startTri >= endTri) {
			finalLen = 0; return;
		}
		if (endTri-startTri < minSize*2) {
			assert(start+endTri-startTri < th->size);
			setLeaf(th, start, startTri, endTri-startTri);
			AAClomerate(th, start, endTri-startTri, minSize, finalLen);
			return;
		}
		int s = startTri, t = endTri-1, mid;
		T tar = (T)1<<digit; 
		while (digit >= 0) {
			if (!(code[set[s]]&tar) && (code[set[t]]&tar)) break;
				digit --; 
				tar = (T)1<<digit;
		}

		if (digit < 0) s = (s+t) >> 1; else
		while (s<t) {
			mid = (s+t) >> 1;
			if (code[set[mid]] & tar)
				t = mid;
			else
				s = mid+1;
		}
		int len1, len2;
		threadBuild(th, start, startTri, s, digit-1, len1);
		threadBuild(th, start+len1, s, endTri, digit-1, len2);
		merge(th, start, len1, len2);
		finalLen = (endTri-startTri >= 100) ? f(endTri-startTri) : table[endTri-startTri];
		AAClomerate(th, start, len1+len2, finalLen, finalLen);
	}

	//Set leaves.
	void setLeaf(Thread *th, int start, int startTri, int numTri) {
		for (int i=0; i<numTri; i++) {
			th->minArea[start+i] = infi;
			th->label[start+i] = set[startTri+i];
		}
		for (int i=start; i<start+numTri; i++)  {
			for (int j=start; j<i; j++) {
				th->area[i][j] = SA(AABB[th->label[i]], AABB[th->label[j]]);
				if (th->area[i][j] < th->minArea[i]) {
					th->minArea[i] = th->area[i][j];
					th->minPos[i] = j; th->minLabel[i] = set[j];
				}
				if (th->area[i][j] < th->minArea[j]) {
					th->minArea[j] = th->area[i][j];
					th->minPos[j] = i; th->minLabel[j] = set[i];
				}
			}
		}
	}

	//The function to process agglomeration.
	void AAClomerate(Thread *th, int start, int startNum, int endNum, int &finalNum)  {
		int n = startNum;
		finalNum = fmin(startNum, endNum);
		int i, j, a, b, last = 0;
		while (n > endNum) {
			float mn = infi;
			for (i=start; i<start+n; i++) {
				if (th->minPos[i] == start+n) 
					if (last == start+n) th->minLabel[i] = -1; 
					else th->minPos[i] = last;
				if (th->minLabel[i] != th->label[th->minPos[i]]) {
					th->minArea[i] = infi;
					for (j=start; j<i; j++)
						if (th->area[i][j] < th->minArea[i]) {
							th->minArea[i] = th->area[i][j];
							th->minPos[i] = j; th->minLabel[i] = th->label[j];
						}
					for (j=i+1; j<start+n; j++)
						if (th->area[j][i] < th->minArea[i]) {
							th->minArea[i] = th->area[j][i];
							th->minPos[i] = j; th->minLabel[i] = th->label[j];
						}
				}
				if (th->minArea[i] < mn) {
					mn = th->minArea[i];
					a = i; b = th->minPos[i];
				}
			}
			lChild[curNode] = th->label[a];
			rChild[curNode] = th->label[b];
			update(AABB[curNode], AABB[lChild[curNode]], AABB[rChild[curNode]]);
			triNum[curNode] = triNum[lChild[curNode]] + triNum[rChild[curNode]];
			for (j=start; j<a; j++)
				th->area[a][j] = SA(AABB[curNode], AABB[th->label[j]]);
			for (j=a+1; j<start+n; j++)
				th->area[j][a] = SA(AABB[curNode], AABB[th->label[j]]);
			n --;
			for (j=start; j<b; j++)
				th->area[b][j] = th->area[start+n][j];
			for (j=b+1; j<start+n; j++)
				th->area[j][b] = th->area[start+n][j];
			th->label[a] = curNode;
			th->label[b] = th->label[start+n];
			th->minArea[b] = th->minArea[start+n];
			th->minLabel[b] = th->minLabel[start+n];
			th->minPos[b] = th->minPos[start+n];
			last = b;
			curNode ++;
		}
	}

	//Combine two subtaskes.
	void merge(Thread *th, int start, int len1, int len2) {
		for (int i=start; i<start+len1; i++)
			for (int j=start+len1; j<start+len1+len2; j++) {
				th->area[j][i] = SA(AABB[th->label[i]], AABB[th->label[j]]);
				if (th->area[j][i] < th->minArea[i]) {
					th->minArea[i] = th->area[j][i];
					th->minPos[i] = j; th->minLabel[i] = set[j];
				}
				if (th->area[j][i] < th->minArea[j]) {
					th->minArea[j] = th->area[j][i];
					th->minPos[j] = i; th->minLabel[j] = set[i];
				}
			}
	}

	// function F.
	int inline f(int len) { return int(minSize*pow(float(len)/minSize/2, 0.5-alpha)-epsi)+1; }

	// Prune tree at last. This can be also done in tree building process.
	void pruneTree() {
		for (int i=0; i<model->numTri; i++) {
			nodeNum[i] = 1;
			cost[i] = 2;
			surA[i] = SurA(AABB[i]);
			isLeaf[i] = true;
		}
		for (int i=model->numTri; i<curNode; i++) {
			surA[i] = SurA(AABB[i]);
			assert(SurA(AABB[i]) - SA(AABB[lChild[i]], AABB[rChild[i]]) < 1e-8);
			double tmp = surA[lChild[i]] / surA[i] * cost[lChild[i]] + surA[lChild[i]] / surA[i] * cost[lChild[i]];
			if (tmp+1 > triNum[i]*2) {
				cost[i] = triNum[i];
				nodeNum[i] = 1;
				isLeaf[i] = true;
			} else
			{
				cost[i] = tmp + 1;
				nodeNum[i] = nodeNum[lChild[i]] + nodeNum[rChild[i]] + 1;
				isLeaf[i] = false;		
			}
		}
	}

	void printTreeBIN(FILE *f, int root, int forb, double topArea) {
		if (forb) {
			if (root < model->numTri) fwrite(&root, sizeof(int), 1, f);
			else {
				printTreeBIN(f, lChild[root], 1, 0);
				printTreeBIN(f, rChild[root], 1, 0);
			}
			return;
		}
		double area = SurA(AABB[root]);
		int skip = (area / topArea > 0.75);
		if (!skip) topArea = area;
		fwrite(&isLeaf[root], sizeof(bool), 1, f);
		int ax = 0;
		if (!isLeaf[root]) {
			ax = nodeNum[lChild[root]]+1;
			fwrite(&ax, sizeof(int), 1, f);
			ax = 0;
			fwrite(&ax, sizeof(int), 1, f);
			fwrite(&skip, sizeof(int), 1, f);
			fwrite(&AABB[root][0].min, sizeof(float), 1, f);
			fwrite(&AABB[root][1].min, sizeof(float), 1, f);
			fwrite(&AABB[root][2].min, sizeof(float), 1, f);
			fwrite(&AABB[root][0].max, sizeof(float), 1, f);
			fwrite(&AABB[root][1].max, sizeof(float), 1, f);
			fwrite(&AABB[root][2].max, sizeof(float), 1, f);
			printTreeBIN(f, lChild[root], 0, topArea);
			printTreeBIN(f, rChild[root], 0, topArea);
		} else {
			fwrite(&triNum[root], sizeof(int), 1, f);
			fwrite(&ax, sizeof(int), 1, f);
			fwrite(&skip, sizeof(int), 1, f);
			fwrite(&AABB[root][0].min, sizeof(float), 1, f);
			fwrite(&AABB[root][1].min, sizeof(float), 1, f);
			fwrite(&AABB[root][2].min, sizeof(float), 1, f);
			fwrite(&AABB[root][0].max, sizeof(float), 1, f);
			fwrite(&AABB[root][1].max, sizeof(float), 1, f);
			fwrite(&AABB[root][2].max, sizeof(float), 1, f);
			printTreeBIN(f, root, 1, 0);
		}
	}

	void printTreeBIN() {
		FILE *f;
		f = fopen(filename, "wb");
		fwrite(&nodeNum[curNode-1], sizeof(int), 1, f);
		double area = SurA(AABB[curNode-1]);
		printTreeBIN(f, curNode-1, 0, area);
		fclose(f);
	}

	int axis(BoundingBox a, BoundingBox b);
};

float inline SA(BoundingBox a, BoundingBox b) {
	float x = fmax(a[0].max, b[0].max) - fmin(a[0].min, b[0].min);
	float y = fmax(a[1].max, b[1].max) - fmin(a[1].min, b[1].min);
	float z = fmax(a[2].max, b[2].max) - fmin(a[2].min, b[2].min);
	return x*y + x*z + y*z;
}

#endif
