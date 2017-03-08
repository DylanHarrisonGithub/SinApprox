#include "stdafx.h"
#include <iostream>
#include <chrono>
#include <intrin.h>
#include <time.h>

typedef std::chrono::high_resolution_clock Clock;

using namespace std;

double hornersMethod(double x, double* c, int length) {

	double y = 0;
	for (int i = 0; i < length; i++) {
		y = c[i] + x*y;
	}

	return y;
}

double hornersMethodSIMD(double x, double* c, unsigned int length) {

	__m128d u, v, w;

	u.m128d_f64[0] = x*x;
	u.m128d_f64[1] = x*x;						//u = | x^2 | x^2 |

	w.m128d_f64[0] = c[0];
	w.m128d_f64[1] = c[1];						//w = | c[1] | c[0] |

	for (int i = 2; i < (length - 1); i += 2) {
		v.m128d_f64[0] = c[i];
		v.m128d_f64[1] = c[i + 1];				//v = | c[i+1] | c[i] |
		w = _mm_mul_pd(u, w);					//w = | (x^2)*w[1] | (x^2)*w[0] |
		w = _mm_add_pd(v, w);					//w = | c[i+1] + x*w[1] | c[i] + x*w[0] |
	}

	if ((length % 2) == 0) {
		w.m128d_f64[0] *= x;					//w = | w[1] | x*w[0] |
	}
	else {
		u.m128d_f64[0] = x*x;
		u.m128d_f64[1] = x;						//u = | x | x^2 |
		w = _mm_mul_pd(u, w);					//w = | x*w[1] | (x^2)*w[0] |
		w.m128d_f64[0] += c[length - 1];		//w = | w[1] | c[n] + w[0] |
	}
	w = _mm_hadd_pd(w, w);						//w = | w[0]+w[1] | w[0]+w[1] |

	return w.m128d_f64[0];
}

double unrolledHornerSinP4(double x) {

	__declspec(align(16)) double c4[] = { 0.03720932737240900, -0.23379309903632944, 0.05446968167436379, 0.98260114780498042, 0.00131345589768425 };
	__declspec(align(16)) double y;

	y = c4[0];
	y = c4[1] + x*y;
	y = c4[2] + x*y;
	y = c4[3] + x*y;
	y = c4[4] + x*y;

	return y;
}

double unrolledHornerSinP4SIMD(double x) {

	__declspec(align(16)) double c4[] = { 0.03720932737240900, -0.23379309903632944, 0.05446968167436379, 0.98260114780498042, 0.00131345589768425 };

	__m128d u, v, w;

	u.m128d_f64[0] = x*x;
	u.m128d_f64[1] = x*x;						//u = | x^2 | x^2 |
	v.m128d_f64[0] = c4[0];
	v.m128d_f64[1] = c4[1];						//v = | c[1] | c[0] |
	w = _mm_mul_pd(u, v);						//w = | (x^2)*c[1] | (x^2)*c[0] |
	v.m128d_f64[0] = c4[2];
	v.m128d_f64[1] = c4[3];						//v = | c[3] | c[2] |
	w = _mm_add_pd(w, v);						//w = | c[3] + (x^2)*c[1] | c[2] + (x^2)*c[0] |
	u.m128d_f64[0] = x*x;
	u.m128d_f64[1] = x;							//u = | x | x^2 |
	w = _mm_mul_pd(u, w);						//w = | x*w[1] | (x^2)*w[0] |
	w.m128d_f64[0] += c4[4];					//w = | w[1] | c[n] + w[0] |
	w = _mm_hadd_pd(w, w);						//w = | w[0]+w[1] | w[0]+w[1] |

	return w.m128d_f64[0];
}


double unrolledHornerSinP8(double x) {

	__declspec(align(16)) double c8[] = { 2.32374889636375e-005, -2.92010889608420e-004, 2.19672210664518e-004, 8.01674986535062e-003, 2.79388739405372e-004, -1.66810967472702e-001, 3.94059931985509e-005, 9.99995401622597e-001, 1.31570823227546e-007 };
	__declspec(align(16)) double y;

	y = c8[0];
	y = c8[1] + x*y;
	y = c8[2] + x*y;
	y = c8[3] + x*y;
	y = c8[4] + x*y;
	y = c8[5] + x*y;
	y = c8[6] + x*y;
	y = c8[7] + x*y;
	y = c8[8] + x*y;

	return y;
}

double unrolledHornerSinP8SIMD(double x) {

	__declspec(align(16)) double c8[] = { 2.32374889636375e-005, -2.92010889608420e-004, 2.19672210664518e-004, 8.01674986535062e-003, 2.79388739405372e-004, -1.66810967472702e-001, 3.94059931985509e-005, 9.99995401622597e-001, 1.31570823227546e-007 };
	__m128d u, v, w;

	u.m128d_f64[0] = x*x;
	u.m128d_f64[1] = x*x;						//u = | x^2 | x^2 |
	v.m128d_f64[0] = c8[0];
	v.m128d_f64[1] = c8[1];						//v = | c[1] | c[0] |

	w = _mm_mul_pd(u, v);						//w = | (x^2)*c[1] | (x^2)*c[0] |
	v.m128d_f64[0] = c8[2];
	v.m128d_f64[1] = c8[3];						//v = | c[3] | c[2] |
	w = _mm_add_pd(w, v);						//w = | c[3] + (x^2)*c[1] | c[2] + (x^2)*c[0] |

	w = _mm_mul_pd(u, w);						//w = | (x^2)*(c[3] + (x^2)*c[1]) | (x^2)*(c[2] + (x^2)*c[0]) |
	v.m128d_f64[0] = c8[4];
	v.m128d_f64[1] = c8[5];						//v = | c[5] | c[6] |
	w = _mm_add_pd(w, v);						//w = | c[5] + (x^2)*(c[3] + (x^2)*c[1]) | c[6] + (x^2)*(c[2] + (x^2)*c[0]) |

	w = _mm_mul_pd(u, w);
	v.m128d_f64[0] = c8[6];
	v.m128d_f64[1] = c8[7];
	w = _mm_add_pd(w, v);						//...					

	u.m128d_f64[0] = x*x;
	u.m128d_f64[1] = x;							//u = | x | x^2 |
	w = _mm_mul_pd(u, w);						//w = | x*w[1] | (x^2)*w[0] |
	w.m128d_f64[0] += c8[8];					//w = | w[1] | c[n] + w[0] |
	w = _mm_hadd_pd(w, w);						//w = | w[0]+w[1] | w[0]+w[1] |

	return w.m128d_f64[0];
}

double factoredP4Sin(double x) {

	__declspec(align(16)) double r4[] = { 4.86760270318095678, 3.14292946639567994, -1.72601004959195414, -0.00133681280580084 };
	__declspec(align(16)) double r4Magnitude = 0.03720932737240900;

	return r4Magnitude*(x - r4[0])*(x - r4[1])*(x - r4[2])*(x - r4[3]);
}

double factoredP4SinSIMD(double x) {

	__declspec(align(16)) double r4[] = { 4.86760270318095678, 3.14292946639567994, -1.72601004959195414, -0.00133681280580084 };
	__declspec(align(16)) double r4Magnitude = 0.03720932737240900;

	__m128d u, v, w;

	u.m128d_f64[0] = r4[0];
	u.m128d_f64[1] = r4[1];				//u = | r4[1] | r4[0] |
	v.m128d_f64[0] = r4[2];
	v.m128d_f64[1] = r4[3];				//v = | r4[3] | r4[2] |
	w.m128d_f64[0] = x;
	w.m128d_f64[1] = x;					//w = | x | x |

	u = _mm_sub_pd(w, u);				//u = | x - r4[1] | x - r4[0] |
	v = _mm_sub_pd(w, v);				//v = | x - r4[3] | x - r4[2] |
	w = _mm_mul_pd(u, v);				//u = | (x - r4[1])(x - r4[3]) | (x - r4[0])(x - r4[2]) |
	w.m128d_f64[0] *= w.m128d_f64[1];	//w[0] = (x - r4[1])(x - r4[3])(x - r4[0])(x - r4[2])
	w.m128d_f64[0] *= r4Magnitude;		//w[0] = r4Magnitude*(x - r4[1])(x - r4[3])(x - r4[0])(x - r4[2])

	return w.m128d_f64[0];
}

double factoredP8Sin(double x) {

	__declspec(align(16)) double r4[] = { 5.895452530035389, 3.141592785174156, -2.753860092985270, -0.000000131571428 };
	__declspec(align(16)) double q1[] = { -13.1185097003180, 48.6766343231151 };
	__declspec(align(16)) double q2[] = { 6.83532455770833, 17.3332241883087 };
	__declspec(align(16)) double r8Magnitude = 2.32374889636375e-005;

	return r8Magnitude*(x - r4[0])*(x - r4[1])*(x - r4[2])*(x - r4[3])*(x*x + q1[0] * x + q1[1])*(x*x + q2[0] * x + q2[1]);
}

double factoredP8SinSIMD(double x) {

	__declspec(align(16)) double r4[] = { 5.895452530035389, 3.141592785174156, -2.753860092985270, -0.000000131571428 };
	__declspec(align(16)) double q1[] = { -13.1185097003180, 48.6766343231151 };
	__declspec(align(16)) double q2[] = { 6.83532455770833, 17.3332241883087 };
	__declspec(align(16)) double r8Magnitude = 2.32374889636375e-005;

	__m128d s, t, u, v, w;

	u.m128d_f64[0] = r4[0];
	u.m128d_f64[1] = r4[1];				//u = | r4[1] | r4[0] |
	v.m128d_f64[0] = r4[2];
	v.m128d_f64[1] = r4[3];				//v = | r4[3] | r4[2] |
	w.m128d_f64[0] = x;
	w.m128d_f64[1] = x;					//w = | x | x |

	u = _mm_sub_pd(w, u);				//u = | x - r4[1] | x - r4[0] |
	v = _mm_sub_pd(w, v);				//v = | x - r4[3] | x - r4[2] |
	u = _mm_mul_pd(u, v);				//u = | (x - r4[1])(x - r4[3]) | (x - r4[0])(x - r4[2]) |

	s.m128d_f64[0] = q1[0];
	s.m128d_f64[1] = q2[0];				//s = | q2[0] | q1[0] |
	s = _mm_mul_pd(s, w);				//s = | x*q2[0] | x * q1[0] |

	t.m128d_f64[0] = q1[1];
	t.m128d_f64[1] = q2[1];				//t = | q2[1] | q1[1] |
	w = _mm_mul_pd(w, w);				//w = | x^2 | x^2 |
	s = _mm_add_pd(t, s);				//s = | x*q2[0] + q2[1] | x * q1[0] + q1[1] |
	w = _mm_add_pd(w, s);				//w = | x^2 + x*q2[0] + q2[1] | x^2 + x * q1[0] + q1[1] |
	w = _mm_mul_pd(w, u);				//w = | (x - r4[1])(x - r4[3])(x^2 + x*q2[0] + q2[1]) | (x - r4[0])(x - r4[2])(x^2 + x * q1[0] + q1[1]) |
	w.m128d_f64[0] *= w.m128d_f64[1];	//w[0] = w[1]*w[0]
	w.m128d_f64[0] *= r8Magnitude;		//w[0] = r4Magnitude*w[0]

	return w.m128d_f64[0];
}

int main()
{

	__declspec(align(16)) double c2[] = { -0.4176977570064662,   1.3122362048324483, -0.0504654977784461 };
	__declspec(align(16)) double c4[] = { 0.03720932737240900, -0.23379309903632944, 0.05446968167436379, 0.98260114780498042, 0.00131345589768425 };
	__declspec(align(16)) double c6[] = { -1.27871387060836e-003, 1.20515943047020e-002, -5.81476368125425e-003, -1.61705542577131e-001, -2.14775276097336e-003, 1.00038803940859e+000, -1.70004824988927e-005 };
	__declspec(align(16)) double c8[] = { 2.32374889636375e-005, -2.92010889608420e-004, 2.19672210664518e-004, 8.01674986535062e-003, 2.79388739405372e-004, -1.66810967472702e-001, 3.94059931985509e-005, 9.99995401622597e-001, 1.31570823227546e-007 };

	__declspec(align(16)) double x;
	int iterations = 1000000;

	cout << endl << "Accuracy Test: " << endl;

	cout << endl << "horner's method c++: " << endl;
	for (int n = 0; n < 100; n++) {
		x = 3.1415*((double)n / 100.0);
		cout << sin(x) - hornersMethod(x, c8, 9)
			<< ", " << sin(x) - hornersMethod(x, c6, 7)
			<< ", " << sin(x) - hornersMethod(x, c4, 5)
			<< ", " << sin(x) - hornersMethod(x, c2, 3)
			<< endl;
	}

	cout << endl << "horner's method intrinsics: " << endl;
	for (int n = 0; n < 100; n++) {
		x = 3.1415*((double)n / 100.0);
		cout << sin(x) - hornersMethodSIMD(x, c8, 9)
			<< ", " << sin(x) - hornersMethodSIMD(x, c6, 7)
			<< ", " << sin(x) - hornersMethodSIMD(x, c4, 5)
			<< ", " << sin(x) - hornersMethodSIMD(x, c2, 3)
			<< endl;
	}

	cout << endl << "unrolled horner's method c++: " << endl;
	for (int n = 0; n < 100; n++) {
		x = 3.1415*((double)n / 100.0);
		cout << sin(x) - unrolledHornerSinP8(x)
			<< ", " << sin(x) - unrolledHornerSinP4(x)
			<< endl;
	}

	cout << endl << "unrolled horner's method intrinsics: " << endl;
	for (int n = 0; n < 100; n++) {
		x = 3.1415*((double)n / 100.0);
		cout << sin(x) - unrolledHornerSinP8SIMD(x)
			<< ", " << sin(x) - unrolledHornerSinP4SIMD(x)
			<< endl;
	}

	cout << endl << "factored c++: " << endl;
	for (int n = 0; n < 100; n++) {
		x = 3.1415*((double)n / 100.0);
		cout << sin(x) - factoredP8Sin(x)
			<< ", " << sin(x) - factoredP4Sin(x)
			<< endl;
	}

	cout << endl << "factored with intrinsics: " << endl;
	for (int n = 0; n < 100; n++) {
		x = 3.1415*((double)n / 100.0);
		cout << sin(x) - factoredP8SinSIMD(x)
			<< ", " << sin(x) - factoredP4SinSIMD(x)
			<< endl;
	}

	srand(time(NULL));
	x = ((double)rand() / (RAND_MAX));

	for (int test = 1; test < 4; test++) {

		cout << endl << "Time Trial: " << test << endl;

		// Machine sin(x).
		cout << "machine sin(x)" << endl;
		auto t1 = Clock::now();
		for (int n = 0; n < iterations; n++) {
			x = sin(x);
		}
		auto t2 = Clock::now();
		std::cout << "time: "
			<< std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count()
			<< " nanoseconds" << std::endl;
		cout << "ns/f(x): "
			<< (double)std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count() / iterations << endl;

		// Horner's method c++.
		cout << endl << "horner's method c++" << endl;
		t1 = Clock::now();
		for (int n = 0; n < iterations; n++) {
			x = hornersMethod(x, c8, 9);
		}
		t2 = Clock::now();
		std::cout << "time: "
			<< std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count()
			<< " nanoseconds" << std::endl;
		cout << "ns/f(x): "
			<< (double)std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count() / iterations << endl;

		// unrolled p4 horners method in c++.
		cout << endl << "unrolled p4 horners method in c++" << endl;
		t1 = Clock::now();
		for (int n = 0; n < iterations; n++) {
			x = unrolledHornerSinP4(x);
		}
		t2 = Clock::now();
		std::cout << "time: "
			<< std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count()
			<< " nanoseconds" << std::endl;
		cout << "ns/f(x): "
			<< (double)std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count() / iterations << endl;

		// unrolled p8 horners method in c++.
		cout << endl << "unrolled p8 horners method in c++" << endl;
		t1 = Clock::now();
		for (int n = 0; n < iterations; n++) {
			x = unrolledHornerSinP8(x);
		}
		t2 = Clock::now();
		std::cout << "time: "
			<< std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count()
			<< " nanoseconds" << std::endl;
		cout << "ns/f(x): "
			<< (double)std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count() / iterations << endl;

		// factored quartic in c++.
		cout << endl << "factored quartic in c++" << endl;
		t1 = Clock::now();
		for (int n = 0; n < iterations; n++) {
			x = factoredP4Sin(x);
		}
		t2 = Clock::now();
		std::cout << "time: "
			<< std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count()
			<< " nanoseconds" << std::endl;
		cout << "ns/f(x): "
			<< (double)std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count() / iterations << endl;

		// factored p_8 in c++.
		cout << endl << "factored p_8 in c++" << endl;
		t1 = Clock::now();
		for (int n = 0; n < iterations; n++) {
			x = factoredP8Sin(x);
		}
		t2 = Clock::now();
		std::cout << "time: "
			<< std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count()
			<< " nanoseconds" << std::endl;
		cout << "ns/f(x): "
			<< (double)std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count() / iterations << endl;

		// Horner's method with intrinsics.
		cout << endl << "horner's method with asm intrinsics" << endl;
		t1 = Clock::now();
		for (int n = 0; n < iterations; n++) {
			x = hornersMethodSIMD(x, c8, 9);
		}
		t2 = Clock::now();
		std::cout << "time: "
			<< std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count()
			<< " nanoseconds" << std::endl;
		cout << "ns/f(x): "
			<< (double)std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count() / iterations << endl;

		// unrolled p4 horners method with asm intrinsics.
		cout << endl << "unrolled p4 horners method with asm intrinsics" << endl;
		t1 = Clock::now();
		for (int n = 0; n < iterations; n++) {
			x = unrolledHornerSinP4SIMD(x);
		}
		t2 = Clock::now();
		std::cout << "time: "
			<< std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count()
			<< " nanoseconds" << std::endl;
		cout << "ns/f(x): "
			<< (double)std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count() / iterations << endl;

		// unrolled p8 horners method with asm intrinsics.
		cout << endl << "unrolled p8 horners method with asm intrinsics" << endl;
		t1 = Clock::now();
		for (int n = 0; n < iterations; n++) {
			x = unrolledHornerSinP8SIMD(x);
		}
		t2 = Clock::now();
		std::cout << "time: "
			<< std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count()
			<< " nanoseconds" << std::endl;
		cout << "ns/f(x): "
			<< (double)std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count() / iterations << endl;

		// factored quartic intrinsics.
		cout << endl << "factored quartic with intrinsics" << endl;
		t1 = Clock::now();
		for (int n = 0; n < iterations; n++) {
			x = factoredP4SinSIMD(x);
		}
		t2 = Clock::now();
		std::cout << "time: "
			<< std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count()
			<< " nanoseconds" << std::endl;
		cout << "ns/f(x): "
			<< (double)std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count() / iterations << endl;

		// factored p_8 SIMD intrinsics.
		cout << endl << "factored p_8 intrinsics" << endl;
		t1 = Clock::now();
		for (int n = 0; n < iterations; n++) {
			x = factoredP8SinSIMD(x);
		}
		t2 = Clock::now();
		std::cout << "time: "
			<< std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count()
			<< " nanoseconds" << std::endl;
		cout << "ns/f(x): "
			<< (double)std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count() / iterations << endl;
	}

	return 0;

}

