#include "utils/include.h"
#include "utils/complex.h"
#include "utils/timer.h"

inline void _mm512_FFT_BI_pd(double * wavere, double * waveim, int n, double * tempre, double* tempim, double** roots = nullptr) {

    // We need the length of the input to be a power of two and greater than 0
	assert((n & (n - 1)) == 0  && n > 0);
  	
    // Length of DFT equals to 4: base case
	if(n == 4){
		double wavere0 = wavere[0], waveim0 = waveim[0];
		double wavere1 = wavere[1], waveim1 = waveim[1];
		double wavere2 = wavere[2], waveim2 = waveim[2];
		double wavere3 = wavere[3], waveim3 = waveim[3];

		wavere[0] = wavere0 + wavere1 + wavere2 + wavere3;
		wavere[1] = wavere0 + waveim1 - wavere2 - waveim3;
		wavere[2] = wavere0 - wavere1 + wavere2 - wavere3;
		wavere[3] = wavere0 - waveim1 - wavere2 + waveim3;

		waveim[0] = waveim0 + waveim1 + waveim2 + waveim3;
		waveim[1] = waveim0 - wavere1 - waveim2 + wavere3;
		waveim[2] = waveim0 - waveim1 + waveim2 - waveim3;
		waveim[3] = waveim0 + wavere1 - waveim2 - wavere3;

		return;
	}

  	// Length of DFT equals to 2: base case
	if(n == 2){
	   double elem0re = wavere[0];
	   double elem0im = waveim[0];
	   double elem1re = wavere[1];
	   double elem1im = waveim[1];

	   wavere[0] = elem0re + elem1re;
	   waveim[0] = elem0im + elem1im;
	   wavere[1] = elem0re - elem1re;
	   waveim[1] = elem0im - elem1im;

	   return;
	}

	// Length of DFT equals to 1: base case
	if(n == 1){
		return;
    }

	double *vore, *voim, *vere, *veim;
    // We use this variable to remember if we need, at the end of the function, to
    // deallocate the look-up table
	bool deallocate = false;

	alignas(64) int baseIndexE[] = {0,2,4,6,8,10,12,14};
	alignas(64) int baseIndexO[] = {1,3,5,7,9,11,13,15};

	// Since n is a power of two, we get the exponent of the realtive power (its logarithm)
    // That's equivalent to the ASM instruction lzcnt
	int logn = 0;
	for(int x=n; x!=1; x>>=1){
        logn++;
    }

	// If it's necessary, we allocate the look-up table
	if(!roots){
		try{
			roots = new double*[2*logn];
		}
		catch (const std::bad_alloc& e) {
			exit(1);
		}
		deallocate = true;
	}

    // If we need them, we compute the necessary roots of the unity
	if(!roots[2*logn-1]){
    	try{
		  roots[2*logn-1] = new (std::align_val_t(64)) double[n/2];
		  roots[2*logn-2] = new (std::align_val_t(64)) double[n/2];
    	}
    	catch (const std::bad_alloc& e) {
			exit(1);
		}
		for(int m = 0; m < n / 2; m++){
			roots[2*logn-1][m] = + cos(2 * PI * m / (real) n);
			roots[2*logn-2][m] = - sin(2 * PI * m / (real) n);
		}
	}

	vere = tempre;
    veim = tempim;
	vore = tempre + n / 2;
    voim = tempim + n / 2;

	__m256i indexE = _mm256_load_si256((__m256i const*) baseIndexE);
	__m256i indexO = _mm256_load_si256((__m256i const*) baseIndexO);

	// We build ve and vo if n==8 (we can't use 512 gather instructions in this case)
	if(n==8){
		for (int k=0; k < n/2; k++) {
			vere[k] = wavere[2*k  ];
			veim[k] = waveim[2*k  ];
			vore[k] = wavere[2*k+1];
			voim[k] = waveim[2*k+1];
		}
	}
  // Otherwise, we can build ve and vo using gather instructions
	else{
		for (int k=0; k < n; k+=16) {
			__m512d temp0;
      
			temp0 = _mm512_i32gather_pd(indexE ,(double const*)&(wavere[k]),8);
			_mm512_store_pd((void*)&(vere[k/2]), temp0);
			temp0 = _mm512_i32gather_pd(indexE ,(double const*)&(waveim[k]),8);
			_mm512_store_pd((void*)&(veim[k/2]), temp0);

			temp0 = _mm512_i32gather_pd(indexO ,(double const*)&(wavere[k]),8);
			_mm512_store_pd((void*)&(vore[k/2]), temp0);
			temp0 = _mm512_i32gather_pd(indexO ,(double const*)&(waveim[k]),8);
			_mm512_store_pd((void*)&(voim[k/2]), temp0);
		}
	}

	// Recursive call of the function on both ve and vo
	_mm512_FFT_BI_pd(vere, veim, n / 2, wavere, waveim, roots);
	_mm512_FFT_BI_pd(vore, voim, n / 2, wavere, waveim, roots);

    // We build the result if n==8 (we need to use the 256 AVX instructions)
	if(n==8){
		__m256d reVect1, imVect1, reVect2, imVect2, reVect3, imVect3, prod;

		for (int m = 0; m < n / 2; m+=4) {
			reVect1 = _mm256_load_pd(&roots[2*logn-1][m]);
			imVect1 = _mm256_load_pd(&roots[2*logn-2][m]);
			reVect2 = _mm256_load_pd(&vore[m]);
			imVect2 = _mm256_load_pd(&voim[m]);

			prod     = _mm256_mul_pd  (imVect1,imVect2);
			reVect3  = _mm256_fmsub_pd(reVect1, reVect2, prod);
			prod     = _mm256_mul_pd  (reVect1,imVect2);
			imVect3  = _mm256_fmadd_pd(imVect1,reVect2, prod);

			reVect1 = _mm256_load_pd(&vere[m]);
			imVect1 = _mm256_load_pd(&veim[m]);

			_mm256_store_pd(&wavere[m],     _mm256_add_pd(reVect1, reVect3));
			_mm256_store_pd(&waveim[m],     _mm256_add_pd(imVect1, imVect3));
			_mm256_store_pd(&wavere[m+n/2], _mm256_sub_pd(reVect1, reVect3));
			_mm256_store_pd(&waveim[m+n/2], _mm256_sub_pd(imVect1, imVect3));
		}
	}

    // We build the result if n>8 (we can use the 512 AVX instructions)	
    else{
		__m512d reVect1, imVect1, reVect2, imVect2, reVect3, imVect3, prod;

		for (int m = 0; m < n / 2; m+=8) {
			reVect1 = _mm512_load_pd((void*)&roots[2*logn-1][m]);
			imVect1 = _mm512_load_pd((void*)&roots[2*logn-2][m]);
			reVect2 = _mm512_load_pd((void*)&vore[m]);
			imVect2 = _mm512_load_pd((void*)&voim[m]);

			prod     = _mm512_mul_pd  (imVect1,imVect2);
			reVect3  = _mm512_fmsub_pd(reVect1, reVect2, prod);
			prod     = _mm512_mul_pd  (reVect1,imVect2);
			imVect3  = _mm512_fmadd_pd(imVect1,reVect2, prod);

			reVect1 = _mm512_load_pd((void*)&vere[m]);
			imVect1 = _mm512_load_pd((void*)&veim[m]);

			_mm512_store_pd(&wavere[m],     _mm512_add_pd(reVect1, reVect3));
			_mm512_store_pd(&waveim[m],     _mm512_add_pd(imVect1, imVect3));
			_mm512_store_pd(&wavere[m+n/2], _mm512_sub_pd(reVect1, reVect3));
			_mm512_store_pd(&waveim[m+n/2], _mm512_sub_pd(imVect1, imVect3));
		}
	}

    // If deallocate==true, we need to deallocate the data 
	if(deallocate){
		for(int i=0; i<2*logn; i++){
			if(roots[i]){
				delete [] roots[i];
				roots[i] = nullptr;
			}
		}
		delete [] roots;
	}
	return;
}

int main(int argc, char* argv[]) {
	
	if(argc!=2) exit(1);
	const int N = atoi(argv[1]);

	alignas(64)  double waveRE[N];
	alignas(64)  double waveIM[N];
	alignas(64)  double scratchRe[N];
	alignas(64)  double scratchIM[N];

    for (int k = 0; k < N; k++) {
        waveRE[k] = 0.125 * cos(2 * PI * k / (real) N) + 0.5 * cos(2 * PI * 10*k / (real) N);
        waveIM[k] = 0.125 * sin(2 * PI * k / (real) N) + 0.5 * sin(2 * PI * 10*k / (real) N);
    }

    #ifdef __DEBUG

        {
            Timer t;
            _mm512_FFT_BI_pd(waveRE, waveIM, N, scratchRe, scratchIM);
        }
        
    #else

	    _mm512_FFT_BI_pd(waveRE, waveIM, N, scratchRe, scratchIM);

        for (int i = 0; i < N; i++){
            std::cout << waveRE[i] << '\t' << waveIM[i] << '\n';
        }

    #endif

	return 0;
}