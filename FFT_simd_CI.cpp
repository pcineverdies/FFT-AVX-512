#include "utils/include.h"
#include "utils/complex.h"
#include "utils/timer.h"

inline void _mm512_FFT_CI_pd(complex * wave, int n, complex * temp, complex** roots = nullptr) {

    // We need the length of the input to be a power of two and greater than 0
	assert((n & (n - 1)) == 0  && n > 0); 

	__m512d roVect, voVect, temp0, temp1, temp2, temp3, Z;

	// Length of DFT equals to 4: base case
	if(n == 4){
		// We get the datas
		temp0 = _mm512_broadcast_f64x2(_mm_load_pd((double*)&(wave[0])));
		temp1 = _mm512_broadcast_f64x2(_mm_load_pd((double*)&(wave[1])));
		temp1 = _mm512_permute_pd(temp1, 0b01100110);
		temp2 = _mm512_broadcast_f64x2(_mm_load_pd((double*)&(wave[2])));
		temp3 = _mm512_broadcast_f64x2(_mm_load_pd((double*)&(wave[3])));
		temp3 = _mm512_permute_pd(temp3, 0b01100110);

		// We compute the DFT using the basic expression
		temp0 = _mm512_mask_sub_pd(temp0, 0b01111000, temp0, temp1);
		temp0 = _mm512_mask_add_pd(temp0, 0b10000111, temp0, temp1);
		temp0 = _mm512_mask_sub_pd(temp0, 0b11001100, temp0, temp2);
		temp0 = _mm512_mask_add_pd(temp0, 0b00110011, temp0, temp2);
		temp0 = _mm512_mask_sub_pd(temp0, 0b10110100, temp0, temp3);
		temp0 = _mm512_mask_add_pd(temp0, 0b01001011, temp0, temp3);

        // We store the data
		_mm512_store_pd((void*)&wave[0], temp0);

		return;
	}

	// Length of DFT equals to 2: base case
	else if(n == 2){
		__m128d elem_0 = _mm_load_pd((double*)&wave[0]);
		__m128d elem_1 = _mm_load_pd((double*)&wave[1]);
		_mm_store_pd((double*)&wave[0], _mm_add_pd(elem_0, elem_1));
		_mm_store_pd((double*)&wave[1], _mm_sub_pd(elem_0, elem_1));
		return;
	}

	// Length of DFT equals to 1: base case
	else if(n == 1){
		return;
    }
	
	complex *vo, *ve;
    // We use this variable to remember if we need, at the end of the function, to
    // deallocate the look-up table
	bool deallocate = false;    

	// Since n is a power of two, we get the exponent of the realtive power (its logarithm)
    // That's equivalent to the ASM instruction lzcnt
	int logn = 0;
	for(int x=n; x!=1; x>>=1){
        logn++;
    }

    // If it's necessary, we allocate the look-up table
	if(!roots){
		try{
			roots = new complex*[logn];
		}
		catch (const std::bad_alloc& e) {
			exit(1);
		}
		deallocate = true;
	}

    // If we need them, we compute the necessary roots of the unity
	if(!roots[logn-1]){
		try{
			roots[logn-1] = new (std::align_val_t(64)) complex[n/2]; 
		}
		catch (const std::bad_alloc& e) {
			exit(1);
		}
		for(int m = 0; m < n / 2; m++){
			roots[logn-1][m].re = + cos(2 * PI * m / (real) n);
			roots[logn-1][m].im = - sin(2 * PI * m / (real) n);
		}
	}

	ve = temp;
    vo = temp + n / 2;
    alignas(64) int baseIndex[] = {0,1,4,5,8,9,12,13};
	__m256i index = _mm256_load_si256((__m256i const*) baseIndex);

	// We build ve and vo using gather instructions and the array baseIndex
	for (int k = 0; k < n; k += 8) {
		temp0 = _mm512_i32gather_pd(index ,(double const*)&(wave[k]),   8);
		_mm512_store_pd((void*)&(ve[k/2]), temp0);
		temp0 = _mm512_i32gather_pd(index ,(double const*)&(wave[k+1]), 8);
		_mm512_store_pd((void*)&(vo[k/2]), temp0);
	}
	
	// Recursive call of the function on both ve and vo
	_mm512_FFT_CI_pd(ve, n / 2, wave, roots);
	_mm512_FFT_CI_pd(vo, n / 2, wave, roots);


	// We build the final result as a simple element wise product
	for(int i = 0; i < n / 2; i += 4){
		roVect = _mm512_load_pd((void*) &roots[logn-1][i]);
		voVect = _mm512_load_pd((void*) &vo[i]);               
														
		temp0  = _mm512_permute_pd(roVect, 0x00000000);
		temp1  = _mm512_permute_pd(roVect, 0b11111111);
		temp2  = _mm512_permute_pd(voVect, 0b01010101);
														
		temp0  = _mm512_mul_pd(temp0, voVect);                                                 
		temp1  = _mm512_mul_pd(temp1, temp2);              
														
		Z      = _mm512_fmaddsub_pd(temp0, _mm512_set1_pd(1), temp1);   

		temp0  = _mm512_load_pd((void*) &ve[i]);
		temp1  = _mm512_add_pd(temp0, Z);
		temp2  = _mm512_sub_pd(temp0, Z);                   

		_mm512_store_pd((void*)&wave[i], temp1);   
		_mm512_store_pd((void*)&wave[i + n / 2], temp2);
	}

	// If deallocate==true, we need to deallocate the data
	if(deallocate){
		for(int i=0; i<logn; i++) {
			if(roots[i]){
				delete [] roots[i];
            }
			roots[i] = nullptr;
		}
		delete [] roots;
	}
}

int main(int argc, char* argv[]) {

	if(argc!=2) exit(1);
	const int N = atoi(argv[1]);
	
	alignas(64)  complex wave[N];
	alignas(64)  complex scratch[N];

    for (int k = 0; k < N; k++) {
        wave[k].re = 0.125 * cos(2 * PI * k / (real) N) + 0.5 * cos(2 * PI * 10*k / (real) N);
        wave[k].im = 0.125 * sin(2 * PI * k / (real) N) + 0.5 * sin(2 * PI * 10*k / (real) N);
    }

    #ifdef __DEBUG 

        {
            Timer t;
            _mm512_FFT_CI_pd(wave, N, scratch); 
        }

    #else

        _mm512_FFT_CI_pd(wave, N, scratch);

        for(int i = 0; i < N; i++){
            std::cout << wave[i].re << '\t' << wave[i].im << '\n';
        }
    
    #endif
	
	return 0;
}



