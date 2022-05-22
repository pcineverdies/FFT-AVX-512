#include "utils/include.h"
#include "utils/complex.h"
#include "utils/timer.h"

inline void FFT(complex * wave, int n, complex * tmp, complex** roots = NULL) {
    
    assert((n & (n - 1)) == 0 && n > 0); 

    if(n == 1){
        return;
    }
        
    complex z, w, *vo, *ve;

    ve = tmp;
    vo = tmp + n/2;

    bool deallocate = false;

    int logn = 0;
    for(int x=n; x!=1; x>>=1){
        logn++;
    }

    if(roots == NULL){
        roots = new complex*[logn];
        deallocate = true;
    }

    if(roots[logn-1]==NULL){
        roots[logn-1] = new (std::align_val_t(64)) complex[n/2]; 
        for(int m = 0; m < n / 2; m++){
            roots[logn-1][m].re = + cos(2 * PI * m / (real) n);
            roots[logn-1][m].im = - sin(2 * PI * m / (real) n);
        }
    }

    for (int k=0; k < n/2; k++) {
        ve[k] = wave[2*k  ]; 
        vo[k] = wave[2*k+1];
    }
    
    FFT(ve, n / 2, wave, roots);
    FFT(vo, n / 2, wave, roots);   

    for (int m = 0; m < n / 2; m++) {
        w = roots[logn-1][m];
        
        z.re = w.re * vo[m].re - w.im * vo[m].im; 
        z.im = w.re * vo[m].im + w.im * vo[m].re; 
        
        wave[m].re = ve[m].re + z.re;
        wave[m].im = ve[m].im + z.im;
        
        wave[m + n / 2].re = ve[m].re - z.re;
        wave[m + n / 2].im = ve[m].im - z.im;
    }

    if(deallocate){
        for(int i=0; i<logn; i++){
            if(roots[i])
                delete [] roots[i];
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
            FFT(wave, N, scratch); 
        }

    #else

        FFT(wave, N, scratch);

        for(int i = 0; i < N; i++){
            std::cout << wave[i].re << '\t' << wave[i].im << '\n';
        }

    #endif

    return 0;
}