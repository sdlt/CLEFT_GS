/* ============================================================ *
 * CONTRIBUTORS:    Antoine Rocher, Michel-Andrès Breton,       *
 *                  Sylvain de la Torre, Martin Kärcher         *
 * 							                                    *
 * Code for computing Zel'dovich, CLPT and CLEFT predictions    *
 * based on [White (2014)], [Wang, Reid & White (2014)] and     *
 * [Vlah, Castorina & White 2016]                               *
 * ============================================================ */

#include <stdbool.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <gsl/gsl_sf_legendre.h>
#include <gsl/gsl_integration.h>
#include <gsl/gsl_errno.h>
#include <gsl/gsl_interp.h>
#include <gsl/gsl_spline.h>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_matrix.h>
#include <time.h>
#include <omp.h>

/************************************************************************************************************************************************************************/
/***********\\Global variable\\******************************************************************************************************************************************/

bool do_zeldovich = false;          // Compute Zel'dovich ingredients
bool do_CLPT = false;               // Compute CLPT ingredients
bool do_CLEFT = false;              // Compute CLEFT ingredients
const unsigned int r_max = 250;     // Maximum scale for CLEFT (see Code_RSD_CLPT.c)
const size_t size = 100;            // workspace size for cquad
const double xpivot_UYXi = 100;     // argument in Bessel function where we start to use approximate form (and qawo integration) for Xi_L, U and Y functions
const double xpivot_TV = 600;       // argument in Bessel function where we start to use approximate form (and qawo integration) for T and V functions
const unsigned int nbins_M0 = 2000; // Integration step for M0 function
const unsigned int nbins_M1 = 3200; // Integration step for M1 function
const unsigned int nbins_M2 = 2000; // Integration step for M2 function
double kmin, kmax;                  // min/max k in the power spectrum file used for interpolations
double kmin_integ, kmax_integ;      // min/max k in the power spectrum used for integrations
double q_v[3];                      // unit vector q_i, q_j, q_k
double EPS = 1e-8;                  // Precision buffer to avoid numerical aterfacts

struct my_f_params2 {
    double a;
    double b;
};
struct my_f_params3 {
    double a;
    double b;
    double c;
};
struct my_f_params4 {
    double a;
    double b;
    double c;
    double d;
};

gsl_interp_accel *acc[34];
gsl_spline *spline[34];

/************************************************************************************************************************************************************************/
/*******************\\ Gauss-Legendre integral quadrature\\***********************************************************************************************************/

static const double x[] = {
    1.56289844215430828714e-02, 4.68716824215916316162e-02,
    7.80685828134366366918e-02, 1.09189203580061115002e-01,
    1.40203137236113973212e-01, 1.71080080538603274883e-01,
    2.01789864095735997236e-01, 2.32302481844973969643e-01,
    2.62588120371503479163e-01, 2.92617188038471964730e-01,
    3.22360343900529151720e-01, 3.51788526372421720979e-01,
    3.80872981624629956772e-01, 4.09585291678301542532e-01,
    4.37897402172031513100e-01, 4.65781649773358042251e-01,
    4.93210789208190933576e-01, 5.20158019881763056670e-01,
    5.46597012065094167460e-01, 5.72501932621381191292e-01,
    5.97847470247178721259e-01, 6.22608860203707771585e-01,
    6.46761908514129279840e-01, 6.70283015603141015784e-01,
    6.93149199355801965946e-01, 7.15338117573056446485e-01,
    7.36828089802020705530e-01, 7.57598118519707176062e-01,
    7.77627909649495475605e-01, 7.96897892390314476375e-01,
    8.15389238339176254384e-01, 8.33083879888400823522e-01,
    8.49964527879591284320e-01, 8.66014688497164623416e-01,
    8.81218679385018415547e-01, 8.95561644970726986709e-01,
    9.09029570982529690453e-01, 9.21609298145333952679e-01,
    9.33288535043079545942e-01, 9.44055870136255977955e-01,
    9.53900782925491742847e-01, 9.62813654255815527284e-01,
    9.70785775763706331929e-01, 9.77809358486918288561e-01,
    9.83877540706057015509e-01, 9.88984395242991747997e-01,
    9.93124937037443459632e-01, 9.96295134733125149166e-01,
    9.98491950639595818382e-01, 9.99713726773441233703e-01};

static const double A[] = {
    3.12554234538633569472e-02, 3.12248842548493577326e-02,
    3.11638356962099067834e-02, 3.10723374275665165874e-02,
    3.09504788504909882337e-02, 3.07983790311525904274e-02,
    3.06161865839804484966e-02, 3.04040795264548200160e-02,
    3.01622651051691449196e-02, 2.98909795933328309169e-02,
    2.95904880599126425122e-02, 2.92610841106382766198e-02,
    2.89030896011252031353e-02, 2.85168543223950979908e-02,
    2.81027556591011733175e-02, 2.76611982207923882944e-02,
    2.71926134465768801373e-02, 2.66974591835709626611e-02,
    2.61762192395456763420e-02, 2.56294029102081160751e-02,
    2.50575444815795897034e-02, 2.44612027079570527207e-02,
    2.38409602659682059633e-02, 2.31974231852541216230e-02,
    2.25312202563362727021e-02, 2.18430024162473863146e-02,
    2.11334421125276415432e-02, 2.04032326462094327666e-02,
    1.96530874944353058650e-02, 1.88837396133749045537e-02,
    1.80959407221281166640e-02, 1.72904605683235824399e-02,
    1.64680861761452126430e-02, 1.56296210775460027242e-02,
    1.47758845274413017686e-02, 1.39077107037187726882e-02,
    1.30259478929715422855e-02, 1.21314576629794974079e-02,
    1.12251140231859771176e-02, 1.03078025748689695861e-02,
    9.38041965369445795116e-03, 8.44387146966897140266e-03,
    7.49907325546471157895e-03, 6.54694845084532276405e-03,
    5.58842800386551515727e-03, 4.62445006342211935096e-03,
    3.65596120132637518238e-03, 2.68392537155348241939e-03,
    1.70939265351810523958e-03, 7.34634490505671730396e-04};

#define NUM_OF_POSITIVE_ZEROS sizeof(x) / sizeof(double)
#define NUM_OF_ZEROS NUM_OF_POSITIVE_ZEROS + NUM_OF_POSITIVE_ZEROS

double gl_int(double a, double b, double (*f)(double, void *), void *prms) {
    double integral = 0.0;
    const double c = 0.5 * (b - a);
    const double d = 0.5 * (b + a);
    double dum;
    const double *px = &x[NUM_OF_POSITIVE_ZEROS - 1];
    const double *pA = &A[NUM_OF_POSITIVE_ZEROS - 1];
    for (; px >= x; pA--, px--) {
        dum = c * *px;
        integral += *pA * ((*f)(d - dum, prms) + (*f)(d + dum, prms));
    }
    return c * integral;
}

void gl_int6(double a, double b, void (*f)(double, double, double[]), void *prms, double result[], int n) {
    unsigned int i;
    double integral[n];
    for (i = 0; i < n; i++)
        integral[i] = 0;
    const double c = 0.5 * (b - a);
    const double d = 0.5 * (b + a);
    double dum;
    const double *px = &x[NUM_OF_POSITIVE_ZEROS - 1];
    const double *pA = &A[NUM_OF_POSITIVE_ZEROS - 1];
    double outi[n], outs[n];
    const double R = *(double *)prms;

    for (; px >= x; pA--, px--) {
        dum = c * *px;
        (*f)(d - dum, R, outi);
        (*f)(d + dum, R, outs);
        for (i = 0; i < n; i++)
            integral[i] += *pA * (outi[i] + outs[i]);
    }
    for (i = 0; i < n; i++)
        result[i] = c * integral[i];
}

/************************************************************************************************************************************************************************/
/****************************\\gsl_cquad integration \\*****************************************************************************************************************/

double int_cquad(double func(double, void *), double alpha) {
    double result, error;
    gsl_function F;
    F.function = func;
    F.params = &alpha;

    gsl_integration_cquad_workspace *w = gsl_integration_cquad_workspace_alloc(size);
    gsl_integration_cquad(&F, kmin_integ, kmax_integ, 0, 1e-9, w, &result, &error, NULL);
    gsl_integration_cquad_workspace_free(w);
    return result;
}

double int_cquad_pivot_UYXi(double func(double, void *), double alpha) {
    double result, error;
    const double q = alpha;
    gsl_function F;
    F.function = func;
    F.params = &alpha;

    gsl_integration_cquad_workspace *w = gsl_integration_cquad_workspace_alloc(size);
    gsl_integration_cquad(&F, kmin_integ, xpivot_UYXi / q, 0, 1e-9, w, &result, &error, NULL);
    gsl_integration_cquad_workspace_free(w);
    return result;
}

double int_cquad_pivot_TV(double func(double, void *), double alpha) {
    double result, error;
    const double q = alpha;
    gsl_function F;
    F.function = func;
    F.params = &alpha;

    gsl_integration_cquad_workspace *w = gsl_integration_cquad_workspace_alloc(size);
    gsl_integration_cquad(&F, kmin_integ, xpivot_TV / q, 0, 1e-9, w, &result, &error, NULL);
    gsl_integration_cquad_workspace_free(w);
    return result;
}

/************************************************************************************************************************************************************************/
/****************************\\Bessel function\\************************************************************************************************************************/

double func_bessel_0(double x) { return sin(x) / x; }
double func_bessel_1(double x) { return sin(x) / (x * x) - cos(x) / x; }
double func_bessel_2(double x) { return (3. / (x * x) - 1.) * sin(x) / x - 3. * cos(x) / (x * x); }
double func_bessel_3(double x) { return (15. / (x * x * x) - 6. / x) * sin(x) / x - (15. / (x * x) - 1.) * cos(x) / x; }

/************************************************************************************************************************************************************************/
/********************************\\ Interpolation Function \\***********************************************************************************************************/

double P_L(double k) {
    if (k >= kmin && k <= kmax)
        return gsl_spline_eval(spline[1], k, acc[1]);
    else
        return 0;
}

double R_1(double k) {
    if (k >= kmin && k <= kmax)
        return gsl_spline_eval(spline[2], k, acc[2]);
    else
        return 0;
}

double R_2(double k) {
    if (k >= kmin && k <= kmax)
        return gsl_spline_eval(spline[3], k, acc[3]);
    else
        return 0;
}

double Q_1(double k) {
    if (k >= kmin && k <= kmax)
        return gsl_spline_eval(spline[4], k, acc[4]);
    else
        return 0;
}

double Q_2(double k) {
    if (k >= kmin && k <= kmax)
        return gsl_spline_eval(spline[5], k, acc[5]);
    else
        return 0;
}

double Q_5(double k) {
    if (k >= kmin && k <= kmax)
        return gsl_spline_eval(spline[6], k, acc[6]);
    else
        return 0;
}

double Q_8(double k) {
    if (k >= kmin && k <= kmax)
        return gsl_spline_eval(spline[7], k, acc[7]);
    else
        return 0;
}

double iXi_L(double q) {
    if (q > 0 && q <= 2000)
        return gsl_spline_eval(spline[8], q, acc[8]);
    else
        return 0;
}

double iU_1(double q) {
    if (q > 0 && q <= 2000)
        return gsl_spline_eval(spline[9], q, acc[9]);
    else
        return 0;
}

double iU_3(double q) {
    if (q > 0 && q <= 2000)
        return gsl_spline_eval(spline[10], q, acc[10]);
    else
        return 0;
}

double iU_11(double q) {
    if (q > 0 && q <= 2000)
        return gsl_spline_eval(spline[11], q, acc[11]);
    else
        return 0;
}

double iU_20(double q) {
    if (q > 0 && q <= 2000)
        return gsl_spline_eval(spline[12], q, acc[12]);
    else
        return 0;
}

double iX_11(double q) {
    if (q > 0 && q <= 2000)
        return gsl_spline_eval(spline[13], q, acc[13]);
    else
        return 0;
}

double iX_13(double q) {
    if (q > 0 && q <= 2000)
        return gsl_spline_eval(spline[14], q, acc[14]);
    else
        return 0;
}

double iX_22(double q) {
    if (q > 0 && q <= 2000)
        return gsl_spline_eval(spline[15], q, acc[15]);
    else
        return 0;
}

double iX_10_12(double q) {
    if (q > 0 && q <= 2000)
        return gsl_spline_eval(spline[16], q, acc[16]);
    else
        return 0;
}

double iY_11(double q) {
    if (q > 0 && q <= 2000)
        return gsl_spline_eval(spline[17], q, acc[17]);
    else
        return 0;
}

double iY_13(double q) {
    if (q > 0 && q <= 2000)
        return gsl_spline_eval(spline[18], q, acc[18]);
    else
        return 0;
}

double iY_22(double q) {
    if (q > 0 && q <= 2000)
        return gsl_spline_eval(spline[19], q, acc[19]);
    else
        return 0;
}

double iY_10_12(double q) {
    if (q > 0 && q <= 2000)
        return gsl_spline_eval(spline[20], q, acc[20]);
    else
        return 0;
}

double iW_V1(double q) {
    if (q > 0 && q <= 2000)
        return gsl_spline_eval(spline[21], q, acc[21]);
    else
        return 0;
}

double iW_V3(double q) {
    if (q > 0 && q <= 2000)
        return gsl_spline_eval(spline[22], q, acc[22]);
    else
        return 0;
}

double iW_T(double q) {
    if (q > 0 && q <= 2000)
        return gsl_spline_eval(spline[23], q, acc[23]);
    else
        return 0;
}

double Q_s(double k) {
    if (k >= kmin && k <= kmax)
        return gsl_spline_eval(spline[24], k, acc[24]);
    else
        return 0;
}

double iV_10(double q) {
    if (q > 0 && q <= 2000)
        return gsl_spline_eval(spline[25], q, acc[25]);
    else
        return 0;
}

double iB_1(double q) {
    if (q > 0 && q <= 2000)
        return gsl_spline_eval(spline[26], q, acc[26]);
    else
        return 0;
}

double iB_2(double q) {
    if (q > 0 && q <= 2000)
        return gsl_spline_eval(spline[27], q, acc[27]);
    else
        return 0;
}

double iJ_2(double q) {
    if (q > 0 && q <= 2000)
        return gsl_spline_eval(spline[28], q, acc[28]);
    else
        return 0;
}

double iJ_3(double q) {
    if (q > 0 && q <= 2000)
        return gsl_spline_eval(spline[29], q, acc[29]);
    else
        return 0;
}

double iJ_4(double q) {
    if (q > 0 && q <= 2000)
        return gsl_spline_eval(spline[30], q, acc[30]);
    else
        return 0;
}

double iV_12(double q) {
    if (q > 0 && q <= 2000)
        return gsl_spline_eval(spline[31], q, acc[31]);
    else
        return 0;
}

double iChi_12(double q) {
    if (q > 0 && q <= 2000)
        return gsl_spline_eval(spline[32], q, acc[32]);
    else
        return 0;
}

double iZeta(double q) {
    if (q > 0 && q <= 2000)
        return gsl_spline_eval(spline[33], q, acc[33]);
    else
        return 0;
}

/************************************************************************************************************************************************************************/
/*****************\\delta kronecker\\***********************************************************************************************************************************/

int delta_K(int i, int j) {
    if (i == j)
        return 1;
    else
        return 0;
}

/************************************************************************************************************************************************************************/
/****************************************\\ R_n FUNCTIONS \\*********************************************************************************************************/

// R_1_tilde function
double fRt1(double x, void *p) {
    const double r = *(double *)p;
    return (r * r * (1. - x * x) * (1. - x * x)) / (1. + r * r - 2. * x * r);
}

double Rt_1(double r) {
    return gl_int(-1, 1, &fRt1, &r);
}

// R_2_tilde function
double fRt2(double x, void *p) {
    const double r = *(double *)p;
    return ((1. - x * x) * r * x * (1. - x * r)) / (1. + r * r - 2. * r * x);
}

double Rt_2(double r) {
    return gl_int(-1, 1, &fRt2, &r);
}

// R1 function
double fR_1(double r, void *p) {
    const double k = *(double *)p;
    return P_L(k * r) * Rt_1(r);
}

double R1(double k) {
    double error;
    double result1 = 0;
    double result2 = 0;
    gsl_function F;
    gsl_integration_workspace *w = gsl_integration_workspace_alloc(200);

    F.function = &fR_1;
    F.params = &k;

    gsl_integration_qag(&F, 0, 1., 0, 1e-3, 200, 6, w, &result1, &error);
    if (kmax_integ / k > 1)
        gsl_integration_qag(&F, 1., kmax_integ / k, 0, 1e-3, 200, 6, w, &result2, &error);
    gsl_integration_workspace_free(w);

    return (k * k * k) / (4. * M_PI * M_PI) * P_L(k) * (result1 + result2);
}

// R2 function
double fR_2(double r, void *p) {
    const double k = *(double *)p;
    return P_L(k * r) * Rt_2(r);
}

double R2(double k) {
    double error;
    double result1 = 0;
    double result2 = 0;
    gsl_function F;
    F.function = &fR_2;
    F.params = &k;

    gsl_integration_workspace *w = gsl_integration_workspace_alloc(200);
    gsl_integration_qag(&F, 0, 1, 0, 1e-3, 200, 6, w, &result1, &error);
    if (kmax_integ / k > 1)
        gsl_integration_qag(&F, 1, kmax_integ / k, 0, 1e-3, 200, 6, w, &result2, &error);
    gsl_integration_workspace_free(w);
    return (k * k * k) / (4. * M_PI * M_PI) * P_L(k) * (result1 + result2);
}

void compute_and_interpolate_R1(void) {
    printf("getting R function 1/2...\n");
    const int nk = 11000; // Number of points
    const double log_kmin = log10(kmin);
    const double log_kmax = log10(kmax);
    const double dk = (log_kmax - log_kmin) / nk;

    double R1_array[nk + 1];
    double k_array[nk + 1];
    for (int i = 0; i <= nk; i++) {
        const double k = pow(10, log_kmin + i * dk);
        R1_array[i] = R1(k);
        k_array[i] = k;
    }

    acc[2] = gsl_interp_accel_alloc();
    spline[2] = gsl_spline_alloc(gsl_interp_cspline, nk + 1);
    gsl_spline_init(spline[2], k_array, R1_array, nk + 1);
    return;
}

void compute_and_interpolate_R2(void) {
    printf("getting R function 2/2...\n");
    const unsigned int nk = 11000; // Number of points
    const double log_kmin = log10(kmin);
    const double log_kmax = log10(kmax);
    const double dk = (log_kmax - log_kmin) / nk;

    double R2_array[nk + 1];
    double k_array[nk + 1];
    for (int i = 0; i <= nk; i++) {
        const double k = pow(10, log_kmin + i * dk);
        R2_array[i] = R2(k);
        k_array[i] = k;
    }

    acc[3] = gsl_interp_accel_alloc();
    spline[3] = gsl_spline_alloc(gsl_interp_cspline, nk + 1);
    gsl_spline_init(spline[3], k_array, R2_array, nk + 1);
    return;
}

/***********************************************************************************************************************************************************************/
/******************************\\FUNCTION Q\\***********************************************************************************************************************/

double Qt_1(double r, double x) {
    return (r * r * (1. - x * x) * (1. - x * x)) / ((1. + r * r - 2. * x * r) * (1. + r * r - 2. * x * r));
}

double Qt_2(double r, double x) {
    return ((1. - x * x) * r * x * (1. - x * r)) / ((1. + r * r - 2. * r * x) * (1. + r * r - 2. * r * x));
}

double Qt_5(double r, double x) {
    return (r * x * (1. - x * x)) / (1. + r * r - 2. * x * r);
}

double Qt_8(double r, double x) {
    return (r * r * (1. - x * x)) / (1. + r * r - 2. * x * r);
}

double Qt_s(double r, double x) {
    return r * r * (x * x - 1.) * (1. - 2 * r * r + 4 * r * x - 3 * x * x) / ((1. + r * r - 2. * x * r) * (1. + r * r - 2. * x * r));
}

double lg_Q_n(double x, void *p) {
    struct my_f_params3 *params = (struct my_f_params3 *)p;
    const double k = (params->a);
    const double r = (params->b);
    const double n = (params->c);
    const double y_p = 1. + r * r - 2. * x * r;
    if (y_p < 0)
        return 0;
    else {
        if ((int)n == 1)
            return P_L(k * sqrt(y_p)) * Qt_1(r, x);
        if ((int)n == 2)
            return P_L(k * sqrt(y_p)) * Qt_2(r, x);
        if ((int)n == 5)
            return P_L(k * sqrt(y_p)) * Qt_5(r, x);
        if ((int)n == 8)
            return P_L(k * sqrt(y_p)) * Qt_8(r, x);
        if ((int)n == 9)
            return P_L(k * sqrt(y_p)) * Qt_s(r, x);
        else
            return -1;
    }
}

double fQ_n(double r, void *p) {
    struct my_f_params2 *params = (struct my_f_params2 *)p;
    const double k = (params->a);
    const double n = (params->b);
    struct my_f_params3 par = {k, r, n};
    return P_L(k * r) * gl_int(-1, 1, &lg_Q_n, &par);
}

double Q_n(int n, double k) {
    struct my_f_params2 p = {k, n};
    double error;
    double result1 = 0;
    double result2 = 0;
    gsl_function F;
    gsl_integration_workspace *w = gsl_integration_workspace_alloc(200);

    F.function = &fQ_n;
    F.params = &p;

    if (n == 9)
        gsl_integration_qag(&F, 0., 1., 0, 5e-2, 200, 6, w, &result1, &error);
    else
        gsl_integration_qag(&F, 0., 1., 0, 1e-3, 200, 6, w, &result1, &error);

    if (kmax_integ / k > 1)
        gsl_integration_qag(&F, 1., kmax_integ / k, 0, 5e-2, 200, 6, w, &result2, &error);

    gsl_integration_workspace_free(w);

    return (k * k * k) / (4. * M_PI * M_PI) * (result1 + result2);
}

void compute_and_interpolate_Qn(void) {
    unsigned int n;
    const unsigned int nk = 3000; // Number of points
    const double log_kmin = log10(kmin);
    const double log_kmax = log10(kmax);
    const double dk = (log_kmax - log_kmin) / nk;
    double Q1_array[nk + 1], Q2_array[nk + 1], Q5_array[nk + 1], Q8_array[nk + 1], Qs_array[nk + 1], k_array[nk + 1];
    for (n = 1; n < 10; n++) {
        printf("getting Q function %d/9...\n", n);
        switch (n) {
        case 1:
            for (int i = 0; i <= nk; i++) {
                const double k = pow(10, log_kmin + i * dk);
                Q1_array[i] = Q_n(1, k);
                k_array[i] = k;
            }
            acc[4] = gsl_interp_accel_alloc();
            spline[4] = gsl_spline_alloc(gsl_interp_cspline, nk + 1);
            gsl_spline_init(spline[4], k_array, Q1_array, nk + 1);
            break;
        case 2:
            for (int i = 0; i <= nk; i++) {
                const double k = pow(10, log_kmin + i * dk);
                Q2_array[i] = Q_n(2, k);
                k_array[i] = k;
            }
            acc[5] = gsl_interp_accel_alloc();
            spline[5] = gsl_spline_alloc(gsl_interp_cspline, nk + 1);
            gsl_spline_init(spline[5], k_array, Q2_array, nk + 1);
            break;
        case 5:
            for (int i = 0; i <= nk; i++) {
                const double k = pow(10, log_kmin + i * dk);
                Q5_array[i] = Q_n(5, k);
                k_array[i] = k;
            }
            acc[6] = gsl_interp_accel_alloc();
            spline[6] = gsl_spline_alloc(gsl_interp_cspline, nk + 1);
            gsl_spline_init(spline[6], k_array, Q5_array, nk + 1);
            break;
        case 8:
            for (int i = 0; i <= nk; i++) {
                const double k = pow(10, log_kmin + i * dk);
                Q8_array[i] = Q_n(8, k);
                k_array[i] = k;
            }
            acc[7] = gsl_interp_accel_alloc();
            spline[7] = gsl_spline_alloc(gsl_interp_cspline, nk + 1);
            gsl_spline_init(spline[7], k_array, Q8_array, nk + 1);
            break;
        case 9:
            for (int i = 0; i <= nk; i++) {
                const double k = pow(10, log_kmin + i * dk);
                Qs_array[i] = Q_n(9, k);
                k_array[i] = k;
            }
            acc[24] = gsl_interp_accel_alloc();
            spline[24] = gsl_spline_alloc(gsl_interp_cspline, nk + 1);
            gsl_spline_init(spline[24], k_array, Qs_array, nk + 1);
            break;
        default:
            break;
        }
    }
    return;
}

/*******************************************************************************************************************************************************************/
/************************\\Compute Xi linear\\********************************************************************************************************************/

double fXi_L(double k, void *p) {
    const double q = *(double *)p;
    return P_L(k) * k * k * func_bessel_0(k * q);
}

double fXi_L_qawo(double k, void *p) {
    const double alpha = *(double *)p;
    return alpha * P_L(k) * k;
}

double Xi_L(double q) {
    if (kmax_integ * q < xpivot_UYXi)
        return 1. / (2. * M_PI * M_PI) * int_cquad(fXi_L, q);
    else {
        double result1, result2, error;
        // First integration using cquad
        result1 = 1. / (2. * M_PI * M_PI) * int_cquad_pivot_UYXi(fXi_L, q);

        // Second integration with qawo
        double alpha = 1. / (2. * M_PI * M_PI * q);
        gsl_function F;
        F.function = &fXi_L_qawo;
        F.params = &alpha;

        gsl_integration_workspace *w1 = gsl_integration_workspace_alloc(200);
        gsl_integration_workspace *w2 = gsl_integration_workspace_alloc(200);
        gsl_integration_qawo_table *t = gsl_integration_qawo_table_alloc(q, 0, GSL_INTEG_SINE, size);
        gsl_integration_qawf(&F, xpivot_UYXi / q, 1e-5, 200, w1, w2, t, &result2, &error);

        gsl_integration_workspace_free(w1);
        gsl_integration_workspace_free(w2);
        gsl_integration_qawo_table_free(t);
        return result1 + result2;
    }
}

/*******************************************************************************************************************************************************************/
/**************\\U function\\**************************************************************************************************************************************/
// U^1(q)
double fU_1(double k, void *p) {
    const double q = *(double *)p;
    return -P_L(k) * k * func_bessel_1(k * q);
}

double fU_1_qawo(double k, void *p) {
    const double alpha = *(double *)p;
    return alpha * P_L(k);
}

double U_1(double q) {
    if (kmax_integ * q < xpivot_UYXi)
        return 1. / (2. * M_PI * M_PI) * int_cquad(fU_1, q);
    else {
        double result1, result2, error;
        // First integration using cquad
        result1 = 1. / (2. * M_PI * M_PI) * int_cquad_pivot_UYXi(fU_1, q);

        // Second integration with qawo
        double alpha = 1. / (2. * M_PI * M_PI * q);
        gsl_function F;
        F.function = &fU_1_qawo;
        F.params = &alpha;

        gsl_integration_workspace *w1 = gsl_integration_workspace_alloc(200);
        gsl_integration_workspace *w2 = gsl_integration_workspace_alloc(200);
        gsl_integration_qawo_table *t = gsl_integration_qawo_table_alloc(q, 0, GSL_INTEG_COSINE, size);
        gsl_integration_qawf(&F, xpivot_UYXi / q, 1e-4, 200, w1, w2, t, &result2, &error);

        gsl_integration_workspace_free(w1);
        gsl_integration_workspace_free(w2);
        gsl_integration_qawo_table_free(t);
        return result1 + result2;
    }
}

// U^3(q)
double fU_3(double k, void *p) {
    const double q = *(double *)p;
    return -5. / 21. * R_1(k) * k * func_bessel_1(k * q);
}

double fU_3_qawo(double k, void *p) {
    const double alpha = *(double *)p;
    return +5. / 21. * alpha * R_1(k);
}

double U_3(double q) {
    if (kmax_integ * q < xpivot_UYXi)
        return 1. / (2. * M_PI * M_PI) * int_cquad(fU_3, q);
    else {
        double result1, result2, error;
        // First integration using cquad
        result1 = 1. / (2. * M_PI * M_PI) * int_cquad_pivot_UYXi(fU_3, q);

        // Second integration with qawo
        double alpha = 1. / (2. * M_PI * M_PI * q);
        gsl_function F;
        F.function = &fU_3_qawo;
        F.params = &alpha;

        gsl_integration_workspace *w1 = gsl_integration_workspace_alloc(200);
        gsl_integration_workspace *w2 = gsl_integration_workspace_alloc(200);
        gsl_integration_qawo_table *t = gsl_integration_qawo_table_alloc(q, 0, GSL_INTEG_COSINE, size);
        gsl_integration_qawf(&F, xpivot_UYXi / q, 1e-4, 200, w1, w2, t, &result2, &error);

        gsl_integration_workspace_free(w1);
        gsl_integration_workspace_free(w2);
        gsl_integration_qawo_table_free(t);
        return result1 + result2;
    }
}

// U_11(q)
double fU_11(double k, void *p) {
    const double q = *(double *)p;
    return -6. / 7. * k * (R_1(k) + R_2(k)) * func_bessel_1(k * q);
}

double fU_11_qawo(double k, void *p) {
    const double alpha = *(double *)p;
    return 6. / 7. * alpha * (R_1(k) + R_2(k));
}

double U_11(double q) {
    if (kmax_integ * q < xpivot_UYXi)
        return 1. / (2. * M_PI * M_PI) * int_cquad(fU_11, q);
    else {
        double result1, result2, error;
        // First integration using cquad
        result1 = 1. / (2. * M_PI * M_PI) * int_cquad_pivot_UYXi(fU_11, q);

        // Second integration with qawo
        double alpha = 1. / (2. * M_PI * M_PI * q);
        gsl_function F;
        F.function = &fU_11_qawo;
        F.params = &alpha;

        gsl_integration_workspace *w1 = gsl_integration_workspace_alloc(200);
        gsl_integration_workspace *w2 = gsl_integration_workspace_alloc(200);
        gsl_integration_qawo_table *t = gsl_integration_qawo_table_alloc(q, 0, GSL_INTEG_COSINE, size);
        gsl_integration_qawf(&F, xpivot_UYXi / q, 1e-4, 200, w1, w2, t, &result2, &error);

        gsl_integration_workspace_free(w1);
        gsl_integration_workspace_free(w2);
        gsl_integration_qawo_table_free(t);
        return result1 + result2;
    }
}

// U_20(q)
double fU_20(double k, void *p) {
    const double q = *(double *)p;
    return -3. / 7. * k * Q_8(k) * func_bessel_1(k * q);
}

double fU_20_qawo(double k, void *p) {
    const double alpha = *(double *)p;
    return 3. / 7. * alpha * Q_8(k);
}

double U_20(double q) {
    if (kmax_integ * q < xpivot_UYXi) {
        return 1. / (2. * M_PI * M_PI) * int_cquad(fU_20, q);
    } else {
        double result1, result2, error;
        // First integration using cquad
        result1 = 1. / (2. * M_PI * M_PI) * int_cquad_pivot_UYXi(fU_20, q);

        // Second integration with qawo
        double alpha = 1. / (2. * M_PI * M_PI * q);
        gsl_function F;
        F.function = &fU_20_qawo;
        F.params = &alpha;

        gsl_integration_workspace *w1 = gsl_integration_workspace_alloc(200);
        gsl_integration_workspace *w2 = gsl_integration_workspace_alloc(200);
        gsl_integration_qawo_table *t = gsl_integration_qawo_table_alloc(q, 0, GSL_INTEG_COSINE, size);
        gsl_integration_qawf(&F, xpivot_UYXi / q, 1e-4, 200, w1, w2, t, &result2, &error);

        gsl_integration_workspace_free(w1);
        gsl_integration_workspace_free(w2);
        gsl_integration_qawo_table_free(t);
        return result1 + result2;
    }
}

// Ui(q)
double get_U(double q) {
    return iU_1(q) + iU_3(q);
}

/*******************************************************************************************************************************************************************/
/*******************\\X FUNCTIONS\\*****************************************************************************************************************************/
// X^11(q)
double fX_11(double k, void *p) {
    const double q = *(double *)p;
    const double x = k * q;
    return P_L(k) * (2. / 3. - 2. * func_bessel_1(x) / x);
}

double X_11(double q) {
    return 1. / (2. * M_PI * M_PI) * int_cquad(fX_11, q);
}

// X^22(q)
double fX_22(double k, void *p) {
    const double q = *(double *)p;
    const double x = k * q;
    return 9. / 98. * Q_1(k) * (2. / 3. - 2. * func_bessel_1(x) / (x));
}

double X_22(double q) {
    return 1. / (2. * M_PI * M_PI) * int_cquad(fX_22, q);
}

// X^13(q)
double fX_13(double k, void *p) {
    const double q = *(double *)p;
    const double x = k * q;
    return 5. / 21. * R_1(k) * (2. / 3. - 2. * func_bessel_1(x) / x);
}

double X_13(double q) {
    return 1. / (2. * M_PI * M_PI) * int_cquad(fX_13, q);
}

// X_10^12(q)
double fX_10_12(double k, void *p) {
    const double q = *(double *)p;
    const double x = k * q;
    return 1. / 14. * (2. * (R_1(k) - R_2(k)) + 3. * R_1(k) * sin(k * q) / (k * q) - 3. * (3. * R_1(k) + 4. * R_2(k) + 2. * Q_5(k)) * func_bessel_1(x) / x);
}

double X_10_12(double q) {
    return 1. / (2. * M_PI * M_PI) * int_cquad(fX_10_12, q);
}

/*******************************************************************************************************************************************************************/
/*******************Y FUNCTIONS*****************************************************************************************************************************/
// Y^11(q)
double fY_11(double k, void *p) {
    const double q = *(double *)p;
    const double x = k * q;
    return P_L(k) * (-2. * sin(k * q) / (k * q) + 6. * func_bessel_1(x) / x);
}

double fY_11_qawo_1_over_2(double k, void *p) {
    const double alpha = *(double *)p;
    return -2 * alpha * P_L(k) / k;
}

double fY_11_qawo_2_over_2(double k, void *p) {
    const double alpha = *(double *)p;
    return -alpha * P_L(k) * 6 / (k * k);
}

double Y_11(double q) {
    if (kmax_integ * q < xpivot_UYXi)
        return 1. / (2. * M_PI * M_PI) * int_cquad(fY_11, q);
    else {
        double result1, result2, result3, error;
        // First integration using cquad
        result1 = 1. / (2. * M_PI * M_PI) * int_cquad_pivot_UYXi(fY_11, q);

        // Second integration with qawo 1/2
        double alpha = 1. / (2. * M_PI * M_PI * q);
        gsl_function F;
        F.function = &fY_11_qawo_1_over_2;
        F.params = &alpha;

        gsl_integration_workspace *w1 = gsl_integration_workspace_alloc(200);
        gsl_integration_workspace *w2 = gsl_integration_workspace_alloc(200);
        gsl_integration_qawo_table *t = gsl_integration_qawo_table_alloc(q, 0, GSL_INTEG_SINE, size);
        gsl_integration_qawf(&F, xpivot_UYXi / q, 1e-5, 200, w1, w2, t, &result2, &error);

        // Second integration with qawo 2/2
        alpha = 1. / (2. * M_PI * M_PI * q * q);
        F.function = &fY_11_qawo_2_over_2;
        F.params = &alpha;
        gsl_integration_qawo_table *t2 = gsl_integration_qawo_table_alloc(q, 0, GSL_INTEG_COSINE, size);
        gsl_integration_qawf(&F, xpivot_UYXi / q, 1e-5, 200, w1, w2, t2, &result3, &error);

        gsl_integration_workspace_free(w1);
        gsl_integration_workspace_free(w2);
        gsl_integration_qawo_table_free(t);
        gsl_integration_qawo_table_free(t2);
        return result1 + result2 + result3;
    }
}

// Y^22(q)
double fY_22(double k, void *p) {
    const double q = *(double *)p;
    const double x = k * q;
    return 9. / 98. * Q_1(k) * (-2. * sin(k * q) / (k * q) + 6. * func_bessel_1(x) / x);
}

double fY_22_qawo_1_over_2(double k, void *p) {
    const double alpha = *(double *)p;
    return 9. / 98. * alpha * Q_1(k) * (-2.) / k;
}

double fY_22_qawo_2_over_2(double k, void *p) {
    const double alpha = *(double *)p;
    return -9. / 98. * alpha * Q_1(k) * 6. / (k * k);
}

double Y_22(double q) {
    if (kmax_integ * q < xpivot_UYXi)
        return 1. / (2. * M_PI * M_PI) * int_cquad(fY_22, q);
    else {
        double result1, result2, result3, error;
        // First integration using cquad
        result1 = 1. / (2. * M_PI * M_PI) * int_cquad_pivot_UYXi(fY_22, q);

        // Second integration with qawo 1/2
        double alpha = 1. / (2. * M_PI * M_PI * q);
        gsl_function F;
        F.function = &fY_22_qawo_1_over_2;
        F.params = &alpha;

        gsl_integration_workspace *w1 = gsl_integration_workspace_alloc(200);
        gsl_integration_workspace *w2 = gsl_integration_workspace_alloc(200);
        gsl_integration_qawo_table *t = gsl_integration_qawo_table_alloc(q, 0, GSL_INTEG_SINE, size);
        gsl_integration_qawf(&F, xpivot_UYXi / q, 1e-5, 200, w1, w2, t, &result2, &error);

        // Second integration with qawo 2/2
        alpha = 1. / (2. * M_PI * M_PI * q * q);
        F.function = &fY_22_qawo_2_over_2;
        F.params = &alpha;
        gsl_integration_qawo_table *t2 = gsl_integration_qawo_table_alloc(q, 0, GSL_INTEG_COSINE, size);
        gsl_integration_qawf(&F, xpivot_UYXi / q, 1e-5, 200, w1, w2, t2, &result3, &error);

        gsl_integration_workspace_free(w1);
        gsl_integration_workspace_free(w2);
        gsl_integration_qawo_table_free(t);
        gsl_integration_qawo_table_free(t2);
        return result1 + result2 + result3;
    }
}

// Y^13(q)
double fY_13(double k, void *p) {
    const double q = *(double *)p;
    const double x = k * q;
    return 5. / 21. * R_1(k) * (-2. * sin(k * q) / (k * q) + 6. * func_bessel_1(x) / x);
}

double fY_13_qawo_1_over_2(double k, void *p) {
    const double alpha = *(double *)p;
    return 5. / 21. * alpha * R_1(k) * (-2.) / k;
}

double fY_13_qawo_2_over_2(double k, void *p) {
    const double alpha = *(double *)p;
    return -5. / 21. * alpha * R_1(k) * 6. / (k * k);
}

double Y_13(double q) {
    if (kmax_integ * q < xpivot_UYXi)
        return 1. / (2. * M_PI * M_PI) * int_cquad(fY_13, q);
    else {
        double result1, result2, result3, error;
        // First integration using cquad
        result1 = 1. / (2. * M_PI * M_PI) * int_cquad_pivot_UYXi(fY_13, q);

        // Second integration with qawo 1/2
        double alpha = 1. / (2. * M_PI * M_PI * q);
        gsl_function F;
        F.function = &fY_13_qawo_1_over_2;
        F.params = &alpha;

        gsl_integration_workspace *w1 = gsl_integration_workspace_alloc(200);
        gsl_integration_workspace *w2 = gsl_integration_workspace_alloc(200);
        gsl_integration_qawo_table *t = gsl_integration_qawo_table_alloc(q, 0, GSL_INTEG_SINE, size);
        gsl_integration_qawf(&F, xpivot_UYXi / q, 1e-5, 200, w1, w2, t, &result2, &error);

        // Second integration with qawo 2/2
        alpha = 1. / (2. * M_PI * M_PI * q * q);
        F.function = &fY_13_qawo_2_over_2;
        F.params = &alpha;
        gsl_integration_qawo_table *t2 = gsl_integration_qawo_table_alloc(q, 0, GSL_INTEG_COSINE, size);
        gsl_integration_qawf(&F, xpivot_UYXi / q, 1e-5, 200, w1, w2, t2, &result3, &error);

        gsl_integration_workspace_free(w1);
        gsl_integration_workspace_free(w2);
        gsl_integration_qawo_table_free(t);
        gsl_integration_qawo_table_free(t2);
        return result1 + result2 + result3;
    }
}

// Y^10_12(q)
double fY_10_12(double k, void *p) {
    const double q = *(double *)p;
    const double x = q * k;
    return -3. / 14. * (3. * R_1(k) + 4. * R_2(k) + 2. * Q_5(k)) * (sin(k * q) / (k * q) - 3. * func_bessel_1(x) / x);
}

double fY_10_12_qawo_1_over_2(double k, void *p) {
    const double alpha = *(double *)p;
    return -3. / 14. * alpha * (3. * R_1(k) + 4. * R_2(k) + 2. * Q_5(k)) / k;
}

double fY_10_12_qawo_2_over_2(double k, void *p) {
    const double alpha = *(double *)p;
    return 3. / 14. * alpha * (3. * R_1(k) + 4. * R_2(k) + 2. * Q_5(k)) * (-3.) / (k * k);
}

double Y_10_12(double q) {
    if (kmax_integ * q < xpivot_UYXi)
        return 1. / (2. * M_PI * M_PI) * int_cquad(fY_10_12, q);
    else {
        double result1, result2, result3, error;
        // First integration using cquad
        result1 = 1. / (2. * M_PI * M_PI) * int_cquad_pivot_UYXi(fY_10_12, q);

        // Second integration with qawo 1/2Y_10_12Y_10_12
        double alpha = 1. / (2. * M_PI * M_PI * q);
        gsl_function F;
        F.function = &fY_10_12_qawo_1_over_2;
        F.params = &alpha;

        gsl_integration_workspace *w1 = gsl_integration_workspace_alloc(200);
        gsl_integration_workspace *w2 = gsl_integration_workspace_alloc(200);
        gsl_integration_qawo_table *t = gsl_integration_qawo_table_alloc(q, 0, GSL_INTEG_SINE, size);
        gsl_integration_qawf(&F, xpivot_UYXi / q, 2e-5, 200, w1, w2, t, &result2, &error);

        // Second integration with qawo 2/2
        alpha = 1. / (2. * M_PI * M_PI * q * q);
        F.function = &fY_10_12_qawo_2_over_2;
        F.params = &alpha;
        gsl_integration_qawo_table *t2 = gsl_integration_qawo_table_alloc(q, 0, GSL_INTEG_COSINE, size);
        gsl_integration_qawf(&F, xpivot_UYXi / q, 1e-5, 200, w1, w2, t2, &result3, &error);

        gsl_integration_workspace_free(w1);
        gsl_integration_workspace_free(w2);
        gsl_integration_qawo_table_free(t);
        gsl_integration_qawo_table_free(t2);
        return result1 + result2 + result3;
    }
}

/************************************************************************************************************************************************************************/
/**************************\\V+S et T function\\************************************************************************************************************************/
// V^112_1(q) + S^112(q)

double fV1_112(double k, void *p) {
    const double q = *(double *)p;
    const double x = k * q;
    return -3. / 7. * R_1(k) * func_bessel_1(x) / k + 3. / (7. * k) * (2. * R_1(k) + 4. * R_2(k) + Q_1(k) + 2. * Q_2(k)) * func_bessel_2(x) / x;
}

double fV1_112_qawo_1_over_2(double k, void *p) {
    const double alpha = *(double *)p;
    return 3. / 7. * alpha * R_1(k) / (k * k);
}

double fV1_112_qawo_2_over_2(double k, void *p) {
    const double alpha = *(double *)p;
    return -3. / (7. * k) * alpha * (2. * R_1(k) + 4. * R_2(k) + Q_1(k) + 2. * Q_2(k)) / (k * k);
}

double V1_112(double q) {
    if (kmin_integ * q < EPS)
        return 0;
    else if (kmax_integ * q < xpivot_TV) {
        return 1. / (2. * M_PI * M_PI) * int_cquad(fV1_112, q);
    } else {
        double result1, result2, result3, error;
        // First integration using cquad
        result1 = 1. / (2. * M_PI * M_PI) * int_cquad_pivot_TV(fV1_112, q);

        // Second integration with qawo 1/2
        double alpha = 1. / (2. * M_PI * M_PI * q * q);
        gsl_function F;
        F.function = &fV1_112_qawo_1_over_2;
        F.params = &alpha;

        gsl_integration_workspace *w1 = gsl_integration_workspace_alloc(200);
        gsl_integration_workspace *w2 = gsl_integration_workspace_alloc(200);
        gsl_integration_qawo_table *t = gsl_integration_qawo_table_alloc(q, 0, GSL_INTEG_COSINE, size);
        gsl_integration_qawf(&F, xpivot_TV / q, 1e-6, 200, w1, w2, t, &result2, &error);

        // Second integration with qawo 2/2
        F.function = &fV1_112_qawo_2_over_2;
        F.params = &alpha;
        gsl_integration_qawo_table *t2 = gsl_integration_qawo_table_alloc(q, 0, GSL_INTEG_SINE, size);
        gsl_integration_qawf(&F, xpivot_TV / q, 1e-6, 200, w1, w2, t2, &result3, &error);

        gsl_integration_workspace_free(w1);
        gsl_integration_workspace_free(w2);
        gsl_integration_qawo_table_free(t);
        gsl_integration_qawo_table_free(t2);
        return result1 + result2 + result3;
    }
}

// V^112_3(q)+ S^112	(q)

double fV3_112(double k, void *p) {
    const double q = *(double *)p;
    const double x = k * q;
    return -3. / 7. * Q_1(k) * func_bessel_1(x) / k + 3. / (7. * k) * (2. * R_1(k) + 4. * R_2(k) + Q_1(k) + 2. * Q_2(k)) * func_bessel_2(x) / x;
}

double fV3_112_qawo_1_over_2(double k, void *p) {
    const double alpha = *(double *)p;
    return 3. / 7. * alpha * Q_1(k) / (k * k);
}

double fV3_112_qawo_2_over_2(double k, void *p) {
    const double alpha = *(double *)p;
    return -3. / (7. * k) * alpha * (2. * R_1(k) + 4. * R_2(k) + Q_1(k) + 2. * Q_2(k)) / (k * k);
}

// V^112_3+S^112

double V3_112(double q) {
    if (kmin_integ * q < EPS)
        return 0;
    else if (kmax_integ * q < xpivot_TV) {
        return 1. / (2. * M_PI * M_PI) * int_cquad(fV3_112, q);
    } else {
        double result1, result2, result3, error;
        // First integration using cquad
        result1 = 1. / (2. * M_PI * M_PI) * int_cquad_pivot_TV(fV3_112, q);

        // Second integration with qawo 1/2
        double alpha = 1. / (2. * M_PI * M_PI * q * q);
        gsl_function F;
        F.function = &fV3_112_qawo_1_over_2;
        F.params = &alpha;

        gsl_integration_workspace *w1 = gsl_integration_workspace_alloc(200);
        gsl_integration_workspace *w2 = gsl_integration_workspace_alloc(200);
        gsl_integration_qawo_table *t = gsl_integration_qawo_table_alloc(q, 0, GSL_INTEG_COSINE, size);
        gsl_integration_qawf(&F, xpivot_TV / q, 1e-6, 200, w1, w2, t, &result2, &error);

        // Second integration with qawo 2/2
        F.function = &fV3_112_qawo_2_over_2;
        F.params = &alpha;
        gsl_integration_qawo_table *t2 = gsl_integration_qawo_table_alloc(q, 0, GSL_INTEG_SINE, size);
        gsl_integration_qawf(&F, xpivot_TV / q, 1e-6, 200, w1, w2, t2, &result3, &error);

        gsl_integration_workspace_free(w1);
        gsl_integration_workspace_free(w2);
        gsl_integration_qawo_table_free(t);
        gsl_integration_qawo_table_free(t2);
        return result1 + result2 + result3;
    }
}

// V^10(q)
double fV_10(double k, void *p) {
    const double q = *(double *)p;
    return -1. / 7. * Q_s(k) * k * func_bessel_1(k * q);
}

double fV_10_qawo(double k, void *p) {
    const double alpha = *(double *)p;
    return -1. / 7. * alpha * Q_s(k);
}

double V_10(double q) {
    if (kmax_integ * q < xpivot_TV)
        return 1. / (2. * M_PI * M_PI) * int_cquad(fV_10, q);
    else {
        double result1, result2, error;
        // First integration using cquad
        result1 = 1. / (2. * M_PI * M_PI) * int_cquad_pivot_UYXi(fV_10, q);

        // Second integration with qawo
        double alpha = 1. / (2. * M_PI * M_PI * q);
        gsl_function F;
        F.function = &fV_10_qawo;
        F.params = &alpha;

        gsl_integration_workspace *w1 = gsl_integration_workspace_alloc(200);
        gsl_integration_workspace *w2 = gsl_integration_workspace_alloc(200);
        gsl_integration_qawo_table *t = gsl_integration_qawo_table_alloc(q, 0, GSL_INTEG_COSINE, size);
        gsl_integration_qawf(&F, xpivot_TV / q, 1e-4, 200, w1, w2, t, &result2, &error);

        gsl_integration_workspace_free(w1);
        gsl_integration_workspace_free(w2);
        gsl_integration_qawo_table_free(t);
        return result1 + result2;
    }
}

// T^112(q)
double fT_112(double k, void *p) {
    const double q = *(double *)p;
    const double x = k * q;
    return -3. / 7. * (2. * R_1(k) + 4. * R_2(k) + Q_1(k) + 2. * Q_2(k)) * func_bessel_3(x) / k;
}

double fT_112_qawo(double k, void *p) {
    const double alpha = *(double *)p;
    return -3. / 7. * alpha * (2. * R_1(k) + 4. * R_2(k) + Q_1(k) + 2. * Q_2(k)) / (k * k);
}

double T_112(double q) {
    if (kmin_integ * q < EPS)
        return 0;
    else if (kmax_integ * q < xpivot_TV)
        return 1. / (2. * M_PI * M_PI) * int_cquad(fT_112, q);
    else {
        double result1, result2, error;
        // First integration using cquad
        result1 = 1. / (2. * M_PI * M_PI) * int_cquad_pivot_TV(fT_112, q);

        // Second integration with qawo 1/2
        double alpha = 1. / (2. * M_PI * M_PI * q * q);
        gsl_function F;
        F.function = &fT_112_qawo;
        F.params = &alpha;

        gsl_integration_workspace *w1 = gsl_integration_workspace_alloc(200);
        gsl_integration_workspace *w2 = gsl_integration_workspace_alloc(200);
        gsl_integration_qawo_table *t = gsl_integration_qawo_table_alloc(q, 0, GSL_INTEG_COSINE, size);
        gsl_integration_qawf(&F, xpivot_TV / q, 1e-6, 200, w1, w2, t, &result2, &error);

        gsl_integration_workspace_free(w1);
        gsl_integration_workspace_free(w2);
        gsl_integration_qawo_table_free(t);
        return result1 + result2;
    }
}

/************************************************************************************************************************************************************************/
/*********************\\Interpolation functions\\***********************************************************************************************************************/

void interpole(int n, char ficher[], int nLines, int nHeader) {
    double T_x[nLines];
    double T_y[nLines];
    FILE *f;
    f = fopen(ficher, "r");
    for (int i = 0; i < nHeader; i++)
        fscanf(f, "%*[^\n]\n");
    for (int i = 0; i < nLines; i++)
        fscanf(f, "%lf %lf\n", &T_x[i], &T_y[i]);
    acc[n] = gsl_interp_accel_alloc();
    spline[n] = gsl_spline_alloc(gsl_interp_cspline, nLines);
    gsl_spline_init(spline[n], T_x, T_y, nLines);
    fclose(f);
}

/************************************************************************************************************************************************************************/
/*****************\\Matrix A \\ *****************************************************************************************************************************************/

/*A_ij(q) = X(q)*delta_k(i,j) + Y(q)*q_v[i]*q_v[j]*/

double Aij_13(int i, int j, double q) {
    return iX_13(q) * delta_K(i, j) + iY_13(q) * q_v[i] * q_v[j]; // q_v[i] unit vector over i component
}

double Aij_22(int i, int j, double q) {
    return iX_22(q) * delta_K(i, j) + iY_22(q) * q_v[i] * q_v[j];
}

double Aij_11(int i, int j, double q) {
    return iX_11(q) * delta_K(i, j) + iY_11(q) * q_v[i] * q_v[j];
}

double A_1loop_ij(int i, int j, double q) {
    return Aij_22(i, j, q) + 2. * Aij_13(i, j, q);
}

double Aij(int i, int j, double q) {
    return Aij_11(i, j, q) + Aij_22(i, j, q) + 2. * Aij_13(i, j, q);
}

double A10_ij(int i, int j, double q) {
    return 2. * (iX_10_12(q) * delta_K(i, j) + iY_10_12(q) * q_v[i] * q_v[j]);
}

// Initialize A matrix
void get_M(double q, gsl_matrix **A) {
    unsigned int i, j;
    for (i = 0; i < 3; i++) {
        for (j = 0; j < 3; j++) {
            if (do_CLPT)
                gsl_matrix_set(*A, i, j, Aij(i, j, q));
            else
                gsl_matrix_set(*A, i, j, Aij_11(i, j, q));
        }
    }
}

/************************************************************************************************************************************************************************/
/******************\\ g_i, G_ij and Gamma_ijk functions\\**************************************************************************************************************/

double g_i(int i, int j, gsl_matrix *M_inv, double y[]) {
    return gsl_matrix_get(M_inv, i, j) * y[j];
}

double G_ij(int i, int j, gsl_matrix *M_inv, double g[]) {
    return gsl_matrix_get(M_inv, i, j) - g[i] * g[j];
}

double Gamma_ijk(int i, int j, int k, gsl_matrix *M_inv, double g[]) {
    return gsl_matrix_get(M_inv, i, j) * g[k] + gsl_matrix_get(M_inv, k, i) * g[j] + gsl_matrix_get(M_inv, j, k) * g[i] - g[i] * g[j] * g[k];
}

/************************************************************************************************************************************************************************/
/******************\\Compute W^112_ijk(q)\\**************************************************************************************************************************/

double W_112(int i, int j, int k, double q) {
    return iW_V1(q) * delta_K(j, k) * q_v[i] + iW_V1(q) * delta_K(k, i) * q_v[j] + iW_V3(q) * delta_K(i, j) * q_v[k] + iW_T(q) * q_v[i] * q_v[j] * q_v[k];
}

/************************************************************************************************************************************************************************/
/*************\\Dot functions\\*****************************************************************************************************************************************/

// Udot_n
double fU_dot(double q) {
    return iU_1(q) + 3. * iU_3(q);
}

// Adot_in
double fA_dot(int i, int n, double q) {
    return Aij_11(i, n, q) + 4. * Aij_13(i, n, q) + 2. * Aij_22(i, n, q);
}

// Adot^10_in
double fA10_dot(int i, int n, double q) {
    return 3. * (iX_10_12(q) * delta_K(i, n) + iY_10_12(q) * q_v[i] * q_v[n]);
}

// Wdot_ijn
double fW_dot(int i, int j, int n, double q) {
    return 2. * W_112(i, j, n, q) + W_112(n, i, j, q) + W_112(j, n, i, q);
}

/************************************************************************************************************************************************************************/
/*********\\Double dot functions\\*************************************************************************************************************************************/

// A2dot_nm
double fA_2dot(int n, int m, double q) {
    return Aij_11(n, m, q) + 6. * Aij_13(n, m, q) + 4. * Aij_22(n, m, q);
}

// A2dot^10_nm
double fA10_2dot(int n, int m, double q) {
    return 4. * (iX_10_12(q) * delta_K(n, m) + iY_10_12(q) * q_v[n] * q_v[m]); // A_nm = A_mn
}

// W2dot_inm
double fW_2dot(int i, int n, int m, double q) {
    return 2. * W_112(i, n, m, q) + 2. * W_112(m, i, n, q) + W_112(n, m, i, q); // W^121_inm = W^112_min and W^211_inm = W^112_nmi
}

/************************************************************************************************************************************************************************/
/**************\\M_0 integral\\*****************************************************************************************************************************************/

void M_0(double y, double R, double M0_fin[]) {
    unsigned int i, j, k;
    const unsigned int n_ingredients = 13;
    double mu, q_n, det_A, y_t[3], q[3], U[3], U1[3], G[3][3], U11[3], U20[3], V10[3], V12[3], B2[3], g[3], Gamma[3][3][3], W[3][3][3], W_fin[3][3][3], A10[3][3], Ups[3][3], Xi, trG, s, fac, F_b[n_ingredients], M0_tab[n_ingredients];
    const double dmu = 2. / nbins_M0;
    int signum = 0;
    const double mumin = -1 + 0.5 * dmu;

    for (i = 0; i < n_ingredients; i++)
        M0_tab[i] = 0; // Inizialise at each step

    for (unsigned int ibin = 0; ibin < nbins_M0; ibin++) { // Compute Riemann integral as int=Sum f(x_i)*delta_x
        mu = mumin + ibin * dmu;
        gsl_matrix *A = gsl_matrix_calloc(3, 3);
        gsl_matrix *A_inv = gsl_matrix_calloc(3, 3); // Declaration of matrix A (3x3)

        // Change variable y = q-r
        y_t[0] = y * sqrt(1 - mu * mu);
        y_t[1] = 0;
        y_t[2] = y * mu;

        // Compute vector q = y+r with r= (0,0,1) along the LOS
        q[0] = y_t[0];
        q[1] = y_t[1];
        q[2] = y_t[2] + R;
        q_n = sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2]); // norm of q

        // Unit vector
        for (i = 0; i < 3; i++)
            q_v[i] = q[i] / q_n;

        /******** Compute functions ****************************************/

        // Xi linear
        Xi = iXi_L(q_n);
        // printf("%.13lf %.13lf # q, Xi\n", q_n, Xi);

        // U functions
        const double U_tmp = get_U(q_n);
        const double U_1_tmp = iU_1(q_n);
        const double U_11_tmp = iU_11(q_n);
        const double U_20_tmp = iU_20(q_n);
        for (i = 0; i < 3; i++) {
            U[i] = U_tmp * q_v[i];
            U1[i] = U_1_tmp * q_v[i];
            U11[i] = U_11_tmp * q_v[i];
            U20[i] = U_20_tmp * q_v[i];
        }

        // 	Matix A
        gsl_permutation *p = gsl_permutation_alloc(3);
        get_M(q_n, &A); // Initialize matix A (A_11 for Zel'dovich and CLEFT)
        gsl_linalg_LU_decomp(A, p, &signum);
        det_A = gsl_linalg_LU_det(A, signum); // Determinant of A
        gsl_linalg_LU_invert(A, p, A_inv);    // Inverse of matrix A

        // 	g_i components
        for (i = 0; i < 3; i++) {
            g[i] = 0;
            for (j = 0; j < 3; j++)
                g[i] += g_i(i, j, A_inv, y_t);
        }

        // 	G_ij components
        for (i = 0; i < 3; i++) {
            for (j = 0; j < 3; j++)
                G[i][j] = G_ij(i, j, A_inv, g);
        }

        trG = 0;
        for (i = 0; i < 3; i++)
            trG += G[i][i];

        // Gamma_ijk
        for (i = 0; i < 3; i++) {
            for (j = 0; j < 3; j++) {
                for (k = 0; k < 3; k++)
                    Gamma[i][j][k] = Gamma_ijk(i, j, k, A_inv, g);
            }
        }

        // 	W_ijk
        for (i = 0; i < 3; i++) {
            for (j = 0; j < 3; j++) {
                for (k = 0; k < 3; k++) {
                    W[i][j][k] = W_112(i, j, k, q_n);
                }
            }
        }
        for (i = 0; i < 3; i++) {
            for (j = 0; j < 3; j++) {
                for (k = 0; k < 3; k++) {
                    W_fin[i][j][k] = W[i][j][k] + W[k][i][j] + W[j][k][i];
                }
            }
        }

        // 	A^10_ij
        for (i = 0; i < 3; i++) {
            for (j = 0; j < 3; j++)
                A10[i][j] = A10_ij(i, j, q_n);
        }

        // Sum over all the bias component
        for (i = 0; i < 13; i++)
            F_b[i] = 0;

        // "1" term
        F_b[0] = 1;

        // b1
        s = 0;
        for (i = 0; i < 3; i++) {
            if (do_zeldovich)
                s += U1[i] * g[i];
            else
                s += U[i] * g[i];
        }

        F_b[1] -= 2. * s;

        // b1²
        F_b[3] += Xi;

        if (!do_zeldovich) {
            // 1 term
            s = 0;
            for (i = 0; i < 3; i++) {
                for (j = 0; j < 3; j++) {
                    for (k = 0; k < 3; k++) {
                        s += Gamma[i][j][k] * W_fin[i][j][k];
                    }
                }
            }
            F_b[0] += s / 6.;

            // b1
            s = 0;
            for (i = 0; i < 3; i++) {
                for (j = 0; j < 3; j++)
                    s += A10[i][j] * G[i][j];
            }
            F_b[1] -= s;

            // b1^2
            s = 0;
            for (i = 0; i < 3; i++)
                s += U11[i] * g[i];

            F_b[3] -= s;
            s = 0;
            for (i = 0; i < 3; i++) {
                for (j = 0; j < 3; j++)
                    s += U1[i] * U1[j] * G[i][j];
            }
            F_b[3] -= s;
            // b2
            s = 0;
            for (i = 0; i < 3; i++)
                s += U20[i] * g[i];
            F_b[2] -= s;

            s = 0;
            for (i = 0; i < 3; i++) {
                for (j = 0; j < 3; j++)
                    s += U1[i] * U1[j] * G[i][j];
            }
            F_b[2] -= s;

            // b1.b2
            s = 0;
            for (i = 0; i < 3; i++)
                s += U1[i] * g[i];
            F_b[4] -= 2. * Xi * s;

            // b2²
            F_b[5] = 0.5 * Xi * Xi;

            if (do_CLEFT) {
                // V functions
                const double V_10_tmp = iV_10(q_n);
                const double V_12_tmp = iV_12(q_n);
                for (i = 0; i < 3; i++) {
                    V10[i] = V_10_tmp * q_v[i];
                    V12[i] = V_12_tmp * q_v[i];
                }

                // B function
                const double B_2_tmp = iB_2(q_n);
                for (i = 0; i < 3; i++) {
                    B2[i] = B_2_tmp * q_v[i];
                }

                // 	Ups_ij
                const double j2 = iJ_2(q_n);
                const double j3 = iJ_3(q_n);
                const double j4 = iJ_4(q_n);
                for (i = 0; i < 3; i++) {
                    for (j = 0; j < 3; j++) {
                        if (i == j)
                            Ups[i][j] = 2 * j3 * j3;
                        else
                            Ups[i][j] = 0;
                        Ups[i][j] += q_v[i] * q_v[j] * (3 * j2 * j2 + 4 * j2 * j3 + 2 * j2 * j4 + 2 * j3 * j3 + 4 * j3 * j4 + j4 * j4);
                        Ups[i][j] *= 2;
                    }
                }

                // 1 term
                s = 0;
                for (i = 0; i < 3; i++) {
                    for (j = 0; j < 3; j++)
                        s += G[i][j] * A_1loop_ij(i, j, q_n);
                }
                F_b[0] -= 0.5 * s; // TBC
                // bs2
                s = 0;
                for (i = 0; i < 3; i++) {
                    for (j = 0; j < 3; j++)
                        s += Ups[i][j] * G[i][j];
                }
                for (i = 0; i < 3; i++)
                    s += 2 * g[i] * V10[i];
                F_b[6] -= s;

                // bs2^2
                F_b[7] += iZeta(q_n);

                // b1 bs2
                s = 0;
                for (i = 0; i < 3; i++)
                    s += 2 * g[i] * V12[i];
                F_b[8] -= s;

                // b2 bs2
                F_b[9] += iChi_12(q_n);

                // alpha_xi
                F_b[10] -= 0.5 * trG;

                // bn2
                F_b[11] += 2 * iB_1(q_n);

                // b1 bn2
                s = 0;
                for (i = 0; i < 3; i++)
                    s += 2 * g[i] * B2[i];
                F_b[12] += s;
            }
        }
        // 	gaussian factor of j
        s = 0;
        for (i = 0; i < 3; i++)
            s += g[i] * y_t[i];
        fac = exp(-0.5 * s) / (sqrt(2. * M_PI) * 2. * M_PI * sqrt(det_A));
        //	Sum over all component
        for (i = 0; i < n_ingredients; i++)
            M0_tab[i] += F_b[i] * fac;

        gsl_permutation_free(p);
        gsl_matrix_free(A);
        gsl_matrix_free(A_inv);
    }
    s = y * y * dmu;
    for (i = 0; i < 13; i++)
        M0_fin[i] = M0_tab[i] * s * 2. * M_PI;

    return;
}

/************************************************************************************************************************************************************************/
/**************\\M_1 integral\\*****************************************************************************************************************************************/

void M_1(double y, double R, double M1_fin[]) {
    const double rn[3] = {0, 0, 1}; // unit vector to project over the LOS
    unsigned int i, j, k;
    const unsigned int n_ingredients = 10;
    double mu, q_n, det_A, y_t[3], q[3], U_dot[3], U1[3], G[3][3], U11_dot[3], U20_dot[3], g[3], B2[3], W[3][3][3], W_dot[3][3][3], A_lin[3][3], A_dot[3][3], A10_dot[3][3], Ups[3][3], Xi, s, fac, F_b[n_ingredients], M1_tab[n_ingredients];
    const double dmu = 2. / nbins_M1;
    int signum = 0; // Use for matrix inversion
    const double mumin = -1 + 0.5 * dmu;

    for (i = 0; i < n_ingredients; i++)
        M1_tab[i] = 0; // Initialize at each step

    for (unsigned int ibin = 0; ibin < nbins_M1; ibin++) { // Compute Riemann integral as int=Sum f(x_i)*delta_x
        mu = mumin + ibin * dmu;
        gsl_matrix *A = gsl_matrix_calloc(3, 3);
        gsl_matrix *A_inv = gsl_matrix_calloc(3, 3); // Declaration of matrix A (3x3)

        // Change variable y = q-r
        y_t[0] = y * sqrt(1 - mu * mu);
        y_t[1] = 0;
        y_t[2] = y * mu;

        // Compute vector q = y+r with r= (0,0,1) along the LOS
        q[0] = y_t[0];
        q[1] = y_t[1];
        q[2] = y_t[2] + R;
        q_n = sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2]); // norm of q

        // Unit vector
        for (i = 0; i < 3; i++)
            q_v[i] = q[i] / q_n;

        /********Compute function ****************************************/

        // Xi linear
        Xi = iXi_L(q_n);

        //  	U functions
        const double U_dot_tmp = fU_dot(q_n);
        const double U_1_tmp = iU_1(q_n);
        const double U_11_tmp = iU_11(q_n);
        const double U_20_tmp = iU_20(q_n);
        for (i = 0; i < 3; i++) {
            U_dot[i] = U_dot_tmp * q_v[i];
            U1[i] = U_1_tmp * q_v[i];
            U11_dot[i] = 2. * U_11_tmp * q_v[i];
            U20_dot[i] = 2. * U_20_tmp * q_v[i];
        }

        // 	Matix A
        gsl_permutation *p = gsl_permutation_alloc(3);
        get_M(q_n, &A); // Initialize matix A
        gsl_linalg_LU_decomp(A, p, &signum);
        det_A = gsl_linalg_LU_det(A, signum); // Determinant of A
        gsl_linalg_LU_invert(A, p, A_inv);    // Inverse of matrix A

        // 	g_i components
        for (i = 0; i < 3; i++) {
            g[i] = 0;
            for (j = 0; j < 3; j++)
                g[i] += g_i(i, j, A_inv, y_t);
        }

        // G_ij components
        for (i = 0; i < 3; i++) {
            for (j = 0; j < 3; j++)
                G[i][j] = G_ij(i, j, A_inv, g);
        }

        // Alin
        for (i = 0; i < 3; i++) {
            for (j = 0; j < 3; j++)
                A_lin[i][j] = Aij_11(i, j, q_n);
        }

        // Adot_in
        for (i = 0; i < 3; i++) {
            for (j = 0; j < 3; j++)
                A_dot[i][j] = fA_dot(i, j, q_n);
        }

        // Adot^10_in
        for (i = 0; i < 3; i++) {
            for (j = 0; j < 3; j++)
                A10_dot[i][j] = fA10_dot(i, j, q_n);
        }

        // W_ijn
        for (i = 0; i < 3; i++) {
            for (j = 0; j < 3; j++) {
                for (k = 0; k < 3; k++)
                    W[i][j][k] = W_112(i, j, k, q_n);
            }
        }
        for (i = 0; i < 3; i++) {
            for (j = 0; j < 3; j++) {
                for (k = 0; k < 3; k++)
                    W_dot[i][j][k] = 2. * W[i][j][k] + W[k][i][j] + W[j][k][i];
            }
        }

        // Sum over all the bias component
        for (i = 0; i < 10; i++)
            F_b[i] = 0;

        // "1" term
        s = 0;
        for (i = 0; i < 3; i++) {
            if (do_zeldovich)
                s += g[i] * A_lin[i][2] * rn[2];
            else
                s += g[i] * A_dot[i][2] * rn[2];
        }
        F_b[0] -= s;

        // 	b1
        s = 0;
        if (do_zeldovich)
            s += U1[2] * rn[2];
        else
            s += U_dot[2] * rn[2];

        F_b[1] += 2. * s;

        if (!do_zeldovich) {
            // 1 term
            s = 0;
            for (i = 0; i < 3; i++) {
                for (j = 0; j < 3; j++) {
                    s += G[i][j] * W_dot[i][j][2] * rn[2];
                }
            }
            F_b[0] -= s / 2.;

            // b1
            s = 0;
            for (i = 0; i < 3; i++) {
                s += g[i] * A10_dot[i][2] * rn[2];
            }
            F_b[1] -= 2. * s;

            s = 0;
            for (i = 0; i < 3; i++) {
                for (j = 0; j < 3; j++) {
                    s += G[i][j] * U1[i] * A_lin[j][2] * rn[2];
                }
            }
            F_b[1] -= 2. * s;

            // b1^2
            F_b[3] += U11_dot[2] * rn[2];

            s = 0;
            for (i = 0; i < 3; i++) {
                s += g[i] * U1[i] * U1[2] * rn[2];
            }
            F_b[3] -= 2. * s;

            s = 0;
            for (i = 0; i < 3; i++)
                s += g[i] * A_lin[i][2] * rn[2];
            F_b[3] -= Xi * s;

            // b2
            F_b[2] += U20_dot[2] * rn[2];

            s = 0;
            for (i = 0; i < 3; i++) {
                s += g[i] * U1[i] * U1[2] * rn[2];
            }
            F_b[2] -= 2. * s;

            // b1.b2
            s = 0;
            s += U1[2] * rn[2];

            F_b[4] += 2. * Xi * s;

            if (do_CLEFT) {
                // B functions
                const double B_2_tmp = iB_2(q_n);
                for (i = 0; i < 3; i++) {
                    B2[i] = B_2_tmp * q_v[i];
                }
                // 	Ups_ij
                const double j2 = iJ_2(q_n);
                const double j3 = iJ_3(q_n);
                const double j4 = iJ_4(q_n);
                for (i = 0; i < 3; i++) {
                    for (j = 0; j < 3; j++) {
                        if (i == j)
                            Ups[i][j] = 2 * j3 * j3;
                        else
                            Ups[i][j] = 0;
                        Ups[i][j] += q_v[i] * q_v[j] * (3 * j2 * j2 + 4 * j2 * j3 + 2 * j2 * j4 + 2 * j3 * j3 + 4 * j3 * j4 + j4 * j4);
                        Ups[i][j] *= 2;
                    }
                }
                // alpha_v
                F_b[5] = -iB_2(q_n) * rn[2];

                // alpha_vp
                F_b[6] = -g[2] * rn[2];

                // bn2
                F_b[7] = -2 * B2[2] * rn[2];

                // bs2
                s = 2 * iV_10(q_n) * rn[2];
                for (i = 0; i < 3; i++)
                    s -= Ups[i][2] * g[i] * rn[2];
                F_b[8] = 2 * s;

                // b1 bs2
                F_b[9] = 2 * iV_12(q_n) * rn[2];
            }
        }
        // 	Gaussian factor of j
        s = 0;
        for (i = 0; i < 3; i++)
            s += g[i] * y_t[i];
        fac = exp(-0.5 * s) / (sqrt(2. * M_PI) * 2. * M_PI * sqrt(det_A));
        //	Sum over all component
        for (i = 0; i < n_ingredients; i++)
            M1_tab[i] += F_b[i] * fac;

        gsl_matrix_free(A);
        gsl_matrix_free(A_inv);
        gsl_permutation_free(p);
    }
    s = y * y * dmu;
    for (i = 0; i < n_ingredients; i++)
        M1_fin[i] = M1_tab[i] * s * 2. * M_PI;

    return;
}

/************************************************************************************************************************************************************************/
/************************************************************************************************************************************************************************/
// Calcul de M2

void M_2(double y, double R, double M2_fin[]) {
    const double rn[3] = {0, 0, 1}; // unit vector to project on n and j
    unsigned int i, j, k, n;
    const unsigned int n_ingredients = 7;
    double mu, q_n, det_A, y_t[3], q[3], U1[3], G[3][3], g[3], W[3][3][3], W_2dot[3][3][3], A_lin[3][3], A_2dot[3][3], A10_2dot[3][3], Ups[3][3], Xi, s[3][3], f, fac, M2_tab[2 * n_ingredients], t;
    double F_par[n_ingredients], F_per[n_ingredients]; // Bias factor for sigma_parallel and sigma_perpendicular
    const double dmu = 2. / nbins_M2;
    int signum = 0; // Use for matrix inversion
    const double mumin = -1 + 0.5 * dmu;

    for (i = 0; i < 2 * n_ingredients; i++)
        M2_tab[i] = 0; // Initialize at each step

    for (unsigned int ibin = 0; ibin < nbins_M2; ibin++) { // Compute Riemann integral as int=Sum f(x_i)*delta_x
        mu = mumin + ibin * dmu;
        gsl_matrix *A = gsl_matrix_calloc(3, 3);
        gsl_matrix *A_inv = gsl_matrix_calloc(3, 3); // Declaration of matrix A (3x3)

        // Change variable y = q-r
        y_t[0] = y * sqrt(1 - mu * mu);
        y_t[1] = 0;
        y_t[2] = y * mu;

        // 	Compute vector q = y+r with r= (0,0,1) along the LOS
        q[0] = y_t[0];
        q[1] = y_t[1];
        q[2] = y_t[2] + R;
        q_n = sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2]); // norm of q

        // 	unit vector
        for (i = 0; i < 3; i++)
            q_v[i] = q[i] / q_n;

        /********Compute function ****************************************/
        // Xi linear
        Xi = iXi_L(q_n);

        // 	U function
        const double U_1_tmp = iU_1(q_n);
        for (i = 0; i < 3; i++) {
            U1[i] = U_1_tmp * q_v[i];
        }

        // 	Matix A
        gsl_permutation *p = gsl_permutation_alloc(3);
        get_M(q_n, &A); // Initialize matix A
        gsl_linalg_LU_decomp(A, p, &signum);
        det_A = gsl_linalg_LU_det(A, signum); // Determinant of A
        gsl_linalg_LU_invert(A, p, A_inv);    // Inverse of matrix A

        // 	g_i components
        for (i = 0; i < 3; i++) {
            g[i] = 0;
            for (j = 0; j < 3; j++)
                g[i] += g_i(i, j, A_inv, y_t);
        }

        // 	G_ij components
        for (i = 0; i < 3; i++) {
            for (j = 0; j < 3; j++)
                G[i][j] = G_ij(i, j, A_inv, g);
        }

        // 	Alin_nm
        for (i = 0; i < 3; i++) {
            for (j = 0; j < 3; j++)
                A_lin[i][j] = Aij_11(i, j, q_n);
        }

        //	A2dot_nm
        for (i = 0; i < 3; i++) {
            for (j = 0; j < 3; j++)
                A_2dot[i][j] = fA_2dot(i, j, q_n);
        }

        //	A2dot^10_nm
        for (i = 0; i < 3; i++) {
            for (j = 0; j < 3; j++)
                A10_2dot[i][j] = fA10_2dot(i, j, q_n);
        }

        //  W_inm
        for (i = 0; i < 3; i++) {
            for (j = 0; j < 3; j++) {
                for (k = 0; k < 3; k++)
                    W[i][j][k] = W_112(i, j, k, q_n);
            }
        }
        for (i = 0; i < 3; i++) {
            for (j = 0; j < 3; j++) {
                for (k = 0; k < 3; k++)
                    W_2dot[i][j][k] = 2. * W[i][j][k] + W[k][i][j] + 2. * W[j][k][i];
            }
        }

        // Sum over all the bias component
        for (i = 0; i < 7; i++)
            F_par[i] = F_per[i] = 0;

        // "1" term
        for (i = 0; i < 3; i++) {
            for (j = 0; j < 3; j++)
                if (do_zeldovich)
                    s[i][j] = A_lin[i][j];
                else
                    s[i][j] = A_2dot[i][j];
        }

        if (!do_zeldovich) {
            // 1 term
            for (i = 0; i < 3; i++) {
                for (j = 0; j < 3; j++) {
                    for (n = 0; n < 3; n++) {
                        for (k = 0; k < 3; k++) {
                            s[n][k] -= A_lin[i][n] * A_lin[j][k] * G[i][j];
                        }
                    }
                }
            }

            for (i = 0; i < 3; i++) {
                for (j = 0; j < 3; j++) {
                    for (k = 0; k < 3; k++)
                        s[j][k] -= W_2dot[i][j][k] * g[i];
                }
            }
            for (i = 0; i < 3; i++) {
                for (j = 0; j < 3; j++) {
                    F_par[0] += s[i][j] * rn[i] * rn[j];
                    F_per[0] += s[i][j] * delta_K(i, j);
                }
            }

            // 	b1
            for (i = 0; i < 3; i++) {
                for (j = 0; j < 3; j++)
                    s[i][j] = A10_2dot[i][j];
            }
            for (i = 0; i < 3; i++) {
                for (j = 0; j < 3; j++) {
                    for (k = 0; k < 3; k++) {
                        s[j][k] -= A_lin[i][j] * g[i] * U1[k] + A_lin[i][k] * g[i] * U1[j];
                        s[j][k] -= U1[i] * g[i] * A_lin[j][k];
                    }
                }
            }
            for (i = 0; i < 3; i++) {
                for (j = 0; j < 3; j++) {
                    F_par[1] += 2. * s[i][j] * rn[i] * rn[j];
                    F_per[1] += 2. * s[i][j] * delta_K(i, j);
                }
            }

            // b1²
            for (i = 0; i < 3; i++) {
                for (j = 0; j < 3; j++) {
                    s[i][j] = Xi * A_lin[i][j] + 2. * U1[i] * U1[j];
                }
            }

            for (i = 0; i < 3; i++) {
                for (j = 0; j < 3; j++) {
                    F_par[3] += s[i][j] * rn[i] * rn[j];
                    F_per[3] += s[i][j] * delta_K(i, j);
                }
            }
            // b2
            for (i = 0; i < 3; i++) {
                for (j = 0; j < 3; j++)
                    s[i][j] = U1[i] * U1[j];
            }
            for (i = 0; i < 3; i++) {
                for (j = 0; j < 3; j++) {
                    F_par[2] += 2. * s[i][j] * rn[i] * rn[j];
                    F_per[2] += 2. * s[i][j] * delta_K(i, j);
                }
            }

            if (do_CLEFT) {
                // Ups_ij
                const double j2 = iJ_2(q_n);
                const double j3 = iJ_3(q_n);
                const double j4 = iJ_4(q_n);
                for (i = 0; i < 3; i++) {
                    for (j = 0; j < 3; j++) {
                        if (i == j)
                            Ups[i][j] = 2 * j3 * j3;
                        else
                            Ups[i][j] = 0;
                        Ups[i][j] += q_v[i] * q_v[j] * (3 * j2 * j2 + 4 * j2 * j3 + 2 * j2 * j4 + 2 * j3 * j3 + 4 * j3 * j4 + j4 * j4);
                        Ups[i][j] *= 2;
                    }
                }

                // bs2
                F_par[4] = 2 * Ups[2][2];
                t = 0;
                for (i = 0; i < 3; i++)
                    t += Ups[i][i];
                F_per[4] = 2 * t;

                // alpha_s
                F_par[5] = 1;
                F_per[5] = 3;

                // beta_s
                F_par[6] += Xi;
                F_per[6] += 3 * Xi;
            }
        }
        // Gaussian factor of j
        f = 0;
        for (i = 0; i < 3; i++)
            f += g[i] * y_t[i];
        fac = exp(-0.5 * f) / (sqrt(2. * M_PI) * 2. * M_PI * sqrt(fabs(det_A)));
        // Sum over all component
        for (i = 0; i < n_ingredients; i++) {
            M2_tab[i] += F_par[i] * fac;
            M2_tab[i + 7] += F_per[i] * fac;
        }

        gsl_permutation_free(p);
        gsl_matrix_free(A);
        gsl_matrix_free(A_inv);
    }

    f = y * y * dmu;
    for (i = 0; i < 7; i++) {
        M2_fin[i] = M2_tab[i] * f * 2. * M_PI;
        M2_fin[i + 7] = 0.5 * f * (M2_tab[i + 7] - M2_tab[i]) * 2. * M_PI;
    }

    return;
}

/************************************************************************************************************************************************************************/
/************************************************************************************************************************************************************************/
/******** Write all terms up to second order *******************************************************************************************************************************/

void write_ingredients(char file[]) {
    unsigned int i, n0 = 13, n1 = 10, n2 = 14;
    double r;
    char filename[strlen(file) + 1 + 7];
    FILE *fi;

    strcpy(filename, file);
    if (do_zeldovich)
        strcat(filename, ".za");
    else if (do_CLPT)
        strcat(filename, ".clpt");
    else if (do_CLEFT)
        strcat(filename, ".cleft");
    else {
        printf("Code should not have arrived here\n");
        exit(-1);
    }

    printf("Calculating xi, v, sigma contributions...\n");

    fi = fopen(filename, "w");
    fprintf(fi, "# All contributions\n");
    fprintf(fi, "# r, xi_L, xi_1, xi_b1, xi_b2, xi_b1^2, xi_b1.b2, xi_b2^2, xi_bs, xi_bs^2, xi_b1.bs, xi_b2bs, xi_A, xi_bn2, xi_b1.bn2, ");
    fprintf(fi, "v_1, v_b1, v_b2, v_b1^2, v_b1.b2, v_A, v_Ap, v_bn2, v_bs, v_b1.bs, ");
    fprintf(fi, "spar_1, spar_b1, spar_b2, spar_b1^2, spar_bs, spar_A, spar_B, s_1, s_b1, s_b2, s_b1^2, s_bs, s_A, s_B\n");

    for (r = 1; r <= r_max; r += 1) {
        double M0[n0], M1[n1], M2[n2];

        gl_int6(0, 100, &M_0, &r, M0, n0);
        gl_int6(0, 100, &M_1, &r, M1, n1);
        gl_int6(0, 100, &M_2, &r, M2, n2);
        M0[0] -= 1;

        fprintf(fi, "%.13le %.13le ", r, iXi_L(r));
        for (i = 0; i < n0; i++)
            fprintf(fi, "%.13le ", M0[i]);
        for (i = 0; i < n1; i++)
            fprintf(fi, "%.13le ", M1[i]);
        for (i = 0; i < n2; i++)
            fprintf(fi, "%.13le ", M2[i]);
        fprintf(fi, "\n");
    }
    fclose(fi);

    return;
}

/************************************************************************************************************************************************************************/
/************\\Count the number of lines in a file\\********************************************************************************************************************/

void compte(FILE *fichier, int *fcounts) {
    int c;
    int nLines = 0;
    int c2 = '\0';
    int nHeader = 0;

    while ((c = fgetc(fichier)) != EOF) {
        if (c == '\n')
            nLines++;
        c2 = c;
    }
    if (c2 != '\n')
        nLines++;

    rewind(fichier);
    char line[BUFSIZ];
    while (fgets(line, sizeof(line), fichier)) {
        if (*line == '#')
            nHeader++;
    }

    fcounts[0] = nLines - nHeader;
    fcounts[1] = nHeader;

    return;
}

void jfuncs(const double q, double sum[]) {
    // Computes the \mathcal{J}_n integrals, which are used in the shear terms, and the other shear-related terms.
    double sum0 = 0, sum1 = 0, sum2 = 0, sum3 = 0, sum4 = 0, sum5 = 0, sum6 = 0, sum7 = 0, sum8 = 0, sum9 = 0, sum10 = 0;
    double zeta;
    double qmin = log(1e-5);
    double qmax = log(1e2);

    int Nint = (int)(10 * exp(qmax) * q + 512);
    double hh = (qmax - qmin) / (double)Nint;

#pragma omp parallel for reduction(+ : sum0, sum1, sum2, sum3, sum4, sum5, sum6, sum7, sum8, sum9, sum10)
    for (int i = 1; i < Nint; ++i) {
        double xx = qmin + i * hh;
        double ap = cos(M_PI / 2. * exp(xx - qmax));
        double kk = exp(xx);
        double k2 = kk * kk;
        double kq = kk * q;
        double pk = P_L(kk);
        double j0 = func_bessel_0(kq);
        double j1 = func_bessel_1(kq);
        double j2, j3, j4;
        int wt = 2 + 2 * (i % 2);

        j2 = 3. * j1 / kq - j0;
        j3 = 5. * j2 / kq - j1;
        j4 = 7. * j3 / kq - j2;

        sum0 += kk * k2 * k2 * pk * j0 * wt * ap; //*exp(-k2)
        sum1 += k2 * k2 * pk * j1 * wt * ap;
        sum2 += k2 * pk * kk * j2 * wt;
        sum3 += k2 * pk * (2. / 15. * j1 - 1. / 5. * j3) * wt * ap;
        sum4 += k2 * pk * (-1. / 5. * j1 - 1. / 5. * j3) * wt;
        sum5 += k2 * pk * (j3)*wt;
        sum6 += k2 * pk * kk * (-14 * j0 - 40 * j2 + 9 * j4) / 315. * wt * ap;
        sum7 += k2 * pk * kk * (7 * j0 + 10 * j2 + 3 * j4) / 105. * wt * ap;
        sum8 += k2 * pk * kk * (4 * j2 - 3 * j4) / 21. * wt * ap;
        sum9 += k2 * pk * kk * (-3 * j2 - 3 * j4) / 21. * wt * ap;
        sum10 += k2 * pk * kk * (j4)*wt * ap;
    }
    sum6 *= hh / 3.0 / (2 * M_PI * M_PI);
    sum7 *= hh / 3.0 / (2 * M_PI * M_PI);
    sum8 *= hh / 3.0 / (2 * M_PI * M_PI);
    sum9 *= hh / 3.0 / (2 * M_PI * M_PI);
    sum10 *= hh / 3.0 / (2 * M_PI * M_PI);

    zeta = sum6 * (9 * sum6 + 12 * sum7 + 12 * sum8 + 8 * sum9 + 2 * sum10) +
           sum7 * (24 * sum7 + 8 * sum8 + 32 * sum9 + 4 * sum10) +
           sum8 * (+8 * sum8 + 16 * sum9 + 4 * sum10) +
           sum9 * (24 * sum9 + 8 * sum10) +
           sum10 * (sum10);

    sum[0] = sum0 * hh / 3.0 / (2 * M_PI * M_PI); // mathcal{B}_1
    sum[1] = sum1 * hh / 3.0 / (2 * M_PI * M_PI); // mathcal{B}_2
    sum[2] = sum2 * hh / 3.0 / (2 * M_PI * M_PI); // mathcal{J}_1
    sum[3] = sum3 * hh / 3.0 / (2 * M_PI * M_PI); // mathcal{J}_2
    sum[4] = sum4 * hh / 3.0 / (2 * M_PI * M_PI); // mathcal{J}_3
    sum[5] = sum5 * hh / 3.0 / (2 * M_PI * M_PI); // mathcal{J}_4
    sum[6] = 4 * sum[2] * sum[3];                 // V_i^{12}
    sum[7] = 4. / 3. * sum[2] * sum[2];           // chi12
    sum[8] = 2 * zeta;                            // zeta

    return;
}

void compute_and_interpolate_jfuncs(void) {
    unsigned int i;
    double q;
    const double qmax = 2000;        // Maximum q to compute the q functions, 2000 by default
    const unsigned int nbins = 3001; // Number of steps
    const double log_qmax = log10(qmax);
    const double dq = log_qmax / (nbins - 2);

    printf("getting jfunctions...\n");
    double q_array[nbins], B1[nbins], B2[nbins], J2[nbins], J3[nbins], J4[nbins], v12[nbins], chi12[nbins], zeta[nbins];

    q_array[0] = v12[0] = chi12[0] = zeta[0] = 0.;

    //  Reset the value of logq
    for (i = 1; i < nbins; i++) {
        q = pow(10, i * dq);
        double funcs[9];
        jfuncs(q, funcs);
        q_array[i] = q;
        B1[i] = funcs[0];
        B2[i] = funcs[1];
        J2[i] = funcs[3];
        J3[i] = funcs[4];
        J4[i] = funcs[5];
        v12[i] = funcs[6];
        chi12[i] = funcs[7];
        zeta[i] = funcs[8];
    }

    acc[26] = gsl_interp_accel_alloc();
    spline[26] = gsl_spline_alloc(gsl_interp_cspline, nbins);
    gsl_spline_init(spline[26], q_array, B1, nbins);

    acc[27] = gsl_interp_accel_alloc();
    spline[27] = gsl_spline_alloc(gsl_interp_cspline, nbins);
    gsl_spline_init(spline[27], q_array, B2, nbins);

    acc[28] = gsl_interp_accel_alloc();
    spline[28] = gsl_spline_alloc(gsl_interp_cspline, nbins);
    gsl_spline_init(spline[28], q_array, J2, nbins);

    acc[29] = gsl_interp_accel_alloc();
    spline[29] = gsl_spline_alloc(gsl_interp_cspline, nbins);
    gsl_spline_init(spline[29], q_array, J3, nbins);

    acc[30] = gsl_interp_accel_alloc();
    spline[30] = gsl_spline_alloc(gsl_interp_cspline, nbins);
    gsl_spline_init(spline[30], q_array, J4, nbins);

    acc[31] = gsl_interp_accel_alloc();
    spline[31] = gsl_spline_alloc(gsl_interp_cspline, nbins);
    gsl_spline_init(spline[31], q_array, v12, nbins);

    acc[32] = gsl_interp_accel_alloc();
    spline[32] = gsl_spline_alloc(gsl_interp_cspline, nbins);
    gsl_spline_init(spline[32], q_array, chi12, nbins);

    acc[33] = gsl_interp_accel_alloc();
    spline[33] = gsl_spline_alloc(gsl_interp_cspline, nbins);
    gsl_spline_init(spline[33], q_array, zeta, nbins);

    return;
}

void qfuncs(const double q, double sum[]) {
    int i;
    double sum0 = 0, sum1 = 0, sum2 = 0, sum3 = 0, sum4 = 0, sum5 = 0, sum6 = 0, sum7 = 0, sum8 = 0, sum9 = 0, sum10 = 0, sum11 = 0, sum12 = 0, sumS = 0;
    double qmin = log(1e-5);
    double qmax = log(1e2);

    int Nint = (int)(10 * exp(qmax) * q + 512);
    double hh = (qmax - qmin) / (double)Nint;

#pragma omp parallel for reduction(+ : sum0, sum1, sum2, sum3, sum4, sum5, sum6, sum7, sum8, sum9, sum10, sum11, sum12, sumS)
    for (i = 1; i < Nint; ++i) {
        double xx = qmin + i * hh;
        double kk = exp(xx);
        double k2 = kk * kk;
        double kq = kk * q;
        double R1 = R_1(kk);
        double R2 = R_2(kk);
        double Q1 = Q_1(kk);
        double Q2 = Q_2(kk);
        double Q5 = Q_5(kk);
        double Q8 = Q_8(kk);
        double Qs = Q_s(kk);
        double j0 = func_bessel_0(kq);
        double j1 = func_bessel_1(kq);
        double j2, j3;
        double j1d = j1 / kq;
        int wt = 2 + 2 * (i % 2);

        if (kq < 0.1) {
            j2 = pow(kq, 2.0) / 15. - pow(kq, 4.0) / 210.;
            j3 = pow(kq, 3.0) / 105. - pow(kq, 5.0) / 1890.;
        } else {
            j2 = 3. * j1 / kq - j0;
            j3 = 5. * j2 / kq - j1;
        }

        sum0 += kk * wt * (9. / 98. * Q1 * (2. / 3. - 2 * j1d));                                      // X^{(22)}
        sum1 += kk * wt * (5. / 21. * R1 * (2. / 3. - 2 * j1d));                                      // X^{(13)}
        sum2 += kk * wt * (9. / 98. * Q1 * (-2 * j0 + 6 * j1d));                                      // Y^{(22)}
        sum3 += kk * wt * (5. / 21. * R1 * (-2 * j0 + 6 * j1d));                                      // Y^{(13)}
        sum4 += kk * wt * (2 * (R1 - R2) + 3 * R1 * j0 - 3 * (3 * R1 + 4 * R2 + 2 * Q5) * j1d) / 14.; // X1210
        sum5 += kk * wt * (3 * R1 + 4 * R2 + 2 * Q5) * (j0 - 3 * j1d) * (-3. / 14.);                  // Y_{10}^{(12)}
        sum6 += wt * (R1 * j1) * (-3. / 7.);                                                          // V_1^{(112)}
        sum7 += wt * (Q1 * j1) * (-3. / 7.);                                                          // V_3^{(112)}
        sumS += wt * (2 * R1 + 4 * R2 + Q1 + 2 * Q2) * (3. / 7. * j2 / (kk * q));                     // S^{(112)}
        sum8 += wt * (2 * R1 + 4 * R2 + Q1 + 2 * Q2) * j3 * (-3. / 7.);                               // T^{(112)}
        sum9 += k2 * wt * (R1 * j1) * (-5. / 21.);                                                    // U^{(3)}
        sum10 += k2 * wt * (Q8 * j1) * (-3. / 7.);                                                    // U_{20}^{(2)}
        sum11 += k2 * wt * ((R1 + R2) * j1) * (-6. / 7.);                                             // U_{11}^{(2)}
        sum12 += k2 * wt * (Qs * j1) * (-1. / 7.);                                                    // Shear term
    }

    sum6 += sumS;
    sum7 += sumS;

    sum[1] = sum0 * hh / 3.0 / (2 * M_PI * M_PI);
    sum[2] = sum1 * hh / 3.0 / (2 * M_PI * M_PI);
    sum[4] = sum2 * hh / 3.0 / (2 * M_PI * M_PI);
    sum[5] = sum3 * hh / 3.0 / (2 * M_PI * M_PI);
    sum[6] = sum4 * hh / 3.0 / (2 * M_PI * M_PI);
    sum[7] = sum5 * hh / 3.0 / (2 * M_PI * M_PI);
    sum[8] = sum6 * hh / 3.0 / (2 * M_PI * M_PI);
    sum[9] = sum7 * hh / 3.0 / (2 * M_PI * M_PI);
    sum[10] = sum8 * hh / 3.0 / (2 * M_PI * M_PI);
    sum[12] = sum9 * hh / 3.0 / (2 * M_PI * M_PI);
    sum[13] = sum10 * hh / 3.0 / (2 * M_PI * M_PI);
    sum[14] = sum11 * hh / 3.0 / (2 * M_PI * M_PI);
    sum[15] = sum12 * hh / 3.0 / (2 * M_PI * M_PI);

    // Now tabulate the pieces going as Plin.
    sum0 = sum1 = sum2 = 0;

#pragma omp parallel for reduction(+ : sum0, sum1, sum2)
    for (int i = 1; i < Nint; ++i) {
        double xx = qmin + i * hh;
        double kk = exp(xx);
        double k2 = kk * kk;
        double kq = kk * q;
        double pk = P_L(kk);
        double j0 = func_bessel_0(kq);
        double j1 = func_bessel_1(kq);
        int wt = 2 + 2 * (i % 2);

        sum0 += kk * wt * pk * (2. / 3. - 2 * j1 / kq);  // X^{(11)}
        sum1 += kk * wt * pk * (-2. * j0 + 6 * j1 / kq); // Y^{(11)}
        sum2 += k2 * wt * pk * (-j1);                    // U^{(1)}
    }

    sum[0] = sum0 * hh / 3.0 / (2 * M_PI * M_PI);
    sum[3] = sum1 * hh / 3.0 / (2 * M_PI * M_PI);
    sum[11] = sum2 * hh / 3.0 / (2 * M_PI * M_PI);

    return;
}

void compute_and_interpolate_qfuncs(int dofast) {
    unsigned int i;
    double q;
    const double qmax = 2000;        // Maximum q to compute the q functions, 2000 by default
    const unsigned int nbins = 3001; // Number of steps
    const double log_qmax = log10(qmax);
    const double dq = log_qmax / (nbins - 2);

    printf("getting qfunctions...\n");
    double q_array[nbins], xil[nbins], X_x11[nbins], X_x13[nbins], X_x22[nbins], X_x1012[nbins], Y_y11[nbins], Y_y13[nbins], Y_y22[nbins], Y_y1012[nbins],
        U_u1[nbins], U_u3[nbins], U_u11[nbins], U_u20[nbins], W_v1[nbins], W_v3[nbins], W_t[nbins], W_v10[nbins];
    q_array[0] = xil[0] = X_x11[0] = X_x13[0] = X_x22[0] = X_x1012[0] = Y_y11[0] = Y_y13[0] = Y_y22[0] = Y_y1012[0] = U_u1[0] = U_u3[0] = U_u11[0] = U_u20[0] = W_v1[0] = W_v3[0] = W_t[0] = W_v10[0] = 0.;

    // Reset logq in the beginning
    for (i = 1; i < nbins; i++) {
        q = pow(10, i * dq);
        q_array[i] = q;
        xil[i] = Xi_L(q);
        if (dofast) {
            double funcs[16];
            qfuncs(q, funcs);
            X_x11[i] = funcs[0];
            X_x22[i] = funcs[1];
            X_x13[i] = funcs[2];
            Y_y11[i] = funcs[3];
            Y_y22[i] = funcs[4];
            Y_y13[i] = funcs[5];
            X_x1012[i] = funcs[6];
            Y_y1012[i] = funcs[7];
            W_v1[i] = funcs[8];
            W_v3[i] = funcs[9];
            W_t[i] = funcs[10];
            U_u1[i] = funcs[11];
            U_u3[i] = funcs[12];
            U_u20[i] = funcs[13];
            U_u11[i] = funcs[14];
            W_v10[i] = funcs[15];
        } else {
            X_x11[i] = X_11(q);
            printf("X11 \n");
            X_x22[i] = X_22(q);
            printf("X22 \n");
            X_x13[i] = X_13(q);
            printf("X13 \n");
            Y_y11[i] = Y_11(q);
            printf("Y11 \n");
            Y_y22[i] = Y_22(q);
            printf("Y22 \n");
            Y_y13[i] = Y_13(q);
            printf("Y13 \n");
            X_x1012[i] = X_10_12(q);
            printf("X1012 \n");
            Y_y1012[i] = Y_10_12(q);
            printf("Y1012 \n");
            W_v1[i] = V1_112(q);
            printf("V1112 \n");
            W_v3[i] = V3_112(q);
            printf("V3112 \n");
            W_t[i] = T_112(q);
            printf("T112 \n");
            U_u1[i] = U_1(q);
            printf("U1 \n");
            U_u3[i] = U_3(q);
            printf("U3 \n");
            U_u20[i] = U_20(q);
            printf("U20 \n");
            U_u11[i] = U_11(q);
            printf("U11 \n");
            W_v10[i] = V_10(q);
        }
    }

    acc[8] = gsl_interp_accel_alloc();
    spline[8] = gsl_spline_alloc(gsl_interp_cspline, nbins);
    gsl_spline_init(spline[8], q_array, xil, nbins);
    acc[9] = gsl_interp_accel_alloc();
    spline[9] = gsl_spline_alloc(gsl_interp_cspline, nbins);
    gsl_spline_init(spline[9], q_array, U_u1, nbins);
    acc[10] = gsl_interp_accel_alloc();
    spline[10] = gsl_spline_alloc(gsl_interp_cspline, nbins);
    gsl_spline_init(spline[10], q_array, U_u3, nbins);
    acc[11] = gsl_interp_accel_alloc();
    spline[11] = gsl_spline_alloc(gsl_interp_cspline, nbins);
    gsl_spline_init(spline[11], q_array, U_u11, nbins);
    acc[12] = gsl_interp_accel_alloc();
    spline[12] = gsl_spline_alloc(gsl_interp_cspline, nbins);
    gsl_spline_init(spline[12], q_array, U_u20, nbins);
    acc[13] = gsl_interp_accel_alloc();
    spline[13] = gsl_spline_alloc(gsl_interp_cspline, nbins);
    gsl_spline_init(spline[13], q_array, X_x11, nbins);
    acc[14] = gsl_interp_accel_alloc();
    spline[14] = gsl_spline_alloc(gsl_interp_cspline, nbins);
    gsl_spline_init(spline[14], q_array, X_x13, nbins);
    acc[15] = gsl_interp_accel_alloc();
    spline[15] = gsl_spline_alloc(gsl_interp_cspline, nbins);
    gsl_spline_init(spline[15], q_array, X_x22, nbins);
    acc[16] = gsl_interp_accel_alloc();
    spline[16] = gsl_spline_alloc(gsl_interp_cspline, nbins);
    gsl_spline_init(spline[16], q_array, X_x1012, nbins);
    acc[17] = gsl_interp_accel_alloc();
    spline[17] = gsl_spline_alloc(gsl_interp_cspline, nbins);
    gsl_spline_init(spline[17], q_array, Y_y11, nbins);
    acc[18] = gsl_interp_accel_alloc();
    spline[18] = gsl_spline_alloc(gsl_interp_cspline, nbins);
    gsl_spline_init(spline[18], q_array, Y_y13, nbins);
    acc[19] = gsl_interp_accel_alloc();
    spline[19] = gsl_spline_alloc(gsl_interp_cspline, nbins);
    gsl_spline_init(spline[19], q_array, Y_y22, nbins);
    acc[20] = gsl_interp_accel_alloc();
    spline[20] = gsl_spline_alloc(gsl_interp_cspline, nbins);
    gsl_spline_init(spline[20], q_array, Y_y1012, nbins);
    acc[21] = gsl_interp_accel_alloc();
    spline[21] = gsl_spline_alloc(gsl_interp_cspline, nbins);
    gsl_spline_init(spline[21], q_array, W_v1, nbins);
    acc[22] = gsl_interp_accel_alloc();
    spline[22] = gsl_spline_alloc(gsl_interp_cspline, nbins);
    gsl_spline_init(spline[22], q_array, W_v3, nbins);
    acc[23] = gsl_interp_accel_alloc();
    spline[23] = gsl_spline_alloc(gsl_interp_cspline, nbins);
    gsl_spline_init(spline[23], q_array, W_t, nbins);
    acc[25] = gsl_interp_accel_alloc();
    spline[25] = gsl_spline_alloc(gsl_interp_cspline, nbins);
    gsl_spline_init(spline[25], q_array, W_v10, nbins);

    return;
}

/************************************************************************************************************************************************************************/
/********Write all components ******************************************************************************************************************************/

void initialize_CLEFT(char pk_filename[]) {
    // Get power spectrum
    int nLines = 0;
    int nHeader = 0;
    FILE *ps;

    // Count number of lines
    ps = fopen(pk_filename, "r");
    int fcounts[2];
    if (ps != 0) {
        compte(ps, fcounts);
        nLines = fcounts[0];
        nHeader = fcounts[1];
    } else {
        printf("Error opening power spectrum file.\n");
        exit(1);
    }

    // Read kmin and kmax
    rewind(ps);
    kmin = 1e30;
    kmax = 0;

    for (int i = 0; i < nHeader; i++)
        fscanf(ps, "%*[^\n]\n");
    for (int i = 0; i < nLines; i++) {
        double val;
        fscanf(ps, "%lf %*f\n", &val);
        if (val < kmin)
            kmin = val;
        if (val > kmax)
            kmax = val;
    }
    fclose(ps);
    // Add buffer zone for P(k) integrals
    kmin_integ = kmin + EPS;
    kmax_integ = kmax - EPS;

    interpole(1, pk_filename, nLines, nHeader);

    // Compute components
    compute_and_interpolate_R1();
    compute_and_interpolate_R2();
    compute_and_interpolate_Qn();
    compute_and_interpolate_qfuncs(0);
    compute_and_interpolate_jfuncs();
    write_ingredients(pk_filename);

    // Finalize

    for (int i = 0; i < 34; i++) {
        gsl_spline_free(spline[i]);
        gsl_interp_accel_free(acc[i]);
    }
}

/************************************************************************************************************************************************************************/
/**********\\MAIN FUNCTION\\***************************************************************************************************************************************/

int main(int argc, char *argv[]) {
    double start_time, end_time;
    start_time = omp_get_wtime();

    if (argv[2] == NULL) {
        do_CLEFT = true; // By default, compute CLEFT ingredients
    } else {
        int mode = strtol(argv[2], NULL, 10);
        if (mode == 0) {
            do_zeldovich = true;
            printf("Computes the Zel'dovich approximation\n");
        } else if (mode == 1) {
            do_CLPT = true;
            printf("Computes the CLPT model\n");
        } else if (mode == 2) {
            do_CLEFT = true;
            printf("Computes the CLEFT model\n");
        } else {
            printf("Error, mode argument should be '0', '1' or '2', currently: %s as integer %d\n", argv[2], mode);
            exit(-1);
        }
    }
    initialize_CLEFT(argv[1]);

    end_time = omp_get_wtime();
    printf("Total time needed %f minutes\n", (end_time - start_time) / 60.);
    return 0;
}
