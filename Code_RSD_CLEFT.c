/* ============================================================ *
 * AUTHOR:       Sylvain de la Torre		                *
 * CONTRIBUTORS: Antoine Rocher, Michel-Andrès Breton		*
 * 							        *
 * Code for computing CLEFT predictions based on                *
 * Vlah, Castorina & White 2016                                 *
 * ============================================================ */

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

const int r_max = 250;								// Maximum scale for CLEFT (see Code_RSD_CLPT.c)
const size_t size = 100;							// workspace size for cquad
const double xpivot_UYXi = 100;					                // argument in Bessel function where we start to use approximate form (and qawo integration) for Xi_L, U and Y functions
const double xpivot_TV = 600;							// argument in Bessel function where we start to use approximate form (and qawo integration) for T and V functions
const double dmu_M0 = 0.001;							// Integration step for M0 function
const double dmu_M1 = 0.000625;					                // Integration step for M1 function
const double dmu_M2 = 0.001;                                                    // Integration step for M2 function
double 	kmin;   								// minimum k in the power spectrum file (defined in main function)
double  kmax; 									// minimum k in the power spectrum file (defined in main function)
double  q_v[3];									// unit vector q_i, q_j, q_k

struct my_f_params2 {double a; double b;}; 
struct my_f_params3 {double a; double b; double c;}; 
struct my_f_params4 {double a; double b; double c; double d;};

gsl_interp_accel *acc[34];				
gsl_spline *spline[34];

/************************************************************************************************************************************************************************/
/*******************\\ Gauss-Legendre integral quadrature\\***********************************************************************************************************/

static const double x[] = {
    1.56289844215430828714e-02,    4.68716824215916316162e-02,
    7.80685828134366366918e-02,    1.09189203580061115002e-01,
    1.40203137236113973212e-01,    1.71080080538603274883e-01,
    2.01789864095735997236e-01,    2.32302481844973969643e-01,
    2.62588120371503479163e-01,    2.92617188038471964730e-01,
    3.22360343900529151720e-01,    3.51788526372421720979e-01,
    3.80872981624629956772e-01,    4.09585291678301542532e-01,
    4.37897402172031513100e-01,    4.65781649773358042251e-01,
    4.93210789208190933576e-01,    5.20158019881763056670e-01,
    5.46597012065094167460e-01,    5.72501932621381191292e-01,
    5.97847470247178721259e-01,    6.22608860203707771585e-01,
    6.46761908514129279840e-01,    6.70283015603141015784e-01,
    6.93149199355801965946e-01,    7.15338117573056446485e-01,
    7.36828089802020705530e-01,    7.57598118519707176062e-01,
    7.77627909649495475605e-01,    7.96897892390314476375e-01,
    8.15389238339176254384e-01,    8.33083879888400823522e-01,
    8.49964527879591284320e-01,    8.66014688497164623416e-01,
    8.81218679385018415547e-01,    8.95561644970726986709e-01,
    9.09029570982529690453e-01,    9.21609298145333952679e-01,
    9.33288535043079545942e-01,    9.44055870136255977955e-01,
    9.53900782925491742847e-01,    9.62813654255815527284e-01,
    9.70785775763706331929e-01,    9.77809358486918288561e-01,
    9.83877540706057015509e-01,    9.88984395242991747997e-01,
    9.93124937037443459632e-01,    9.96295134733125149166e-01,
    9.98491950639595818382e-01,    9.99713726773441233703e-01
};

static const double A[] = {
    3.12554234538633569472e-02,    3.12248842548493577326e-02,
    3.11638356962099067834e-02,    3.10723374275665165874e-02,
    3.09504788504909882337e-02,    3.07983790311525904274e-02,
    3.06161865839804484966e-02,    3.04040795264548200160e-02,
    3.01622651051691449196e-02,    2.98909795933328309169e-02,
    2.95904880599126425122e-02,    2.92610841106382766198e-02,
    2.89030896011252031353e-02,    2.85168543223950979908e-02,
    2.81027556591011733175e-02,    2.76611982207923882944e-02,
    2.71926134465768801373e-02,    2.66974591835709626611e-02,
    2.61762192395456763420e-02,    2.56294029102081160751e-02,
    2.50575444815795897034e-02,    2.44612027079570527207e-02,
    2.38409602659682059633e-02,    2.31974231852541216230e-02,
    2.25312202563362727021e-02,    2.18430024162473863146e-02,
    2.11334421125276415432e-02,    2.04032326462094327666e-02,
    1.96530874944353058650e-02,    1.88837396133749045537e-02,
    1.80959407221281166640e-02,    1.72904605683235824399e-02,
    1.64680861761452126430e-02,    1.56296210775460027242e-02,
    1.47758845274413017686e-02,    1.39077107037187726882e-02,
    1.30259478929715422855e-02,    1.21314576629794974079e-02,
    1.12251140231859771176e-02,    1.03078025748689695861e-02,
    9.38041965369445795116e-03,    8.44387146966897140266e-03,
    7.49907325546471157895e-03,    6.54694845084532276405e-03,
    5.58842800386551515727e-03,    4.62445006342211935096e-03,
    3.65596120132637518238e-03,    2.68392537155348241939e-03,
    1.70939265351810523958e-03,    7.34634490505671730396e-04
};

#define NUM_OF_POSITIVE_ZEROS  sizeof(x) / sizeof(double)
#define NUM_OF_ZEROS           NUM_OF_POSITIVE_ZEROS+NUM_OF_POSITIVE_ZEROS

double gl_int(double a, double b, double (*f)(double, void *), void *prms)
{
  double integral = 0.0; 
  double c = 0.5 * (b - a);
  double d = 0.5 * (b + a);
  double dum;
  const double *px = &x[NUM_OF_POSITIVE_ZEROS - 1];
  const double *pA = &A[NUM_OF_POSITIVE_ZEROS - 1];
  for (; px >= x; pA--, px--) {
    dum = c * *px;
    integral += *pA * ( (*f)(d - dum,prms) + (*f)(d + dum,prms) );
  }
  return c * integral;
}

void gl_int6(double a, double b, void (*f)(double, double, double []), void *prms,double result[], int n)
{
  int i;
  double integral[n];
  for (i=0; i<n; i++)
    integral[i] = 0;	   		 
  double c = 0.5 * (b - a);
  double d = 0.5 * (b + a);
  double dum;
  const double *px = &x[NUM_OF_POSITIVE_ZEROS - 1];
  const double *pA = &A[NUM_OF_POSITIVE_ZEROS - 1];
  double outi[n], outs[n];
  double R=* (double *)prms;
  
  for (; px >= x; pA--, px--) {
    dum = c * *px;
    (*f)(d - dum,R,outi);
    (*f)(d + dum,R,outs);
    for (i=0; i<n; i++)
      integral[i] += *pA * (outi[i] + outs[i]);
  }
  for (i=0; i<n; i++)
    result[i] = c*integral[i];
}

/************************************************************************************************************************************************************************/
/****************************\\gsl_cquad integration \\*****************************************************************************************************************/

double int_cquad (double func(double, void*), double alpha) 
{
  double result, error;  
  gsl_function F;
  F.function = func; 
  F.params = &alpha;		
   
  gsl_integration_cquad_workspace *w = gsl_integration_cquad_workspace_alloc(size);
  gsl_integration_cquad(&F, kmin, kmax, 0, 1e-9, w, &result, &error, NULL);
  gsl_integration_cquad_workspace_free(w);
  return result;   
}

double int_cquad_pivot_UYXi (double func(double, void*), double alpha) 
{
  double result, error; 
  double q = alpha;                   
  gsl_function F;
  F.function = func; 
  F.params = &alpha;				
  
  gsl_integration_cquad_workspace *w = gsl_integration_cquad_workspace_alloc(size);
  gsl_integration_cquad(&F, kmin, xpivot_UYXi/q, 0, 1e-9, w, &result, &error, NULL);
  gsl_integration_cquad_workspace_free(w);
  return result;
}

double int_cquad_pivot_TV (double func(double, void*), double alpha) 
{
  double result, error; 
  double q = alpha;                   
  gsl_function F;
  F.function = func; 
  F.params = &alpha;				
  
  gsl_integration_cquad_workspace *w = gsl_integration_cquad_workspace_alloc(size);
  gsl_integration_cquad(&F, kmin, xpivot_TV/q, 0, 1e-9, w, &result, &error, NULL);
  gsl_integration_cquad_workspace_free(w);
  return result;
}

/************************************************************************************************************************************************************************/
/****************************\\Bessel function\\************************************************************************************************************************/

double func_bessel_0(double x) {return sin(x)/x;}
double func_bessel_1(double x) {return sin(x)/(x*x)-cos(x)/x;}
double func_bessel_2(double x) {return (3./(x*x)-1.)*sin(x)/x - 3.*cos(x)/(x*x);}
double func_bessel_3(double x) {return (15./(x*x*x)-6./x)*sin(x)/x-(15./(x*x)-1.)*cos(x)/x;}

/************************************************************************************************************************************************************************/
/********************************\\ Interpolation Function \\***********************************************************************************************************/

double P_L (double k)
{
    if (k >= kmin && k <= kmax)  return gsl_spline_eval (spline[1], k, acc[1]);
    else return 0;
}

double R_1(double k)
{
    if (k >= kmin && k <= kmax)  return gsl_spline_eval (spline[2], k, acc[2]);
    else return 0;
}

double R_2(double k)
{
    if (k >= kmin && k <= kmax)  return gsl_spline_eval (spline[3], k, acc[3]);
    else return 0;
}

double Q_1(double k)
{
    if (k >= kmin && k <= kmax)  return gsl_spline_eval (spline[4], k, acc[4]);
    else return 0;
}

double Q_2(double k)
{
    if (k >= kmin && k <= kmax)  return gsl_spline_eval (spline[5], k, acc[5]);
    else return 0;
}

double Q_5(double k)
{
    if (k >= kmin && k <= kmax)  return gsl_spline_eval (spline[6], k, acc[6]);
    else return 0;
}

double Q_8(double k)
{
    if (k >= kmin && k <= kmax)  return gsl_spline_eval (spline[7], k, acc[7]);
    else return 0;
}

double iXi_L(double q)
{
    if (q>0 && q<=2000)  return gsl_spline_eval (spline[8], q, acc[8]);
    else return 0;
}

double iU_1(double q)
{
    if (q>0 && q<=2000)  return gsl_spline_eval (spline[9], q, acc[9]);
    else return 0;
}

double iU_3(double q)
{
    if (q>0 && q<=2000)  return gsl_spline_eval (spline[10], q, acc[10]);
    else return 0;
}

double iU_11(double q)
{
    if (q>0 && q<=2000)  return gsl_spline_eval (spline[11], q, acc[11]);
    else return 0;
}

double iU_20(double q)
{
    if (q>0 && q<=2000)  return gsl_spline_eval (spline[12], q, acc[12]);
    else return 0;
}

double iX_11(double q)
{
    if (q>0 && q<=2000)  return gsl_spline_eval (spline[13], q, acc[13]);
    else return 0;
}

double iX_13(double q)
{
    if (q>0 && q<=2000)  return gsl_spline_eval (spline[14], q, acc[14]);
    else return 0;
}

double iX_22(double q)
{
    if (q>0 && q<=2000)  return gsl_spline_eval (spline[15], q, acc[15]);
    else return 0;
}

double iX_10_12(double q)
{
    if (q>0 && q<=2000)  return gsl_spline_eval (spline[16], q, acc[16]);
    else return 0;
}

double iY_11(double q)
{
    if (q>0 && q<=2000)  return gsl_spline_eval (spline[17], q, acc[17]);
    else return 0;
}

double iY_13(double q)
{
    if (q>0 && q<=2000)  return gsl_spline_eval (spline[18], q, acc[18]);
    else return 0;
}

double iY_22(double q)
{
    if (q>0 && q<=2000)  return gsl_spline_eval (spline[19], q, acc[19]);
    else return 0;
}

double iY_10_12(double q)
{
    if (q>0 && q<=2000)  return gsl_spline_eval (spline[20], q, acc[20]);
    else return 0;
}

double iW_V1(double q)
{
    if (q>0 && q<=2000)  return gsl_spline_eval (spline[21], q, acc[21]);
    else return 0;
}

double iW_V3(double q)
{
    if (q>0 && q<=2000)  return gsl_spline_eval (spline[22], q, acc[22]);
    else return 0;
}

double iW_T(double q)
{
    if (q>0 && q<=2000)  return gsl_spline_eval (spline[23], q, acc[23]);
    else return 0;
}

double Q_s(double k)
{
    if (k >= kmin && k <= kmax)  return gsl_spline_eval (spline[24], k, acc[24]);
    else return 0;
}

double iV_10(double q)
{
    if (q>0 && q<=2000)  return gsl_spline_eval (spline[25], q, acc[25]);
    else return 0;
}

double iB_1(double q)
{
    if (q>0 && q<=2000)  return gsl_spline_eval (spline[26], q, acc[26]);
    else return 0;
}

double iB_2(double q)
{
    if (q>0 && q<=2000)  return gsl_spline_eval (spline[27], q, acc[27]);
    else return 0;
}

double iJ_2(double q)
{
    if (q>0 && q<=2000)  return gsl_spline_eval (spline[28], q, acc[28]);
    else return 0;
}

double iJ_3(double q)
{
    if (q>0 && q<=2000)  return gsl_spline_eval (spline[29], q, acc[29]);
    else return 0;
}

double iJ_4(double q)
{
    if (q>0 && q<=2000)  return gsl_spline_eval (spline[30], q, acc[30]);
    else return 0;
}

double iV_12(double q)
{
    if (q>0 && q<=2000)  return gsl_spline_eval (spline[31], q, acc[31]);
    else return 0;
}

double iChi_12(double q)
{
    if (q>0 && q<=2000)  return gsl_spline_eval (spline[32], q, acc[32]);
    else return 0;
}

double iZeta(double q)
{
    if (q>0 && q<=2000)  return gsl_spline_eval (spline[33], q, acc[33]);
    else return 0;
}

/************************************************************************************************************************************************************************/
/*****************\\delta kronecker\\***********************************************************************************************************************************/

int delta_K (int i, int j)
{
  if (i == j) return 1;
  else return 0;
}

/************************************************************************************************************************************************************************/
/****************************************\\ R_n FUNCTIONS \\*********************************************************************************************************/

// R_1_tilde function
double fRt1 (double x, void *p)
{	
  double r = *(double *) p; 
  return (r*r*(1.-x*x)*(1.-x*x))/(1.+r*r-2.*x*r);
}

double Rt_1 (double r)
{
  return gl_int(-1, 1, &fRt1, &r);
}

// R_2_tilde function
double fRt2 (double x, void *p)
{
  double r = *(double *) p; 
  return ((1.-x*x)*r*x*(1.-x*r))/(1.+r*r-2.*r*x);	
}

double Rt_2 (double r)
{
  return gl_int(-1, 1, &fRt2, &r);
}

// R1 function
double fR_1 (double r, void *p)
{
  double k = *(double *) p;
  return P_L(k*r)*Rt_1(r);
}

double R1(double k)
{
  double error;     
  double result1 = 0;
  double result2 = 0;
  gsl_function F;
  gsl_integration_workspace *w = gsl_integration_workspace_alloc(200);

  F.function = &fR_1; 
  F.params = &k;
  
  gsl_integration_qag(&F, 0, 1., 0, 1e-3, 200, 6, w, &result1, &error);
  if (kmax/k > 1 + 1e-8) gsl_integration_qag(&F, 1., kmax/k, 0, 1e-3, 200, 6, w, &result2, &error);   // Kmax/k car k*r<=! kmax sinon Pk = 0
  gsl_integration_workspace_free(w);
  
  return (k*k*k)/(4.*M_PI*M_PI)*P_L(k)*(result1+result2);		
}

// R2 function 
double fR_2 (double r, void *p)
{ 
  double k = *(double *) p; 
  return P_L(k*r)*Rt_2(r);
}

double R2(double k)
{
  double error;
  double result1 = 0;
  double result2 = 0;
  gsl_function F;
  F.function = &fR_2; 
  F.params = &k;
  
  gsl_integration_workspace *w = gsl_integration_workspace_alloc(200); 
  gsl_integration_qag(&F, 0, 1, 0, 1e-3, 200, 6, w, &result1, &error);
  if(kmax/k > 1 + 1e-8) gsl_integration_qag(&F, 1, kmax/k, 0, 1e-3, 200, 6, w, &result2, &error);
  gsl_integration_workspace_free(w);
  return (k*k*k)/(4.*M_PI*M_PI)*P_L(k)*(result1+result2);
}

void write_R(void)
{
  printf("getting R function...\n");
  FILE* f;
  const int nk = 11000; // Nombre de points
  const double log_kmin = log10(kmin);
  const double log_kmax = log10(kmax);
  const double dk = (log_kmax - log_kmin)/nk;

  f = fopen("data/func_R.dat","w+");
  
  double R1_array[nk+1];
  double R2_array[nk+1];
  for (int i = 0; i <= nk; i++) {
    double k = pow(10, log_kmin + i*dk);
    R1_array[i] = R1(k);
    R2_array[i] = R2(k);
    fprintf(f, "%.13le %.13le %.13le\n", k, R1_array[i], R2_array[i]);
  }
  fclose(f);
  return;
}

void write_R1(void)
{
  printf("getting R function 1/2...\n");
  FILE* f;
  const int nk = 11000; // Nombre de points
  const double log_kmin = log10(kmin);
  const double log_kmax = log10(kmax);
  const double dk = (log_kmax - log_kmin)/nk;
  
  f = fopen("data/func_R1.dat","w+");
  
  double R1_array[nk+1];
  for (int i = 0; i <= nk; i++){
    double k = pow(10, log_kmin + i*dk);
    R1_array[i] = R1(k);
    fprintf(f, "%.13le %.13le\n", k, R1_array[i]);
  }
  fclose(f);
  return;
}

void write_R2(void)
{
  printf("getting R function 2/2...\n");
  FILE* f;
  int nk = 11000; // Nombre de points
  double log_kmin = log10(kmin);
  double log_kmax = log10(kmax);
  double dk = (log_kmax - log_kmin)/nk;
  
  f = fopen("data/func_R2.dat","w+");
  
  double R2_array[nk+1];
  for (int i = 0; i <= nk; i++){
    double k = pow(10, log_kmin + i*dk);
    R2_array[i] = R2(k);
    fprintf(f, "%.13le %.13le\n", k, R2_array[i]);
  }
  fclose(f);
  return;
}	

/***********************************************************************************************************************************************************************/
/******************************\\FUNCTION Q\\***********************************************************************************************************************/

double Qt_1 (double r, double x)
{
  return (r*r*(1.-x*x)*(1.-x*x))/((1.+r*r-2.*x*r)*(1.+r*r-2.*x*r));
}

double Qt_2 (double r, double x)
{
  return ((1.-x*x)*r*x*(1.-x*r))/((1.+r*r-2.*r*x)*(1.+r*r-2.*r*x));
}

double Qt_5 (double r, double x)
{
  return (r*x*(1.-x*x))/(1.+r*r-2.*x*r);
}

double Qt_8 (double r, double x)
{
  return (r*r*(1.-x*x))/(1.+r*r-2.*x*r);
}

double Qt_s (double r, double x)
{
  return r*r*(x*x-1.)*(1.-2*r*r+4*r*x-3*x*x)/((1.+r*r-2.*x*r)*(1.+r*r-2.*x*r));
}

double lg_Q_n( double x, void *p)
{
  struct my_f_params3 * params = (struct my_f_params3 *)p;
  double k = (params->a);
  double r = (params->b);
  double n = (params->c);
  double y_p = 1.+r*r-2.*x*r;
  if  (y_p < 0)	return 0;
  else {
    if ((int)n == 1)	return P_L(k*sqrt(y_p))*Qt_1(r,x);
    if ((int)n == 2)	return P_L(k*sqrt(y_p))*Qt_2(r,x);
    if ((int)n == 5)	return P_L(k*sqrt(y_p))*Qt_5(r,x);
    if ((int)n == 8)	return P_L(k*sqrt(y_p))*Qt_8(r,x);
    if ((int)n == 9)	return P_L(k*sqrt(y_p))*Qt_s(r,x);
    else return -1;
  }
}

double fQ_n (double r, void *p)		
{
  struct my_f_params2 * params = (struct my_f_params2 *)p;
  double k = (params->a);
  double n = (params->b);
  struct my_f_params3 par = {k, r, n}; 
  return P_L(k*r)*gl_int(-1, 1, &lg_Q_n, &par);
}

double Q_n (int n, double k)					
{
  struct my_f_params2 p = {k, n}; 
  double error; 
  double result1 = 0;
  double result2 = 0;
  gsl_function F;
  gsl_integration_workspace *w = gsl_integration_workspace_alloc(200);
  
  F.function = &fQ_n; 
  F.params = &p;

  if (n==9) gsl_integration_qag(&F, 0., 1., 0, 5e-3, 200, 6, w, &result1, &error);
  else gsl_integration_qag(&F, 0., 1., 0, 1e-3, 200, 6, w, &result1, &error);
  
  if (kmax/k > 1) gsl_integration_qag(&F, 1., kmax/k, 0, 1e-2, 200, 6, w, &result2, &error);
  
  gsl_integration_workspace_free(w);
  
  return (k*k*k)/(4.*M_PI*M_PI)*(result1+result2);
}

void write_Qn(void)
{
  FILE* f;
  int n;
  const int nk = 3000; // Nombre de points
  const double log_kmin = log10(kmin);
  const double log_kmax = log10(kmax);
  const double dk = (log_kmax - log_kmin)/nk; 
  double Q1_array[nk+1], Q2_array[nk+1], Q5_array[nk+1], Q8_array[nk+1], Qs_array[nk+1];
  for (n=1; n<10; n++){
    printf("getting Q function %d/9...\n",n);
    switch (n){
    case 1:
      f = fopen("data/func_Q1.dat","w+");
      for (int i = 0; i <= nk; i++){
	double k = pow(10, log_kmin + i*dk);
	Q1_array[i] = Q_n(1, k);
	fprintf(f, "%.13le %.13le\n", k, Q1_array[i]);
      }
      fclose(f);
      break;
    case 2:
      f = fopen("data/func_Q2.dat","w+");
      for (int i = 0; i <= nk; i++){
	double k = pow(10, log_kmin + i*dk);
	Q2_array[i] = Q_n(2, k);
	fprintf(f, "%.13le %.13le\n", k, Q2_array[i]);
      }
      fclose(f);
      break;
    case 5:
      f = fopen("data/func_Q5.dat","w+");
      for (int i = 0; i <= nk; i++){
	double k = pow(10, log_kmin + i*dk);
	Q5_array[i] = Q_n(5, k);
	fprintf(f, "%.13le %.13le\n", k, Q5_array[i]);
      }
      fclose(f);
      break;
    case 8:
      f = fopen("data/func_Q8.dat","w+");
      for (int i = 0; i <= nk; i++){
	double k = pow(10, log_kmin + i*dk);
	Q8_array[i] = Q_n(8, k);
	fprintf(f, "%.13le %.13le\n", k, Q8_array[i]);
      }
      fclose(f);
      break;
    case 9:
      f = fopen("data/func_Qs.dat","w+");
      for (int i = 0; i <= nk; i++){
	double k = pow(10, log_kmin + i*dk);
	Qs_array[i] = Q_n(9, k);
	fprintf(f, "%.13le %.13le\n", k, Qs_array[i]);
      }
      fclose(f);
      break;
    default:	
      break;
    }
  }
  return;
}

/*******************************************************************************************************************************************************************/
/************************\\Compute Xi linear\\********************************************************************************************************************/
										
double fXi_L (double k, void * p)
{
    double q = *(double *) p;
    return P_L(k)*k*k*func_bessel_0(k*q);
}

double fXi_L_qawo (double k, void * p)
{
    double alpha = *(double *) p; 
    return alpha*P_L(k)*k;
}

double Xi_L (double q)
{
  if(kmax < xpivot_UYXi/(q+1e-8))
    return 1./(2.*M_PI*M_PI)*int_cquad(fXi_L, q); 
  else{
    double result1, result2, error;
    // First integration using cquad
    result1 = 1./(2.*M_PI*M_PI)*int_cquad_pivot_UYXi(fXi_L, q); 
    
    // Second integration with qawo
    double alpha = 1./(2.*M_PI*M_PI*q);
    gsl_function F;
    F.function = &fXi_L_qawo;
    F.params = &alpha;       
    
    gsl_integration_workspace *w1 = gsl_integration_workspace_alloc(200);
    gsl_integration_workspace *w2 = gsl_integration_workspace_alloc(200);    
    gsl_integration_qawo_table *t = gsl_integration_qawo_table_alloc(q, kmax-kmin, GSL_INTEG_SINE, size);
    gsl_integration_qawf(&F, xpivot_UYXi/q, 1e-5, 200, w1, w2, t, &result2, &error);
    
    gsl_integration_workspace_free(w1);
    gsl_integration_workspace_free(w2);
    gsl_integration_qawo_table_free(t);
    return result1+result2;
  }
}		

/*******************************************************************************************************************************************************************/
/**************\\U function\\**************************************************************************************************************************************/						
// U^1(q)														
double fU_1 (double k, void *p)
{
	double q = *(double *) p; 
	return -P_L(k)*k*func_bessel_1(k*q);
}

double fU_1_qawo (double k, void *p)
{
	double alpha = *(double *) p; 
	return alpha*P_L(k);
}

double U_1 (double q)
{
  if(kmax < xpivot_UYXi/(q+1e-8))
    return 1./(2.*M_PI*M_PI)*int_cquad(fU_1, q);  
  else{
    double result1, result2, error;
    // First integration using cquad
    result1 = 1./(2.*M_PI*M_PI)*int_cquad_pivot_UYXi(fU_1, q); 
    
    // Second integration with qawo
    double alpha = 1./(2.*M_PI*M_PI*q);
    gsl_function F;
    F.function = &fU_1_qawo;
    F.params = &alpha;       
    
    gsl_integration_workspace *w1 = gsl_integration_workspace_alloc(200);
    gsl_integration_workspace *w2 = gsl_integration_workspace_alloc(200);    
    gsl_integration_qawo_table *t = gsl_integration_qawo_table_alloc(q, kmax-kmin, GSL_INTEG_COSINE, size);
    gsl_integration_qawf(&F, xpivot_UYXi/q, 1e-4, 200, w1, w2, t, &result2, &error);
    
    gsl_integration_workspace_free(w1);
    gsl_integration_workspace_free(w2);
    gsl_integration_qawo_table_free(t);
    return result1+result2;
  }
}

// U^3(q)		
double fU_3 (double k, void *p)
{
  double q = *(double *) p; 
  return -5./21.*R_1(k)*k*func_bessel_1(k*q);
}

double fU_3_qawo (double k, void *p)
{
  double alpha = *(double *) p; 
  return +5./21.*alpha*R_1(k);
}

double U_3 (double q)
{
  if(kmax < xpivot_UYXi/(q+1e-8))
    return 1./(2.*M_PI*M_PI)*int_cquad(fU_3, q);
  else{
    double result1, result2, error;
    // First integration using cquad
    result1 = 1./(2.*M_PI*M_PI)*int_cquad_pivot_UYXi(fU_3, q); 
    
    // Second integration with qawo
    double alpha = 1./(2.*M_PI*M_PI*q);
    gsl_function F;
    F.function = &fU_3_qawo;
    F.params = &alpha;       
    
    gsl_integration_workspace *w1 = gsl_integration_workspace_alloc(200);
    gsl_integration_workspace *w2 = gsl_integration_workspace_alloc(200);    
    gsl_integration_qawo_table *t = gsl_integration_qawo_table_alloc(q, kmax-kmin, GSL_INTEG_COSINE, size);
    gsl_integration_qawf(&F, xpivot_UYXi/q, 1e-4, 200, w1, w2, t, &result2, &error);
    
    gsl_integration_workspace_free(w1);
    gsl_integration_workspace_free(w2);
    gsl_integration_qawo_table_free(t);
    return result1+result2;
  } 
}

// U_11(q)	
double fU_11 (double k, void *p)
{
  double q = *(double *) p; 
  return -6./7.*k*(R_1(k)+R_2(k))*func_bessel_1(k*q);	
}

double fU_11_qawo (double k, void *p)
{
  double alpha = *(double *) p; 
  return 6./7.*alpha*(R_1(k)+R_2(k));	
}

double U_11 (double q)
{
  if(kmax < xpivot_UYXi/(q+1e-8))
    return 1./(2.*M_PI*M_PI)*int_cquad(fU_11, q); 
  else{
    double result1, result2, error;
    // First integration using cquad
    result1 = 1./(2.*M_PI*M_PI)*int_cquad_pivot_UYXi(fU_11, q); 
    
    // Second integration with qawo
    double alpha = 1./(2.*M_PI*M_PI*q);
    gsl_function F;
    F.function = &fU_11_qawo;
    F.params = &alpha;       
    
    gsl_integration_workspace *w1 = gsl_integration_workspace_alloc(200);
    gsl_integration_workspace *w2 = gsl_integration_workspace_alloc(200);    
    gsl_integration_qawo_table *t = gsl_integration_qawo_table_alloc(q, kmax-kmin, GSL_INTEG_COSINE, size);
    gsl_integration_qawf(&F, xpivot_UYXi/q, 1e-4, 200, w1, w2, t, &result2, &error);
    
    gsl_integration_workspace_free(w1);
    gsl_integration_workspace_free(w2);
    gsl_integration_qawo_table_free(t);
    return result1+result2;
  } 
}

// U_20(q)	
double fU_20 (double k, void *p)
{
  double q = *(double *) p; 
  return -3./7.*k*Q_8(k)*func_bessel_1(k*q);
}

double fU_20_qawo (double k, void *p)
{
  double alpha = *(double *) p; 
  return 3./7.*alpha*Q_8(k);
}

double U_20 (double q)
{
  if(kmax < xpivot_UYXi/(q+1e-8)){
    return 1./(2.*M_PI*M_PI)*int_cquad(fU_20, q);
  }
  else{
    double result1, result2, error;
    // First integration using cquad
    result1 = 1./(2.*M_PI*M_PI)*int_cquad_pivot_UYXi(fU_20, q); 
    
    // Second integration with qawo
    double alpha = 1./(2.*M_PI*M_PI*q);
    gsl_function F;
    F.function = &fU_20_qawo;
    F.params = &alpha;       
    
    gsl_integration_workspace *w1 = gsl_integration_workspace_alloc(200);
    gsl_integration_workspace *w2 = gsl_integration_workspace_alloc(200);    
    gsl_integration_qawo_table *t = gsl_integration_qawo_table_alloc(q, kmax-kmin, GSL_INTEG_COSINE, size);
    gsl_integration_qawf(&F, xpivot_UYXi/q, 1e-4, 200, w1, w2, t, &result2, &error);
    
    gsl_integration_workspace_free(w1);
    gsl_integration_workspace_free(w2);
    gsl_integration_qawo_table_free(t);
    return result1+result2;
  }  
}	

//Ui(q)
double get_U(double q)
{
  return iU_1(q)+iU_3(q);
}

/*******************************************************************************************************************************************************************/
/*******************\\X FUNCTIONS\\*****************************************************************************************************************************/
// X^11(q)																			
double fX_11 (double k, void *p)
{
  double q = *(double *) p; 
  double x = k*q;
  return P_L(k)*(2./3.-2.*func_bessel_1(x)/x);
}																											

double X_11 (double q)
{
  return 1./(2.*M_PI*M_PI)*int_cquad(fX_11, q);  
}

// X^22(q)																						
double fX_22 (double k, void *p)
{
  double q = *(double *) p; 
  double x = k*q;
  return 9./98.*Q_1(k)*(2./3.-2.*func_bessel_1(x)/(x));
}				

double X_22 (double q)
{
  return 1./(2.*M_PI*M_PI)*int_cquad(fX_22, q); 
}

// X^13(q)
double fX_13 (double k, void *p)
{
  double q = *(double *) p; 
  double x = k*q;
  return 5./21.*R_1(k)*(2./3.-2.*func_bessel_1(x)/x);
}				

double X_13 (double q)
{
  return 1./(2.*M_PI*M_PI)*int_cquad(fX_13, q); 
}

// X_10^12(q)
double fX_10_12 (double k, void *p)
{
  double q = *(double *) p; 
  double x = k*q;
  return 1./14.*(2.*(R_1(k)-R_2(k))+3.*R_1(k)*sin(k*q)/(k*q)-3.*(3.*R_1(k)+4.*R_2(k)+2.*Q_5(k))*func_bessel_1(x)/x);
}

double X_10_12 (double q)
{
  return 1./(2.*M_PI*M_PI)*int_cquad(fX_10_12, q); 
}

/*******************************************************************************************************************************************************************/
/*******************Y FUNCTIONS*****************************************************************************************************************************/
// Y^11(q)
double fY_11 (double k, void *p)
{
  double q = *(double *) p; 
  double x = k*q;
  return P_L(k)*(-2.*sin(k*q)/(k*q)+6.*func_bessel_1(x)/x);
}

double fY_11_qawo_1_over_2 (double k, void *p)
{
  double alpha = *(double *) p; 
  return -2*alpha*P_L(k)/k;
}	

double fY_11_qawo_2_over_2 (double k, void *p)
{
  double alpha = *(double *) p; 
  return -alpha*P_L(k)*6/(k*k);
}					

double Y_11 (double q)
{
  if(kmax < xpivot_UYXi/(q+1e-8))
    return 1./(2.*M_PI*M_PI)*int_cquad(fY_11, q); 
  else{
    double result1, result2, result3, error;
    // First integration using cquad
    result1 = 1./(2.*M_PI*M_PI)*int_cquad_pivot_UYXi(fY_11, q); 
    
    // Second integration with qawo 1/2
    double alpha = 1./(2.*M_PI*M_PI*q);
    gsl_function F;
    F.function = &fY_11_qawo_1_over_2;
    F.params = &alpha;       
    
    gsl_integration_workspace *w1 = gsl_integration_workspace_alloc(200);
    gsl_integration_workspace *w2 = gsl_integration_workspace_alloc(200);    
    gsl_integration_qawo_table *t = gsl_integration_qawo_table_alloc(q, kmax-kmin, GSL_INTEG_SINE, size);
    gsl_integration_qawf(&F, xpivot_UYXi/q, 1e-5, 200, w1, w2, t, &result2, &error);
    
    // Second integration with qawo 2/2
    alpha = 1./(2.*M_PI*M_PI*q*q);
    F.function = &fY_11_qawo_2_over_2;
    F.params = &alpha;         
    gsl_integration_qawo_table *t2 = gsl_integration_qawo_table_alloc(q, kmax-kmin, GSL_INTEG_COSINE, size);
    gsl_integration_qawf(&F, xpivot_UYXi/q, 1e-5, 200, w1, w2, t2, &result3, &error);
    
    gsl_integration_workspace_free(w1);
    gsl_integration_workspace_free(w2);
    gsl_integration_qawo_table_free(t);
    gsl_integration_qawo_table_free(t2);
    return result1+result2+result3;
  }   
}

// Y^22(q)
double fY_22 (double k, void *p)
{	
  double q = *(double *) p; 
  double x = k*q;
  return 9./98.*Q_1(k)*(-2.*sin(k*q)/(k*q)+6.*func_bessel_1(x)/x);
}	

double fY_22_qawo_1_over_2 (double k, void *p)
{	
  double alpha = *(double *) p; 
  return 9./98.*alpha*Q_1(k)*(-2.)/k;
}

double fY_22_qawo_2_over_2 (double k, void *p)
{	
  double alpha = *(double *) p; 
  return -9./98.*alpha*Q_1(k)*6./(k*k);
}				

double Y_22 (double q) 
{
  if(kmax < xpivot_UYXi/(q+1e-8))
    return 1./(2.*M_PI*M_PI)*int_cquad(fY_22, q); 
  else{
    double result1, result2, result3, error;
    // First integration using cquad
    result1 = 1./(2.*M_PI*M_PI)*int_cquad_pivot_UYXi(fY_22, q); 
    
    // Second integration with qawo 1/2
    double alpha = 1./(2.*M_PI*M_PI*q);
    gsl_function F;
    F.function = &fY_22_qawo_1_over_2;
    F.params = &alpha;       
    
    gsl_integration_workspace *w1 = gsl_integration_workspace_alloc(200);
    gsl_integration_workspace *w2 = gsl_integration_workspace_alloc(200);    
    gsl_integration_qawo_table *t = gsl_integration_qawo_table_alloc(q, kmax-kmin, GSL_INTEG_SINE, size);
    gsl_integration_qawf(&F, xpivot_UYXi/q, 1e-5, 200, w1, w2, t, &result2, &error);
    
    // Second integration with qawo 2/2
    alpha = 1./(2.*M_PI*M_PI*q*q);
    F.function = &fY_22_qawo_2_over_2;
    F.params = &alpha;         
    gsl_integration_qawo_table *t2 = gsl_integration_qawo_table_alloc(q, kmax-kmin, GSL_INTEG_COSINE, size);
    gsl_integration_qawf(&F, xpivot_UYXi/q, 1e-5, 200, w1, w2, t2, &result3, &error);
    
    gsl_integration_workspace_free(w1);
    gsl_integration_workspace_free(w2);
    gsl_integration_qawo_table_free(t);
    gsl_integration_qawo_table_free(t2);
    return result1+result2+result3;
  }  
}

// Y^13(q)
double fY_13 (double k, void *p)
{
  double q = *(double *) p; 
  double x = k*q;
  return 5./21.*R_1(k)*(-2.*sin(k*q)/(k*q)+6.*func_bessel_1(x)/x);
}	

double fY_13_qawo_1_over_2 (double k, void *p)
{
  double alpha = *(double *) p; 
  return 5./21.*alpha*R_1(k)*(-2.)/k;
}

double fY_13_qawo_2_over_2 (double k, void *p)
{
  double alpha = *(double *) p; 
  return -5./21.*alpha*R_1(k)*6./(k*k);
}					

double Y_13 (double q)
{		
  if(kmax < xpivot_UYXi/(q+1e-8))   	
    return 1./(2.*M_PI*M_PI)*int_cquad(fY_13, q); 
  else{
    double result1, result2, result3, error;
    // First integration using cquad
    result1 = 1./(2.*M_PI*M_PI)*int_cquad_pivot_UYXi(fY_13, q); 
    
    // Second integration with qawo 1/2
    double alpha = 1./(2.*M_PI*M_PI*q);
    gsl_function F;
    F.function = &fY_13_qawo_1_over_2;
    F.params = &alpha;       
    
    gsl_integration_workspace *w1 = gsl_integration_workspace_alloc(200);
    gsl_integration_workspace *w2 = gsl_integration_workspace_alloc(200);    
    gsl_integration_qawo_table *t = gsl_integration_qawo_table_alloc(q, kmax-kmin, GSL_INTEG_SINE, size);
    gsl_integration_qawf(&F, xpivot_UYXi/q, 1e-5, 200, w1, w2, t, &result2, &error);
    
    // Second integration with qawo 2/2
    alpha = 1./(2.*M_PI*M_PI*q*q);
    F.function = &fY_13_qawo_2_over_2;
    F.params = &alpha;           
    gsl_integration_qawo_table *t2 = gsl_integration_qawo_table_alloc(q, kmax-kmin, GSL_INTEG_COSINE, size);
    gsl_integration_qawf(&F, xpivot_UYXi/q, 1e-5, 200, w1, w2, t2, &result3, &error);
    
    gsl_integration_workspace_free(w1);
    gsl_integration_workspace_free(w2);
    gsl_integration_qawo_table_free(t);
    gsl_integration_qawo_table_free(t2);
    return result1+result2+result3;
  } 
}

// Y^10_12(q)
double fY_10_12 (double k, void *p)
{
  double q = *(double *) p; 
  double x = q*k;
  return -3./14.*(3.*R_1(k)+4.*R_2(k)+2.*Q_5(k))*(sin(k*q)/(k*q)-3.*func_bessel_1(x)/x);
}

double fY_10_12_qawo_1_over_2 (double k, void *p)
{
  double alpha = *(double *) p; 
  return -3./14.*alpha*(3.*R_1(k)+4.*R_2(k)+2.*Q_5(k))/k;
}

double fY_10_12_qawo_2_over_2 (double k, void *p)
{
  double alpha = *(double *) p; 
  return 3./14.*alpha*(3.*R_1(k)+4.*R_2(k)+2.*Q_5(k))*(-3.)/(k*k);
}

double Y_10_12 (double q)
{
  if(kmax < xpivot_UYXi/(q+1e-8))   	
    return 1./(2.*M_PI*M_PI)*int_cquad(fY_10_12, q); 
  else{
    double result1, result2, result3, error;
    // First integration using cquad
    result1 = 1./(2.*M_PI*M_PI)*int_cquad_pivot_UYXi(fY_10_12, q); 
    
    // Second integration with qawo 1/2
    double alpha = 1./(2.*M_PI*M_PI*q);
    gsl_function F;
    F.function = &fY_10_12_qawo_1_over_2;
    F.params = &alpha;       
    
    gsl_integration_workspace *w1 = gsl_integration_workspace_alloc(200);
    gsl_integration_workspace *w2 = gsl_integration_workspace_alloc(200);    
    gsl_integration_qawo_table *t = gsl_integration_qawo_table_alloc(q, kmax-kmin, GSL_INTEG_SINE, size);
    gsl_integration_qawf(&F, xpivot_UYXi/q, 1e-5, 200, w1, w2, t, &result2, &error);
    
    // Second integration with qawo 2/2
    alpha = 1./(2.*M_PI*M_PI*q*q);
    F.function = &fY_10_12_qawo_2_over_2;
    F.params = &alpha;         
    gsl_integration_qawo_table *t2 = gsl_integration_qawo_table_alloc(q, kmax-kmin, GSL_INTEG_COSINE, size);
    gsl_integration_qawf(&F, xpivot_UYXi/q, 1e-5, 200, w1, w2, t2, &result3, &error);
    
    gsl_integration_workspace_free(w1);
    gsl_integration_workspace_free(w2);
    gsl_integration_qawo_table_free(t);
    gsl_integration_qawo_table_free(t2);
    return result1+result2+result3;
  } 
}

/************************************************************************************************************************************************************************/
/**************************\\V+S et T function\\************************************************************************************************************************/
// V^112_1(q) + S^112(q)

double fV1_112 (double k, void*p)
{
  double q = *(double *) p; 
  double x = k*q;
  return -3./7.*R_1(k)*func_bessel_1(x)/k + 3./(7.*k)*(2.*R_1(k)+4.*R_2(k)+Q_1(k)+2.*Q_2(k))*func_bessel_2(x)/x; 
}

double fV1_112_qawo_1_over_2 (double k, void*p)
{
  double alpha = *(double *) p; 
  return 3./7.*alpha*R_1(k)/(k*k); 
}	

double fV1_112_qawo_2_over_2 (double k, void*p)
{
  double alpha = *(double *) p; 
  return -3./(7.*k)*alpha*(2.*R_1(k)+4.*R_2(k)+Q_1(k)+2.*Q_2(k))/(k*k); 
}							

double V1_112 (double q)
{
  if(q < 3e-8/kmin) 
    return 0;
  else if(kmax < xpivot_TV/q){   
    return 1./(2.*M_PI*M_PI)*int_cquad(fV1_112, q); 
  }
  else{
    double result1, result2, result3, error;
    // First integration using cquad
    result1 = 1./(2.*M_PI*M_PI)*int_cquad_pivot_TV(fV1_112, q); 
    
    // Second integration with qawo 1/2
    double alpha = 1./(2.*M_PI*M_PI*q*q);
    gsl_function F;
    F.function = &fV1_112_qawo_1_over_2;
    F.params = &alpha;       
    
    gsl_integration_workspace *w1 = gsl_integration_workspace_alloc(200);
    gsl_integration_workspace *w2 = gsl_integration_workspace_alloc(200);    
    gsl_integration_qawo_table *t = gsl_integration_qawo_table_alloc(q, kmax-kmin, GSL_INTEG_COSINE, size);
    gsl_integration_qawf(&F, xpivot_TV/q, 1e-6, 200, w1, w2, t, &result2, &error);
    
    // Second integration with qawo 2/2
    F.function = &fV1_112_qawo_2_over_2;
    F.params = &alpha;          
    gsl_integration_qawo_table *t2 = gsl_integration_qawo_table_alloc(q, kmax-kmin, GSL_INTEG_SINE, size);
    gsl_integration_qawf(&F, xpivot_TV/q, 1e-6, 200, w1, w2, t2, &result3, &error);
    
    gsl_integration_workspace_free(w1);
    gsl_integration_workspace_free(w2);
    gsl_integration_qawo_table_free(t);
    gsl_integration_qawo_table_free(t2);
    return result1+result2+result3;
  }
}

// V^112_3(q)+ S^112	(q)

double fV3_112 (double k, void *p)
{
  double q = *(double *) p; 
  double x = k*q;
  return -3./7.*Q_1(k)*func_bessel_1(x)/k + 3./(7.*k)*(2.*R_1(k)+4.*R_2(k)+Q_1(k)+2.*Q_2(k))*func_bessel_2(x)/x;
}

double fV3_112_qawo_1_over_2 (double k, void *p)
{
  double alpha = *(double *) p; 
  return 3./7.*alpha*Q_1(k)/(k*k);
}

double fV3_112_qawo_2_over_2 (double k, void *p)
{
  double alpha = *(double *) p; 
  return -3./(7.*k)*alpha*(2.*R_1(k)+4.*R_2(k)+Q_1(k)+2.*Q_2(k))/(k*k);
}

// V^112_3+S^112	

double V3_112 (double q)
{
  if(q < 3e-8/kmin) 
    return 0;
  else if(kmax < xpivot_TV/q){   
    return 1./(2.*M_PI*M_PI)*int_cquad(fV3_112, q);
  }
  else{
    double result1, result2, result3, error;
    // First integration using cquad
    result1 = 1./(2.*M_PI*M_PI)*int_cquad_pivot_TV(fV3_112, q); 
    
    // Second integration with qawo 1/2
    double alpha = 1./(2.*M_PI*M_PI*q*q);
    gsl_function F;
    F.function = &fV3_112_qawo_1_over_2;
    F.params = &alpha;       
    
    gsl_integration_workspace *w1 = gsl_integration_workspace_alloc(200);
    gsl_integration_workspace *w2 = gsl_integration_workspace_alloc(200);    
    gsl_integration_qawo_table *t = gsl_integration_qawo_table_alloc(q, kmax-kmin, GSL_INTEG_COSINE, size);
    gsl_integration_qawf(&F, xpivot_TV/q, 1e-6, 200, w1, w2, t, &result2, &error);
    
    // Second integration with qawo 2/2
    F.function = &fV3_112_qawo_2_over_2;
    F.params = &alpha;       
    gsl_integration_qawo_table *t2 = gsl_integration_qawo_table_alloc(q, kmax-kmin, GSL_INTEG_SINE, size);
    gsl_integration_qawf(&F, xpivot_TV/q, 1e-6, 200, w1, w2, t2, &result3, &error);
    
    gsl_integration_workspace_free(w1);
    gsl_integration_workspace_free(w2);
    gsl_integration_qawo_table_free(t);
    gsl_integration_qawo_table_free(t2);
    return result1+result2+result3;
  }
}

// V^10(q)		
double fV_10 (double k, void *p)
{
  double q = *(double *) p; 
  return -1./7.*Q_s(k)*k*func_bessel_1(k*q);
}

double fV_10_qawo (double k, void *p)
{
  double alpha = *(double *) p; 
  return -1./7.*alpha*Q_s(k);
}

double V_10 (double q)
{
  if(kmax < xpivot_TV/(q+1e-8))
    return 1./(2.*M_PI*M_PI)*int_cquad(fV_10, q);
  else{
    double result1, result2, error;
    // First integration using cquad
    result1 = 1./(2.*M_PI*M_PI)*int_cquad_pivot_UYXi(fV_10, q); 
    
    // Second integration with qawo
    double alpha = 1./(2.*M_PI*M_PI*q);
    gsl_function F;
    F.function = &fV_10_qawo;
    F.params = &alpha;       
    
    gsl_integration_workspace *w1 = gsl_integration_workspace_alloc(200);
    gsl_integration_workspace *w2 = gsl_integration_workspace_alloc(200);    
    gsl_integration_qawo_table *t = gsl_integration_qawo_table_alloc(q, kmax-kmin, GSL_INTEG_COSINE, size);
    gsl_integration_qawf(&F, xpivot_TV/q, 1e-4, 200, w1, w2, t, &result2, &error);
    
    gsl_integration_workspace_free(w1);
    gsl_integration_workspace_free(w2);
    gsl_integration_qawo_table_free(t);
    return result1+result2;
  } 
}

// T^112(q)
double fT_112 (double k, void *p)
{
  double q = *(double *) p; 
  double x = k*q;
  return -3./7.*(2.*R_1(k)+4.*R_2(k)+Q_1(k)+2.*Q_2(k))*func_bessel_3(x)/k;
}

double fT_112_qawo (double k, void *p)
{
  double alpha = *(double *) p; 
  return -3./7.*alpha*(2.*R_1(k)+4.*R_2(k)+Q_1(k)+2.*Q_2(k))/(k*k);
}

double T_112 (double q)
{
  if(q < 3e-8/kmin) return 0;
  else if(kmax < xpivot_TV/q) return 1./(2.*M_PI*M_PI)*int_cquad(fT_112, q); 
  else {
    double result1, result2, error;
    // First integration using cquad
    result1 = 1./(2.*M_PI*M_PI)*int_cquad_pivot_TV(fT_112, q); 
    
    // Second integration with qawo 1/2
    double alpha = 1./(2.*M_PI*M_PI*q*q);
    gsl_function F;
    F.function = &fT_112_qawo;
    F.params = &alpha;       
    
    gsl_integration_workspace *w1 = gsl_integration_workspace_alloc(200);
    gsl_integration_workspace *w2 = gsl_integration_workspace_alloc(200);    
    gsl_integration_qawo_table *t = gsl_integration_qawo_table_alloc(q, kmax-kmin, GSL_INTEG_COSINE, size);
    gsl_integration_qawf(&F, xpivot_TV/q, 1e-6, 200, w1, w2, t, &result2, &error);
    
    gsl_integration_workspace_free(w1);
    gsl_integration_workspace_free(w2);
    gsl_integration_qawo_table_free(t);
    return result1+result2;
  }
}

/************************************************************************************************************************************************************************/
/***************************\\Compute U, X, Y, V and T (q) in text file for interpolation\\******************************************************************************/
												
void q_function(void)
 {
   FILE *fx, *fy, *fw, *fu, *fxi;
   double q, logq; 
   double qmin = 0.001;
   double qmax = 2000; // Maximum q to compute the q functions // 2000 by default
   int nq = 3000; // nombre de pas
   double log_qmin = log10(qmin);
   double log_qmax = log10(qmax);
   double dq = (log_qmax - log_qmin)/nq;
   
   printf("getting Xi_L function...\n");
   fxi = fopen("data/Xi_L_func.dat","w+"); 
   fprintf(fxi,"%.13lf %.13lf\n",0., Xi_L(0.));
   fclose(fxi);	
   fxi = fopen("data/Xi_L_func.dat","a+"); 
   for (logq=log_qmin; logq<log_qmax; logq+=dq){
     q = pow(10, logq);
     fprintf(fxi,"%.13lf %.13lf\n",q, Xi_L(q));
   }
   fclose(fxi);	
   printf("getting U function...\n");
   fu = fopen("data/U_func.dat","w+"); 
   fprintf(fu,"%.13lf %.13lf %.13lf %.13lf %.13lf\n",0., 0., 0., 0., 0.);
   fclose(fu);	
   fu = fopen("data/U_func.dat","a+"); 
   for (logq=log_qmin; logq<log_qmax; logq+=dq){
     q = pow(10, logq);
     //printf("U(%f)\n", q);
     fprintf(fu,"%.13lf %.13lf %.13lf %.13lf %.13lf\n",q, U_1(q), U_3(q), U_11(q), U_20(q));
   }
   fclose(fu);	
   printf("getting X function...\n");
   fx = fopen("data/X_func.dat","w+"); 
   fprintf(fx,"%.13lf %.13lf %.13lf %.13lf %.13lf\n",0., 0., 0., 0., 0.);
   fclose(fx);
   fx = fopen("data/X_func.dat","a+"); 
   for (logq=log_qmin; logq<log_qmax; logq+=dq){
     q = pow(10, logq);
     fprintf(fx,"%.13lf %.13lf %.13lf %.13lf %.13lf\n",q, X_11(q), X_13(q), X_22(q), X_10_12(q));
   }
   fclose(fx);	
   printf("getting Y function...\n");
   fy = fopen("data/Y_func.dat","w+"); 
   fprintf(fy,"%.13lf %.13lf %.13lf %.13lf %.13lf\n",0., 0., 0., 0., 0.);
   fclose(fy);
   fy = fopen("data/Y_func.dat","a+"); 
   for (logq=log_qmin; logq<log_qmax; logq+=dq){
     q = pow(10, logq);
     fprintf(fy,"%.13lf %.13lf %.13lf %.13lf %.13lf\n",q, Y_11(q), Y_13(q), Y_22(q), Y_10_12(q));
   }
   fclose(fy);	
   printf("getting W function...\n");
   fw = fopen("data/W_func.dat","w+"); 
   fprintf(fw,"%.13lf %.13lf %.13lf %.13lf\n",0., 0., 0., 0.);
   fclose(fw);
   fw = fopen("data/W_func.dat","a+"); 
   for (logq=log_qmin; logq<log_qmax; logq+=dq){
     q = pow(10, logq);
     fprintf(fw,"%.13lf %.13lf %.13lf %.13lf\n",q, V1_112(q), V3_112(q), T_112(q));
   }
   fclose(fw);
   return;
 }

void write_qfuncs(void)
 {
   FILE *fx;
   double q, logq;
   double qmin = 0.001;
   double qmax = 2000; // Maximum q to compute the q functions // 2000 by default                                                                        
   int nq = 3000; // nombre de pas                                                                                                                       
   double log_qmin = log10(qmin);
   double log_qmax = log10(qmax);
   double dq = (log_qmax - log_qmin)/nq;

   printf("getting Xi_L and q functions...\n");
   
   fx = fopen("data/qfuncs.dat","w+");
   fprintf(fx,"0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.\n");
   
   for (logq=log_qmin; logq<log_qmax; logq+=dq){
     q = pow(10, logq);
     fprintf(fx,"%.13le %.13le %.13le %.13le %.13le %.13le %.13le %.13le %.13le %.13le %.13le %.13le %.13le %.13le %.13le %.13le %.13le %.13le\n",
	     q, Xi_L(q), U_1(q), U_3(q), U_11(q), U_20(q), X_11(q), X_13(q), X_22(q), X_10_12(q), Y_11(q), Y_13(q), Y_22(q), Y_10_12(q),
	     V1_112(q), V3_112(q), T_112(q), V_10(q));
   }
   
   fclose(fx);
   return;
 }

/************************************************************************************************************************************************************************/
/*********************\\Interpolation functions\\***********************************************************************************************************************/
																										
void interpole(int n, char ficher[100], int nLines, int nHeader)
{
  double T_x[nLines];
  double T_y[nLines];
  FILE* f;
  f = fopen(ficher, "r");
  for (int i = 0; i < nHeader; i++) fscanf(f,"%*[^\n]\n");
  for (int i=0; i < nLines; i++) fscanf(f, "%lf %lf\n", &T_x[i], &T_y[i]);
  acc[n] = gsl_interp_accel_alloc ();
  spline[n] = gsl_spline_alloc(gsl_interp_cspline, nLines);
  gsl_spline_init (spline[n], T_x, T_y, nLines);
  fclose(f);
}

void interp_qfunc()
{
  FILE *fu, *fx, *fy, *fw;
  int i;
  const int val = 3002; // WARNING : number of lines in files (might change if we want to increase 
  double X_q[val], X_x11[val], X_x13[val], X_x22[val], X_x1012[val], Y_q[val], Y_y11[val], Y_y13[val], Y_y22[val], Y_y1012[val],
    U_q[val], U_u1[val], U_u3[val], U_u11[val], U_u20[val], W_q[val], W_v1[val], W_v3[val], W_t[val];
  
  fu = fopen("data/U_func.dat", "r");
  for(i=0; i < val; i++) fscanf(fu, "%lf %lf %lf %lf %lf\n", &U_q[i], &U_u1[i], &U_u3[i], &U_u11[i], &U_u20[i]);
  
  acc[9] = gsl_interp_accel_alloc ();
  spline[9] = gsl_spline_alloc(gsl_interp_cspline, val);
  gsl_spline_init (spline[9], U_q, U_u1, val);
  acc[10] = gsl_interp_accel_alloc ();
  spline[10] = gsl_spline_alloc(gsl_interp_cspline, val);
  gsl_spline_init (spline[10], U_q, U_u3, val);
  acc[11] = gsl_interp_accel_alloc ();
  spline[11] = gsl_spline_alloc(gsl_interp_cspline, val);
  gsl_spline_init (spline[11], U_q, U_u11, val);
  acc[12] = gsl_interp_accel_alloc ();
  spline[12] = gsl_spline_alloc(gsl_interp_cspline, val);
  gsl_spline_init (spline[12], U_q, U_u20, val);
  fclose(fu);
  
  fx = fopen("data/X_func.dat", "r");
  for(i=0; i < val; i++) fscanf(fx, "%lf %lf %lf %lf %lf\n", &X_q[i], &X_x11[i], &X_x13[i], &X_x22[i], &X_x1012[i]);
  
  acc[13] = gsl_interp_accel_alloc ();
  spline[13] = gsl_spline_alloc(gsl_interp_cspline, val);
  gsl_spline_init (spline[13], X_q, X_x11, val);
  acc[14] = gsl_interp_accel_alloc ();
  spline[14] = gsl_spline_alloc(gsl_interp_cspline, val);
  gsl_spline_init (spline[14], X_q, X_x13, val);
  acc[15] = gsl_interp_accel_alloc ();
  spline[15] = gsl_spline_alloc(gsl_interp_cspline, val);
  gsl_spline_init (spline[15], X_q, X_x22, val);
  acc[16] = gsl_interp_accel_alloc ();
  spline[16] = gsl_spline_alloc(gsl_interp_cspline, val);
  gsl_spline_init (spline[16], X_q, X_x1012, val);
  fclose(fx);
  
  fy = fopen("data/Y_func.dat", "r");
  for(i=0; i < val; i++) fscanf(fy, "%lf %lf %lf %lf %lf\n", &Y_q[i], &Y_y11[i], &Y_y13[i], &Y_y22[i], &Y_y1012[i]);
  acc[17] = gsl_interp_accel_alloc ();
  spline[17] = gsl_spline_alloc(gsl_interp_cspline, val);
  gsl_spline_init (spline[17], Y_q, Y_y11, val);
  acc[18] = gsl_interp_accel_alloc ();
  spline[18] = gsl_spline_alloc(gsl_interp_cspline, val);
  gsl_spline_init (spline[18], Y_q, Y_y13, val);
  acc[19] = gsl_interp_accel_alloc ();
  spline[19] = gsl_spline_alloc(gsl_interp_cspline, val);
  gsl_spline_init (spline[19], Y_q, Y_y22, val);
  acc[20] = gsl_interp_accel_alloc ();
  spline[20] = gsl_spline_alloc(gsl_interp_cspline, val);
  gsl_spline_init (spline[20], Y_q, Y_y1012, val);
  fclose(fy);
  
  fw = fopen("data/W_func.dat", "r");
  for(i=0; i < val; i++) fscanf(fw, "%lf %lf %lf %lf\n", &W_q[i], &W_v1[i], &W_v3[i], &W_t[i]);
  
  acc[21] = gsl_interp_accel_alloc ();
  spline[21] = gsl_spline_alloc(gsl_interp_cspline, val);
  gsl_spline_init (spline[21], W_q, W_v1, val);
  acc[22] = gsl_interp_accel_alloc ();
  spline[22] = gsl_spline_alloc(gsl_interp_cspline, val);
  gsl_spline_init (spline[22], W_q, W_v3, val);
  acc[23] = gsl_interp_accel_alloc ();
  spline[23] = gsl_spline_alloc(gsl_interp_cspline, val);
  gsl_spline_init (spline[23], W_q, W_t, val);
  fclose(fw);
}

void interpole_qfuncs(int dofast)
{
  FILE *fu;
  int i;
  const int val = 3002; // WARNING : number of lines in files (might change if we want to increase                                                       
  double q[val], xil[val], X_x11[val], X_x13[val], X_x22[val], X_x1012[val], Y_y11[val], Y_y13[val], Y_y22[val], Y_y1012[val],
    U_u1[val], U_u3[val], U_u11[val], U_u20[val], W_v1[val], W_v3[val], W_t[val], W_v10[val];

  if (dofast==0) {
    fu = fopen("data/qfuncs.dat", "r");
    for(i=0; i < val; i++) fscanf(fu, "%lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf\n", &q[i], &xil[i],
				  &U_u1[i], &U_u3[i], &U_u11[i], &U_u20[i], &X_x11[i], &X_x13[i], &X_x22[i], &X_x1012[i],
				  &Y_y11[i], &Y_y13[i], &Y_y22[i], &Y_y1012[i], &W_v1[i], &W_v3[i], &W_t[i], &W_v10[i]);
    fclose(fu);
  } else {
    q[0] = xil[0] = X_x11[0] = X_x13[0] = X_x22[0] = X_x1012[0] = Y_y11[0] = Y_y13[0] = Y_y22[0] = Y_y1012[0] = U_u1[0] = U_u3[0] = U_u11[0] = U_u20[0] = W_v1[0] = W_v3[0] = W_t[0] = W_v10[0] = 0.;
    fu = fopen("data/qfuncs_fast.dat", "r");
    for(i=1; i < val; i++) fscanf(fu, "%lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf\n", &q[i], &xil[i],
				  &X_x11[i], &X_x22[i], &X_x13[i], &Y_y11[i], &Y_y22[i], &Y_y13[i], &X_x1012[i], &Y_y1012[i],
				  &W_v1[i], &W_v3[i], &W_t[i], &U_u1[i], &U_u3[i], &U_u20[i], &U_u11[i], &W_v10[i]);
  }

  acc[8] = gsl_interp_accel_alloc ();
  spline[8] = gsl_spline_alloc(gsl_interp_cspline, val);
  gsl_spline_init (spline[8], q, xil, val);
  acc[9] = gsl_interp_accel_alloc ();
  spline[9] = gsl_spline_alloc(gsl_interp_cspline, val);
  gsl_spline_init (spline[9], q, U_u1, val);
  acc[10] = gsl_interp_accel_alloc ();
  spline[10] = gsl_spline_alloc(gsl_interp_cspline, val);
  gsl_spline_init (spline[10], q, U_u3, val);
  acc[11] = gsl_interp_accel_alloc ();
  spline[11] = gsl_spline_alloc(gsl_interp_cspline, val);
  gsl_spline_init (spline[11], q, U_u11, val);
  acc[12] = gsl_interp_accel_alloc ();
  spline[12] = gsl_spline_alloc(gsl_interp_cspline, val);
  gsl_spline_init (spline[12], q, U_u20, val);
  acc[13] = gsl_interp_accel_alloc ();
  spline[13] = gsl_spline_alloc(gsl_interp_cspline, val);
  gsl_spline_init (spline[13], q, X_x11, val);
  acc[14] = gsl_interp_accel_alloc ();
  spline[14] = gsl_spline_alloc(gsl_interp_cspline, val);
  gsl_spline_init (spline[14], q, X_x13, val);
  acc[15] = gsl_interp_accel_alloc ();
  spline[15] = gsl_spline_alloc(gsl_interp_cspline, val);
  gsl_spline_init (spline[15], q, X_x22, val);
  acc[16] = gsl_interp_accel_alloc ();
  spline[16] = gsl_spline_alloc(gsl_interp_cspline, val);
  gsl_spline_init (spline[16], q, X_x1012, val);
  acc[17] = gsl_interp_accel_alloc ();
  spline[17] = gsl_spline_alloc(gsl_interp_cspline, val);
  gsl_spline_init (spline[17], q, Y_y11, val);
  acc[18] = gsl_interp_accel_alloc ();
  spline[18] = gsl_spline_alloc(gsl_interp_cspline, val);
  gsl_spline_init (spline[18], q, Y_y13, val);
  acc[19] = gsl_interp_accel_alloc ();
  spline[19] = gsl_spline_alloc(gsl_interp_cspline, val);
  gsl_spline_init (spline[19], q, Y_y22, val);
  acc[20] = gsl_interp_accel_alloc ();
  spline[20] = gsl_spline_alloc(gsl_interp_cspline, val);
  gsl_spline_init (spline[20], q, Y_y1012, val);
  acc[21] = gsl_interp_accel_alloc ();
  spline[21] = gsl_spline_alloc(gsl_interp_cspline, val);
  gsl_spline_init (spline[21], q, W_v1, val);
  acc[22] = gsl_interp_accel_alloc ();
  spline[22] = gsl_spline_alloc(gsl_interp_cspline, val);
  gsl_spline_init (spline[22], q, W_v3, val);
  acc[23] = gsl_interp_accel_alloc ();
  spline[23] = gsl_spline_alloc(gsl_interp_cspline, val);
  gsl_spline_init (spline[23], q, W_t, val);
  acc[25] = gsl_interp_accel_alloc ();
  spline[25] = gsl_spline_alloc(gsl_interp_cspline, val);
  gsl_spline_init (spline[25], q, W_v10, val);

  return;
}

void interpole_jfuncs(void)
{
  FILE *fu;
  int i;
  const int val = 3002; // WARNING : number of lines in files (might change if we want to increase                                                       
  double q[val], B1[val], B2[val], J2[val], J3[val], J4[val], v12[val], chi12[val], zeta[val];
  
  q[0] = v12[0] = chi12[0] = zeta[0] = 0.;
  fu = fopen("data/jfuncs_fast.dat", "r");
  for(i=1; i < val; i++) fscanf(fu, "%lf %lf %lf %*f %lf %lf %lf %lf %lf %lf\n", &q[i], &B1[i], &B2[i], &J2[i], &J3[i], &J4[i], &v12[i], &chi12[i], &zeta[i]);

  acc[26] = gsl_interp_accel_alloc ();
  spline[26] = gsl_spline_alloc(gsl_interp_cspline, val);
  gsl_spline_init (spline[26], q, B1, val);

  acc[27] = gsl_interp_accel_alloc ();
  spline[27] = gsl_spline_alloc(gsl_interp_cspline, val);
  gsl_spline_init (spline[27], q, B2, val);

  acc[28] = gsl_interp_accel_alloc ();
  spline[28] = gsl_spline_alloc(gsl_interp_cspline, val);
  gsl_spline_init (spline[28], q, J2, val);
  
  acc[29] = gsl_interp_accel_alloc ();
  spline[29] = gsl_spline_alloc(gsl_interp_cspline, val);
  gsl_spline_init (spline[29], q, J3, val);

  acc[30] = gsl_interp_accel_alloc ();
  spline[30] = gsl_spline_alloc(gsl_interp_cspline, val);
  gsl_spline_init (spline[30], q, J4, val);
  
  acc[31] = gsl_interp_accel_alloc ();
  spline[31] = gsl_spline_alloc(gsl_interp_cspline, val);
  gsl_spline_init (spline[31], q, v12, val);

  acc[32] = gsl_interp_accel_alloc ();
  spline[32] = gsl_spline_alloc(gsl_interp_cspline, val);
  gsl_spline_init (spline[32], q, chi12, val);

  acc[33] = gsl_interp_accel_alloc ();
  spline[33] = gsl_spline_alloc(gsl_interp_cspline, val);
  gsl_spline_init (spline[33], q, zeta, val);

  return;
}


/************************************************************************************************************************************************************************/
/*****************\\Matrix A \\ *****************************************************************************************************************************************/

/*A_ij(q) = X(q)*delta_k(i,j) + Y(q)*q_v[i]*q_v[j]*/ 	
																					
double Aij_13 (int i, int j, double q)			
{
  return iX_13(q)*delta_K(i,j)+iY_13(q)*q_v[i]*q_v[j];  // q_v[i] unit vector over i component  
}																					

double Aij_22 (int i, int j, double q)			
{		
  return iX_22(q)*delta_K(i,j)+iY_22(q)*q_v[i]*q_v[j];		
}	

double Aij_11 (int i, int j, double q)		
{	
  return iX_11(q)*delta_K(i,j)+iY_11(q)*q_v[i]*q_v[j];		
}	

double Aloop_ij(int i, int j, double q)		
{	
  return Aij_22(i, j, q)+2.*Aij_13(i, j, q);
}

double Aij(int i, int j, double q)
{
  return Aij_11(i, j, q)+Aij_22(i, j, q)+2.*Aij_13(i, j, q);
}

double A10_ij(int i, int j, double q)
{
  return 2.*(iX_10_12(q)*delta_K(i,j)+iY_10_12(q)*q_v[i]*q_v[j]);
}

// Initialize A matrix 
void get_M(double q, gsl_matrix **A)
{	
  int i, j;																				
  for (i=0; i<3; i++){		   		
    for (j=0; j<3;j++) 
      gsl_matrix_set(*A, i, j, Aij_11(i,j,q));
  }
}

/************************************************************************************************************************************************************************/
/******************\\ g_i, G_ij and Gamma_ijk functions\\**************************************************************************************************************/
														
double g_i (int i, int j, gsl_matrix *M_inv, double y[])
{
  return gsl_matrix_get(M_inv, i, j)*y[j];
}																	

double G_ij (int i, int j, gsl_matrix *M_inv, double g[])
{
  return gsl_matrix_get(M_inv, i, j)-g[i]*g[j];
}																			

double Gamma_ijk (int i, int j, int k, gsl_matrix *M_inv, double g[])
{
  return gsl_matrix_get(M_inv, i, j)*g[k]+ gsl_matrix_get(M_inv, k, i)*g[j] + gsl_matrix_get(M_inv, j, k)*g[i] - g[i]*g[j]*g[k];
}

/************************************************************************************************************************************************************************/
/******************\\Compute W^112_ijk(q)\\**************************************************************************************************************************/
																
double W_112 (int i, int j, int k, double q)
{
  return iW_V1(q)*delta_K(j,k)*q_v[i] + iW_V1(q)*delta_K(k,i)*q_v[j] + iW_V3(q)*delta_K(i,j)*q_v[k] + iW_T(q)*q_v[i]*q_v[j]*q_v[k];	
}

/************************************************************************************************************************************************************************/
/*************\\Dot functions\\*****************************************************************************************************************************************/

//Udot_n
double fU_dot (double q)
{
  return iU_1(q) + 3.*iU_3(q);
}

//Adot_in
double fA_dot (int i, int n, double q)
{
  return Aij_11(i,n,q)+4.*Aij_13(i,n,q)+2.*Aij_22(i,n,q);
}

//Adot^10_in	
double fA10_dot (int i, int n, double q)
{
  return 3.*(iX_10_12(q)*delta_K(i,n)+iY_10_12(q)*q_v[i]*q_v[n]);
}

//Wdot_ijn
double fW_dot (int i, int j, int n, double q)
{
  return 2.*W_112(i,j,n,q)+W_112(n,i,j,q)+W_112(j,n,i,q);
}

/************************************************************************************************************************************************************************/
/*********\\Double dot functions\\*************************************************************************************************************************************/

//A2dot_nm
double fA_2dot (int n, int m, double q)
{
  return Aij_11(n,m,q)+6.*Aij_13(n,m,q)+4.*Aij_22(n,m,q);
}

//A2dot^10_nm	
double fA10_2dot (int n, int m, double q)
{
  return 4.*(iX_10_12(q)*delta_K(n,m)+iY_10_12(q)*q_v[n]*q_v[m]);	//A_nm = A_mn
}

// W2dot_inm
double fW_2dot (int i, int n, int m, double q)
{
  return 2.*W_112(i,n,m,q)+2.*W_112(m,i,n,q)+W_112(n,m,i,q);		//W^121_inm = W^112_min and W^211_inm = W^112_nmi
}

/************************************************************************************************************************************************************************/
/**************\\M_0 integral\\*****************************************************************************************************************************************/

void M_0(double y, double R, double M0_fin[])
{
  int l, m,o;
  double mu, q_n, det_A, y_t[3], q[3], U[3], U1[3], G[3][3], U11[3], U20[3], V10[3], V12[3], B2[3], g[3], Gamma[3][3][3], W[3][3][3], W_fin[3][3][3], A10[3][3], Ups[3][3], Xi, trG, s, fac, F_b[13], M0_tab[13];
  double dmu = dmu_M0;
  int signum = 0;
  
  for (l=0; l<13; l++) M0_tab[l] = 0;				// Inizialise at each step
  
  for (mu=-1; mu <= 1; mu +=dmu){				// Decomp integrale dtheta en somme de riemann int=somme f(x_i)*delta_x		
    gsl_matrix *A = gsl_matrix_calloc (3, 3);
    gsl_matrix *A_inv = gsl_matrix_calloc (3, 3);		// Declaration of matrix A (3x3)
    
    // Change variable y = q-r
    y_t[0] = y*sqrt(1-mu*mu);
    y_t[1] = 0;
    y_t[2] = y*mu;
    
    // Compute vector q = y+r with r= (0,0,1) along the LOS
    q[0] = y_t[0];
    q[1] = y_t[1];
    q[2] = y_t[2]+R;
    q_n = sqrt(q[0]*q[0]+q[1]*q[1]+q[2]*q[2]); 	// norm of q
    
    // Unit vector
    for (l=0; l<3; l++)	q_v[l] = q[l]/q_n; 
    
    /******** Compute functions ****************************************/
    
    // Xi linear
    Xi = iXi_L(q_n);

    // U functions
    for (l = 0; l<3 ; l++){
      U[l] = get_U(q_n)*q_v[l];
      U1[l] = iU_1(q_n)*q_v[l];
      U11[l] = iU_11(q_n)*q_v[l];
      U20[l] = iU_20(q_n)*q_v[l]; 
    }

    // V functions
    for (l = 0; l<3 ; l++){
      V10[l] = iV_10(q_n)*q_v[l];
      V12[l] = iV_12(q_n)*q_v[l];
    }

    // B function
    
    for (l = 0; l<3 ; l++){
      B2[l] = iB_2(q_n)*q_v[l];
    }
    
    // 	Matix A			
    gsl_permutation *p = gsl_permutation_alloc (3);
    get_M(q_n, &A);							        // Initialize matix A (A_11)	   
    gsl_linalg_LU_decomp(A, p, &signum); 					
    det_A = gsl_linalg_LU_det(A, signum);					// Determinant of A
    gsl_linalg_LU_invert(A, p, A_inv); 	    					// Inverse of matrix A 
      
    // 	g_i components	
    for (l=0; l<3; l++){
      g[l] = 0;
      for (m=0; m<3; m++)	
	g[l] += g_i(l, m, A_inv, y_t);
    }	     
	    	
    // 	G_ij components 
    for (l=0; l<3; l++){
      for (m=0; m<3; m++)	
	G[l][m] = G_ij(l, m, A_inv, g);
    }	    		

    trG = 0;
    for (l=0; l<3; l++) trG += G[l][l];
    
    // Gamma_ijk
    for (l=0; l<3; l++){
      for (m=0; m<3; m++){
	for (o=0; o<3; o++) 
	  Gamma[l][m][o] = Gamma_ijk(l, m, o, A_inv, g);
      }
    }	 	
    
    // 	W_ijk
    for (l=0; l<3; l++){
      for (m=0; m<3; m++){
	for (o=0; o<3; o++){
	  W[l][m][o] = W_112(l, m, o, q_n);
	}
      }		
    }
    for (l=0; l<3; l++){
      for (m=0; m<3; m++){
	for (o=0; o<3; o++){
	  W_fin[l][m][o] = W[l][m][o]+ W[o][l][m]+ W[m][o][l]; 
	}
      }		
    }		
    
    // 	A^10_ij
    for (l=0; l<3; l++){
      for (m=0; m<3; m++)
	A10[l][m] = A10_ij(l, m, q_n);
    }

    // 	Ups_ij
    for (l=0; l<3; l++){
      for (m=0; m<3; m++) {
	double j2 = iJ_2(q_n);
	double j3 = iJ_3(q_n);
	double j4 = iJ_4(q_n);
	if (l==m) Ups[l][m] = 2*j3*j3;
	else Ups[l][m] = 0;
	Ups[l][m] += q_v[l]*q_v[m]*(3*j2*j2 + 4*j2*j3 + 2*j2*j4 + 2*j3*j3 +4*j3*j4 +j4*j4);
	Ups[l][m] *= 2;
      }
    }
    
    // Sum over all the bias component	
    for(l=0; l<13; l++) F_b[l] = 0;
    
    // "1" term
    s=0;
    for (l=0; l<3; l++){
      for (m=0; m<3; m++){
	for (o=0; o<3; o++){	
	  s += Gamma[l][m][o]*W_fin[l][m][o];
	}
      }
    }
    F_b[0] += s/6.;
    
    s=0;
    for (l=0; l<3; l++){
      for (m=0; m<3; m++)
	s += G[l][m]*Aloop_ij(l, m, q_n);
    }
    F_b[0] += 1.0 - 0.5*s; // TBC
    
    // b1
    s=0;
    for (l=0; l<3; l++) s += U[l]*g[l];
    F_b[1] -= 2.*s;
    
    s=0;
    for (l=0; l<3; l++){
      for (m=0; m<3; m++)
	s += A10[l][m]*G[l][m];
    }
    F_b[1] -= s;

    // b2
    s=0;
    for (l=0; l<3; l++) s += U20[l]*g[l];
    F_b[2] -= s;
    s=0;
    for (l=0; l<3; l++){
      for (m=0; m<3; m++) s += U1[l]*U1[m]*G[l][m];
    }
    F_b[2] -= s;

    // b1² 
    F_b[3] += Xi;
    s=0;
    for (l=0; l<3; l++)
      s += U11[l]*g[l];
    F_b[3] -= s;
    s=0;
    for (l=0; l<3; l++){
      for (m=0; m<3; m++) s += U1[l]*U1[m]*G[l][m];
    }
    F_b[3] -= s;
    
    // b1.b2
    s=0;
    for (l=0; l<3; l++) s += U1[l]*g[l];
    F_b[4] -= 2.*Xi*s;
    
    // b2²
    F_b[5] = 0.5*Xi*Xi;

    // bs2
    s=0;
    for (l=0; l<3; l++){
      for (m=0; m<3; m++)
	s += Ups[l][m]*G[l][m];
    }
    for (l=0; l<3; l++) s += 2*g[l]*V10[l];
    F_b[6] -= s;

    // bs2^2
    F_b[7] += iZeta(q_n);

    // b1 bs2
    s=0;
    for (l=0; l<3; l++) s += 2*g[l]*V12[l];
    F_b[8] -= s;

    // b2 bs2
    F_b[9] += iChi_12(q_n);

    // alpha_xi
    F_b[10] -= 0.5*trG;

    // bn2
    F_b[11] += 2*iB_1(q_n);

    // b1 bn2
    s=0;
    for (l=0; l<3; l++) s += 2*g[l]*B2[l];
    F_b[12] += s;
    
    // 	gaussian factor of M
    s = 0; 
    for (l=0; l<3; l++) s += g[l]*y_t[l];
    fac = exp(-0.5*s)/(sqrt(2.*M_PI)*2.*M_PI*sqrt(det_A));
    
    //	Sum over all component
    for (l=0; l<13; l++) M0_tab[l] += F_b[l]*fac;
    
    gsl_permutation_free (p);
    gsl_matrix_free(A);
    gsl_matrix_free(A_inv);
  }
  s = y*y*dmu;
  for (l=0; l<13; l++) M0_fin[l] = M0_tab[l]*s*2.*M_PI;

  return;
}

/************************************************************************************************************************************************************************/
/**************\\M_1 integral\\*****************************************************************************************************************************************/

void M_1(double y, double R, double M1_fin[])
{
  const double rn[3] = {0, 0, 1}; // unit vector to project over the LOS
  int l, m,o;
  double mu, q_n, det_A, y_t[3], q[3], U_dot[3], U1[3], G[3][3], U11_dot[3], U20_dot[3], g[3], B2[3], W[3][3][3], W_dot[3][3][3], A_lin[3][3], A_dot[3][3], A10_dot[3][3], Ups[3][3], Xi, s, fac, F_b[10], M1_tab[10]; 
  double dmu = dmu_M1;
  int signum = 0;					// Use for matrix inversion
  
  for (l = 0; l<10 ; l++) M1_tab[l] = 0;			// Initialize at each step
  
  for (mu=-1; mu <= 1; mu +=dmu){				// Decomp integrale dtheta en somme de riemann int=somme f(x_i)*delta_x		
    gsl_matrix *A = gsl_matrix_calloc (3, 3);
    gsl_matrix *A_inv = gsl_matrix_calloc (3, 3);		// Declaration of matrix A (3x3)
    
    // Change variable y = q-r
    y_t[0] = y*sqrt(1-mu*mu);
    y_t[1] = 0;
    y_t[2] = y*mu;
    
    // Compute vector q = y+r with r= (0,0,1) along the LOS
    q[0] = y_t[0];
    q[1] = y_t[1];
    q[2] = y_t[2]+R;
    q_n = sqrt(q[0]*q[0]+q[1]*q[1]+q[2]*q[2]); 	// norm of q
    
    // Unit vector
    for (l=0; l<3; l++)	q_v[l] = q[l]/q_n; 
    
    /********Compute function ****************************************/
    
    // Xi linear
    Xi = iXi_L(q_n);
    
    //  	U functions
    for (l = 0; l<3 ; l++){
      U_dot[l] = fU_dot(q_n)*q_v[l];
      U1[l] = iU_1(q_n)*q_v[l];
      U11_dot[l] = 2.*iU_11(q_n)*q_v[l];
      U20_dot[l] = 2.*iU_20(q_n)*q_v[l];
    }

    // B functions
    for (l = 0; l<3 ; l++){
      B2[l] = iB_2(q_n)*q_v[l];
    }
    
    // 	Matix A			
    gsl_permutation *p = gsl_permutation_alloc (3);
    get_M(q_n, &A);				       //Initialize matix A	    	    			   				   
    gsl_linalg_LU_decomp(A, p, &signum); 					
    det_A = gsl_linalg_LU_det(A, signum);	       // Determinant of A
    gsl_linalg_LU_invert(A, p, A_inv); 	    	       // Inverse of matrix A 
    
    // 	g_i components	
    for (l=0; l<3; l++){
      g[l] = 0;
      for (m=0; m<3; m++)	
	g[l] += g_i(l, m, A_inv, y_t);
    }	     
    
    // G_ij components 
    for (l=0; l<3; l++){
      for (m=0; m<3; m++)	
	G[l][m] = G_ij(l, m, A_inv, g);
    }	    		
    
    // Alin
    for (l=0; l<3; l++){
      for(m=0; m<3; m++)
	A_lin[l][m] = Aij_11(l, m, q_n);
    }

    // Adot_in
    for (l=0; l<3; l++){
      for(m=0; m<3; m++)
	A_dot[l][m] = fA_dot(l, m, q_n);
    }
    
    // Adot^10_in
    for (l=0; l<3; l++){
      for (m=0; m<3; m++)
	A10_dot[l][m] = fA10_dot(l, m, q_n);
    }
    
    // W_ijn
    for (l=0; l<3; l++){
      for(m=0; m<3; m++){
	for (o=0; o<3; o++)
	  W[l][m][o] = W_112(l, m, o, q_n);
      }		
    }	
    for (l=0; l<3; l++){
      for(m=0; m<3; m++){
	for (o=0; o<3; o++)
	  W_dot[l][m][o] = 2.*W[l][m][o]+ W[o][l][m]+ W[m][o][l];
      }	
    }	

    // 	Ups_ij
    for (l=0; l<3; l++){
      for (m=0; m<3; m++) {
	double j2 = iJ_2(q_n);
	double j3 = iJ_3(q_n);
	double j4 = iJ_4(q_n);
	if (l==m) Ups[l][m] = 2*j3*j3;
	else Ups[l][m] = 0;
	Ups[l][m] += q_v[l]*q_v[m]*(3*j2*j2 + 4*j2*j3 + 2*j2*j4 + 2*j3*j3 +4*j3*j4 +j4*j4);
	Ups[l][m] *= 2;
      }
    }
    
    // Sum over all the bias component	
    for(l=0; l<10; l++) F_b[l] = 0;

    // "1" term
    s=0;
    for (l=0; l<3; l++){
      for (m=0; m<3; m++){
	for (o=0; o<3; o++) s += G[l][m]*W_dot[l][m][o]*rn[o];
      }
    }
    F_b[0] -= s/2.;
    s=0;
    for (l=0; l<3; l++){
      for (m=0; m<3; m++) s += g[l]*A_dot[l][m]*rn[m];
    }
    F_b[0] -= s;
    
    // 	b1
    s = 0;
    for (l=0; l<3; l++) s += U_dot[l]*rn[l];
    F_b[1] += 2.*s;
    s = 0;
    for (l=0; l<3; l++){
      for (o=0; o<3; o++)
	s += g[l]*A10_dot[l][o]*rn[o];
    }
    F_b[1] -= 2.*s;
    s = 0;
    for (l=0; l<3; l++){
      for (m=0; m<3; m++){
	for (o=0; o<3; o++)
	  s += G[l][m]*U1[l]*A_lin[m][o]*rn[o];
      }
    }
    F_b[1] -= 2.*s;
    
    // b2
    F_b[2] += U20_dot[2]*rn[2];
    s=0;
    for (l=0; l<3; l++){
      for (m=0; m<3; m++)
	s += g[l]*U1[l]*U1[m]*rn[m];
    }
    F_b[2] -= 2.*s;
    
    // b1²
    F_b[3] += U11_dot[2]*rn[2];
    s=0;
    for (l=0; l<3; l++){
      for (m=0; m<3; m++)
	s += g[l]*U1[l]*U1[m]*rn[m];
    }
    F_b[3] -= 2.*s;

    s=0;
    for (l=0; l<3; l++) s += g[l]*A_lin[l][2]*rn[2];
    F_b[3] -= Xi*s;
    
    // b1.b2
    s=0;
    for (l=0; l<3; l++) s += U1[l]*rn[l];
    F_b[4] += 2.*Xi*s;

    // alpha_v
    F_b[5] = -iB_2(q_n)*rn[2];

    // alpha_vp
    F_b[6] = -g[2]*rn[2];

    // bn2
    F_b[7] = -2*B2[2]*rn[2];
    
    // bs2
    s = 2*iV_10(q_n)*rn[2];
    for (l=0; l<3; l++) s -= Ups[l][2]*rn[2]*g[l];
    F_b[8] = 2*s;
    
    // b1 bs2
    F_b[9] = 2*iV_12(q_n)*rn[2];
    
    // 	Gaussian factor of M
    s = 0;
    for (l=0; l<3; l++) s += g[l]*y_t[l];
    fac = exp(-0.5*s)/(sqrt(2.*M_PI)*2.*M_PI*sqrt(det_A));
    
    //	Sum over all component
    for (l=0; l<10; l++) M1_tab[l] += F_b[l]*fac;
    
    gsl_matrix_free(A);
    gsl_matrix_free(A_inv);		
    gsl_permutation_free (p);
  }	
  s = y*y*dmu;
  for (l=0; l<10; l++) M1_fin[l] = M1_tab[l]*s*2.*M_PI;

  return;
}

/************************************************************************************************************************************************************************/
/************************************************************************************************************************************************************************/
//Calcul de M2

void M_2 (double y, double R, double M2_fin[])
{
  const double rn[3] = {0, 0, 1}; // unit vector to project on n and m
  int l, m,o, n;
  double mu, q_n, det_A, y_t[3], q[3], U1[3], G[3][3], g[3], W[3][3][3], W_2dot[3][3][3], A_lin[3][3], A_2dot[3][3], A10_2dot[3][3], Ups[3][3], Xi, s[3][3], f, fac, M2_tab[14], t;
  double F_par[7],F_per[7]; 		//Bias factor for sigma_parallel and sigma_perpendicular	
  double dmu = dmu_M2;
  int signum = 0; 		        // Use for matrix inversion
  
  for (l=0; l<14; l++) M2_tab[l] = 0;	//Initialize at each step
  
  for (mu=-1; mu <= 1; mu +=dmu){				// Decomp integrale dtheta en somme de riemann int=somme f(x_i)*delta_x		
    gsl_matrix *A = gsl_matrix_calloc (3, 3);
    gsl_matrix *A_inv = gsl_matrix_calloc (3, 3);		// Declaration of matrix A (3x3)
    
    // Change variable y = q-r
    y_t[0] = y*sqrt(1-mu*mu);
    y_t[1] = 0;
    y_t[2] = y*mu;
    
    // 	Compute vector q = y+r with r= (0,0,1) along the LOS
    q[0] = y_t[0];
    q[1] = y_t[1];
    q[2] = y_t[2]+R;
    q_n = sqrt(q[0]*q[0]+q[1]*q[1]+q[2]*q[2]); 	// norm of q
    
    // 	unit vector
    for (l=0; l<3; l++)	q_v[l] = q[l]/q_n; 
    
    /********Compute function ****************************************/
    // Xi linear
    Xi = iXi_L(q_n);
    
    // 	U function
    for (l = 0; l<3 ; l++) {
      U1[l] = iU_1(q_n)*q_v[l];
    }	
    
    // 	Matix A			
    gsl_permutation *p = gsl_permutation_alloc (3);
    get_M(q_n, &A);							        //Initialize matix A	    	    			   				   
    gsl_linalg_LU_decomp(A, p, &signum); 					
    det_A = gsl_linalg_LU_det(A, signum);					// Determinant of A
    gsl_linalg_LU_invert(A, p, A_inv); 	    					// Inverse of matrix A 
    
    // 	g_i components	
    for (l=0; l<3; l++){
      g[l] = 0;
      for (m=0; m<3; m++)	
	g[l] += g_i(l, m, A_inv, y_t);
    }
    
    // 	G_ij components 
    for (l=0; l<3; l++){
      for (m=0; m<3; m++)	
	G[l][m] = G_ij(l, m, A_inv, g);
    }
    
    // 	Alin_nm
    for (l=0; l<3; l++){
      for (m=0; m<3; m++)
	A_lin[l][m] = Aij_11(l, m, q_n);
    }    	
    
    //	A2dot_nm
    for (l=0; l<3; l++){	
      for (m=0; m<3; m++)
	A_2dot[l][m] = fA_2dot(l, m, q_n);
    }
    
    //	A2dot^10_nm
    for (l=0; l<3; l++){
      for (m=0; m<3; m++)
	A10_2dot[l][m] = fA10_2dot(l, m, q_n);
    }

    //  W_inm
    for (l=0; l<3; l++){
      for (m=0; m<3; m++){
	for (o=0; o<3; o++)
	  W[l][m][o] = W_112(l, m, o, q_n);
      }		
    }
    for (l=0; l<3; l++){
      for (m=0; m<3; m++){
	for (o=0; o<3; o++)
	  W_2dot[l][m][o] = 2.*W[l][m][o]+ W[o][l][m]+ 2.*W[m][o][l];
      }	
    }				
    
    // Ups_ij
    for (l=0; l<3; l++){
      for (m=0; m<3; m++) {
	double j2 = iJ_2(q_n);
	double j3 = iJ_3(q_n);
	double j4 = iJ_4(q_n);
	if (l==m) Ups[l][m] = 2*j3*j3;
	else Ups[l][m] = 0;
	Ups[l][m] += q_v[l]*q_v[m]*(3*j2*j2 + 4*j2*j3 + 2*j2*j4 + 2*j3*j3 +4*j3*j4 +j4*j4);
	Ups[l][m] *= 2;
      }
    }
    
    // Sum over all the bias component	
    for (l=0; l<7; l++) F_par[l] = F_per[l] = 0;

    // "1" term		
    for (l=0; l<3; l++){
      for (m=0; m<3; m++)
	s[l][m] = A_2dot[l][m];
    }
    for (l=0; l<3; l++){
      for (m=0; m<3; m++){
	for(n=0; n<3; n++){
	  for (o=0; o<3; o++)
	    s[n][o] -= A_lin[l][n]*A_lin[m][o]*G[l][m];
	}
      }
    }
    for (l=0; l<3; l++){
      for (m=0; m<3; m++){
	for (o=0; o<3; o++)
	  s[m][o] -= W_2dot[l][m][o]*g[l];
      }
    }
    for (l=0; l<3; l++){
      for (m=0; m<3; m++){
	F_par[0] += s[l][m]*rn[l]*rn[m];
	F_per[0] += s[l][m]*delta_K(l,m);
      }
    }
    
    // 	b1 						
    for (l=0; l<3; l++){
      for (m=0; m<3; m++)
	s[l][m] = A10_2dot[l][m];
    }
    for (l=0; l<3; l++){
      for (m=0; m<3; m++){
	for (o=0; o<3; o++){
	  s[m][o] -= A_lin[l][m]*g[l]*U1[o]+A_lin[l][o]*g[l]*U1[m];
	  s[m][o] -= U1[l]*g[l]*A_lin[m][o];
	}
      }
    }
    for (l=0; l<3; l++){
      for (m=0; m<3; m++){
	F_par[1] += 2.*s[l][m]*rn[l]*rn[m];			
	F_per[1] += 2.*s[l][m]*delta_K(l,m);	 
      }
    }
    
    // b2
    for (l=0; l<3; l++){
      for (m=0; m<3; m++)
	s[l][m] = U1[l]*U1[m];
    }
    for (l=0; l<3; l++){
      for (m=0; m<3; m++){
	F_par[2] += 2.*s[l][m]*rn[l]*rn[m]; 
	F_per[2] += 2.*s[l][m]*delta_K(l,m);
      }
    }
    
    // b1²	
    for (l=0; l<3; l++){
      for (m=0; m<3; m++){
	s[l][m] = Xi*A_lin[l][m] + 2.*U1[l]*U1[m];
      }
    }		
    for (l=0; l<3; l++){
      for (m=0; m<3; m++){
	F_par[3] += s[l][m]*rn[l]*rn[m];				
	F_per[3] += s[l][m]*delta_K(l,m);
      }
    }

    // bs2
    F_par[4] = 2*Ups[2][2];
    t = 0;
    for (l=0; l<3; l++) t += Ups[l][l];
    F_per[4] = 2*t;

    // alpha_s
    F_par[5] = 1;				
    F_per[5] = 3;

    // beta_s
    F_par[6] += Xi;				
    F_per[6] += 3*Xi;
    
    // Gaussian factor of M
    f = 0;
    for (l=0; l<3; l++) f += g[l]*y_t[l];
    fac = exp(-0.5*f)/(sqrt(2.*M_PI)*2.*M_PI*sqrt(fabs(det_A)));
    
    // Sum over all component
    for (l=0; l<7; l++) {
      M2_tab[l] += F_par[l]*fac;
      M2_tab[l+7] += F_per[l]*fac;
    }
    gsl_permutation_free (p);
  }
    
  f = y*y*dmu;
  for (l=0; l<7; l++) {
    M2_fin[l] = M2_tab[l]*f*2.*M_PI;
    M2_fin[l+7] = 0.5*f*(M2_tab[l+7]-M2_tab[l])*2.*M_PI;
  }

  return;
}

/************************************************************************************************************************************************************************/
/************************************************************************************************************************************************************************/
/******** Write all terms up to second order *******************************************************************************************************************************/

void write_cleft(char file[])
{
  int i, n0 = 13, n1 = 10, n2 = 14;
  double r;
  char filename[strlen(file)+1+7];
  FILE *fi;
  
  strcpy(filename,file);
  strcat(filename,".cleft");
  
  printf("Calculating xi, v, sigma contributions...\n");

  fi = fopen(filename,"w");
  fprintf(fi,"# All contributions\n");
  fprintf(fi,"# r, xi_L, xi_1, xi_b1, xi_b2, xi_b1^2, xi_b1.b2, xi_b2^2, xi_bs, xi_bs^2, xi_b1.bs, xi_b2bs, xi_A, xi_bn2, xi_b1.bn2, ");
  fprintf(fi,"v_1, v_b1, v_b2, v_b1^2, v_b1.b2, v_A, v_Ap, v_bn2, v_bs, v_b1.bs, ");
  fprintf(fi,"spar_1, spar_b1, spar_b2, spar_b1^2, spar_bs, spar_A, spar_B, s_1, s_b1, s_b2, s_b1^2, s_bs, s_A, s_B\n");
  
  for (r=1; r<=r_max; r+=1){
    double M0[n0],M1[n1],M2[n2];
    	
    gl_int6(0, 50, &M_0, &r, M0, n0);
    gl_int6(0, 50, &M_1, &r, M1, n1);
    gl_int6(0, 50, &M_2, &r, M2, n2);
    M0[0] -= 1;
     
    fprintf(fi,"%.13le %.13le ", r, iXi_L(r));
    for (i=0;i<n0;i++) fprintf(fi,"%.13le ", M0[i]);
    for (i=0;i<n1;i++) fprintf(fi,"%.13le ", M1[i]);
    for (i=0;i<n2;i++) fprintf(fi,"%.13le ", M2[i]);
    fprintf(fi,"\n");
  }
  fclose(fi);
  
  return;   	  
}

/************************************************************************************************************************************************************************/
/************\\Count the number of ligne in a file\\********************************************************************************************************************/

void compte(FILE *fichier, int* fcounts)
{
  int c;
  int nLines = 0;
  int c2 = '\0';
  int nHeader = 0;
  
  while((c=fgetc(fichier)) != EOF){
    if(c=='\n')	
      nLines++;	
    c2 = c;
  }
  if(c2 != '\n') 
    nLines++;  
  
  rewind(fichier);
  char line[BUFSIZ];
  while(fgets(line, sizeof(line), fichier)){
    if(*line == '#')
      nHeader++;
  }
  
  fcounts[0] = nLines - nHeader;
  fcounts[1] = nHeader;

  return;
}

void jfuncs(const double q, double sum[])
{
  // Computes the \mathcal{J}_n integrals, which are used in the shear terms, and the other shear-related terms.
  double sum0=0,sum1=0,sum2=0,sum3=0,sum4=0,sum5=0,sum6=0,sum7=0,sum8=0,sum9=0,sum10=0;
  double zeta;
  double qmin = log(1e-5);
  double qmax = log(1e2);
  
  int Nint=(int)(10*exp(qmax)*q+512);
  double hh=(qmax-qmin)/(double)Nint;
  //if (Nint>=20000) Nint=20000;
  
#pragma omp parallel for reduction(+:sum0,sum1,sum2,sum3,sum4,sum5,sum6,sum7,sum8,sum9,sum10)
  for (int i=1; i<Nint; ++i) {
    double xx = qmin+i*hh;
    double ap = cos(M_PI/2.*exp(xx-qmax));
    double kk = exp(xx);
    double k2 = kk*kk;
    double kq = kk*q;
    double pk = P_L(kk);
    double j0 = func_bessel_0(kq);
    double j1 = func_bessel_1(kq);
    double j2,j3,j4;
    int wt = 2+2*(i%2);

    j2 = 3.*j1/kq-j0;
    j3 = 5.*j2/kq-j1;
    j4 = 7.*j3/kq-j2;

    sum0  += kk*k2*k2*pk*j0*wt*ap; //*exp(-k2)
    sum1  += k2*k2*pk*j1*wt*ap;
    sum2  += k2*pk*kk*j2*wt;
    sum3  += k2*pk*(2./15.*j1-1./5.*j3)*wt*ap;
    sum4  += k2*pk*(-1./5.*j1-1./5.*j3)*wt;
    sum5  += k2*pk*(j3)*wt;
    sum6  += k2*pk*kk*(-14*j0-40*j2+9*j4)/315.*wt*ap;
    sum7  += k2*pk*kk*(  7*j0+10*j2+3*j4)/105.*wt*ap;
    sum8  += k2*pk*kk*(        4*j2-3*j4)/ 21.*wt*ap;
    sum9  += k2*pk*kk*(       -3*j2-3*j4)/ 21.*wt*ap;
    sum10 += k2*pk*kk*(               j4)     *wt*ap;
  }
  sum6  *= hh/3.0/(2*M_PI*M_PI);
  sum7  *= hh/3.0/(2*M_PI*M_PI);
  sum8  *= hh/3.0/(2*M_PI*M_PI);
  sum9  *= hh/3.0/(2*M_PI*M_PI);
  sum10 *= hh/3.0/(2*M_PI*M_PI);
  
  zeta = sum6*( 9*sum6+12*sum7+12*sum8+ 8*sum9+ 2*sum10)+
         sum7*(        24*sum7+ 8*sum8+32*sum9+ 4*sum10)+
         sum8*(               + 8*sum8+16*sum9+ 4*sum10)+
         sum9*(                        24*sum9+ 8*sum10)+
         sum10*(                                  sum10);

  sum[0] = sum0 * hh/3.0/(2*M_PI*M_PI);	// mathcal{B}_1
  sum[1] = sum1 * hh/3.0/(2*M_PI*M_PI);	// mathcal{B}_2
  sum[2] = sum2 * hh/3.0/(2*M_PI*M_PI);	// mathcal{J}_1
  sum[3] = sum3 * hh/3.0/(2*M_PI*M_PI);	// mathcal{J}_2
  sum[4] = sum4 * hh/3.0/(2*M_PI*M_PI);	// mathcal{J}_3
  sum[5] = sum5 * hh/3.0/(2*M_PI*M_PI);	// mathcal{J}_4
  sum[6] = 4    * sum[2]*sum[3];	// V_i^{12}
  sum[7] = 4./3.* sum[2]*sum[2];	// chi12
  sum[8] = 2 * zeta;			// zeta

  return;
}

void qfuncs(const double q, double sum[])
{
  int i;
  double sum0=0,sum1=0,sum2=0,sum3=0,sum4=0,sum5=0,sum6=0,sum7=0,sum8=0,sum9=0,sum10=0,sum11=0,sum12=0,sumS=0;
  double qmin = log(1e-5);
  double qmax = log(1e2);
  
  int Nint=(int)(10*exp(qmax)*q+512);
  double hh=(qmax-qmin)/(double)Nint;
  //if (Nint>=20000) Nint=20000;
   
  #pragma omp parallel for reduction(+:sum0,sum1,sum2,sum3,sum4,sum5,sum6,sum7,sum8,sum9,sum10,sum11,sum12,sumS)
  for (i=1; i<Nint; ++i) {
    double xx = qmin+i*hh;
    double kk = exp(xx);
    double k2 = kk*kk;
    double kq = kk*q;
    double R1 = R_1(kk);
    double R2 = R_2(kk);
    double Q1 = Q_1(kk);
    double Q2 = Q_2(kk);
    double Q5 = Q_5(kk);
    double Q8 = Q_8(kk);
    double Qs = Q_s(kk);
    double j0 = func_bessel_0(kq);
    double j1 = func_bessel_1(kq);
    double j2,j3;
    double j1d = j1/kq;
    int wt = 2+2*(i%2);
    
    if (kq<0.1) {
      j2 = pow(kq,2.0)/15.  - pow(kq,4.0)/210.;
      j3 = pow(kq,3.0)/105. - pow(kq,5.0)/1890.;
    } else {
      j2 = 3.*j1/kq-j0;
      j3 = 5.*j2/kq-j1;
    }
    
    sum0 += kk*wt*(9./98.*Q1*(2./3.-2*j1d));		// X^{(22)}
    sum1 += kk*wt*(5./21.*R1*(2./3.-2*j1d));		// X^{(13)}
    sum2 += kk*wt*(9./98.*Q1*(-2*j0+6*j1d));		// Y^{(22)}
    sum3 += kk*wt*(5./21.*R1*(-2*j0+6*j1d));		// Y^{(13)}
    sum4 += kk*wt*(2*(R1-R2)+3*R1*j0-3*(3*R1+4*R2+2*Q5)*j1d)/14.;//X1210
    sum5 += kk*wt*(3*R1+4*R2+2*Q5)*(j0-3*j1d)*(-3./14.);// Y_{10}^{(12)}
    sum6 +=    wt*(R1*j1)*(-3./7.);			// V_1^{(112)}
    sum7 +=    wt*(Q1*j1)*(-3./7.);			// V_3^{(112)}
    sumS +=    wt*(2*R1+4*R2+Q1+2*Q2)*(3./7.*j2/(kk*q));// S^{(112)}
    sum8 +=    wt*(2*R1+4*R2+Q1+2*Q2)*j3*(-3./7.);	// T^{(112)}
    sum9 += k2*wt*(R1*j1)*(-5./21.);			// U^{(3)}
    sum10+= k2*wt*(Q8*j1)*(-3./7.);			// U_{20}^{(2)}
    sum11+= k2*wt*((R1+R2)*j1)*(-6./7.);		// U_{11}^{(2)}
    sum12+= k2*wt*(Qs*j1)*(-1./7.);		        // Shear term
  }
  
  sum6 += sumS;
  sum7 += sumS;
  
  sum[ 1] = sum0 * hh/3.0/(2*M_PI*M_PI);
  sum[ 2] = sum1 * hh/3.0/(2*M_PI*M_PI);
  sum[ 4] = sum2 * hh/3.0/(2*M_PI*M_PI);
  sum[ 5] = sum3 * hh/3.0/(2*M_PI*M_PI);
  sum[ 6] = sum4 * hh/3.0/(2*M_PI*M_PI);
  sum[ 7] = sum5 * hh/3.0/(2*M_PI*M_PI);
  sum[ 8] = sum6 * hh/3.0/(2*M_PI*M_PI);
  sum[ 9] = sum7 * hh/3.0/(2*M_PI*M_PI);	
  sum[10] = sum8 * hh/3.0/(2*M_PI*M_PI);	
  sum[12] = sum9 * hh/3.0/(2*M_PI*M_PI);	
  sum[13] = sum10* hh/3.0/(2*M_PI*M_PI);	
  sum[14] = sum11* hh/3.0/(2*M_PI*M_PI);	
  sum[15] = sum12* hh/3.0/(2*M_PI*M_PI);
  
  // Now tabulate the pieces going as Plin.
  sum0 = sum1 = sum2 = 0;
  
  #pragma omp parallel for reduction(+:sum0,sum1,sum2)
  for (int i=1; i<Nint; ++i) {
    double xx = qmin+i*hh;
    double kk = exp(xx);
    double k2 = kk*kk;
    double kq = kk*q;
    double pk = P_L(kk);
    double j0 = func_bessel_0(kq);
    double j1 = func_bessel_1(kq);
    int wt = 2+2*(i%2);
    
    sum0 += kk*wt*pk*(2./3.-2*j1/kq);         // X^{(11)}
    sum1 += kk*wt*pk*(-2.*j0+6*j1/kq);        // Y^{(11)}
    sum2 += k2*wt*pk*(-j1);                   // U^{(1)}
  }
  
  sum[ 0] = sum0 * hh/3.0/(2*M_PI*M_PI);
  sum[ 3] = sum1 * hh/3.0/(2*M_PI*M_PI);
  sum[11] = sum2 * hh/3.0/(2*M_PI*M_PI);
  
  return;
}

void write_qfuncs_fast(void)
 {
   FILE *f;
   int i;
   double q, logq;
   double qmin = 0.001;
   double qmax = 2000; // Maximum q to compute the q functions, 2000 by default                                                                        
   int nq = 3000; // nombre de pas                                                                                                                       
   double log_qmin = log10(qmin);
   double log_qmax = log10(qmax);
   double dq = (log_qmax - log_qmin)/nq;

   printf("getting qfunctions...\n");
   f = fopen("data/qfuncs_fast.dat","w+");
   
   for (logq=log_qmin; logq<log_qmax; logq+=dq){
     q = pow(10, logq);
     double funcs[16];
     qfuncs(q,funcs);
     fprintf(f,"%.13le %.13le ",q,Xi_L(q));
     for (i=0; i<16; i++) fprintf(f,"%.13le ",funcs[i]);
     fprintf(f,"\n");
   }
   
   fclose(f);
   return;
 }

void write_jfuncs_fast(void)
 {
   FILE *f;
   int i;
   double q, logq;
   double qmin = 0.001;
   double qmax = 2000; // Maximum q to compute the q functions, 2000 by default                                                                        
   int nq = 3000; // nombre de pas                                                                                                                       
   double log_qmin = log10(qmin);
   double log_qmax = log10(qmax);
   double dq = (log_qmax - log_qmin)/nq;

   printf("getting jfunctions...\n");
   f = fopen("data/jfuncs_fast.dat","w+");
   
   for (logq=log_qmin; logq<log_qmax; logq+=dq){
     q = pow(10, logq);
     double funcs[9];
     jfuncs(q,funcs);
     fprintf(f,"%.13le ",q);
     for (i=0; i<9; i++) fprintf(f,"%.13le ",funcs[i]);
     fprintf(f,"\n");
   }
   
   fclose(f);
   return;
 }

/************************************************************************************************************************************************************************/
/********Write all CLT components ******************************************************************************************************************************/

void initialize_CLEFT(char pk_filename[])
{
  // Get power spectrum
  int nLines = 0;
  int nHeader = 0;
  FILE *ps;
  
  // Count number of lines
  ps = fopen(pk_filename, "r"); 
  int fcounts[2];
  if (ps!=0){
    compte(ps, fcounts);
    nLines = fcounts[0];
    nHeader = fcounts[1];
  } else {
    printf("Error opening power spectrum file.\n");
    exit(1);
  }
  
  // Read kmin and kmax	
  rewind(ps);
  kmin=1e30;
  kmax=0;	
  
  for (int i = 0; i < nHeader; i++) fscanf(ps,"%*[^\n]\n");
  for (int i = 0; i < nLines; i++) {
    double val;
    fscanf(ps, "%lf %*f\n", &val);
    if (val<kmin) kmin=val;
    if (val>kmax) kmax=val;
  }
  fclose(ps);
  
  interpole(1, pk_filename, nLines, nHeader);
  
  // Compute CLEFT components
  write_R1();
  write_R2();
  write_Qn();
  
  interpole(2,"data/func_R1.dat", 11001, 0);
  interpole(3,"data/func_R2.dat", 11001, 0);
  interpole(4,"data/func_Q1.dat", 3001, 0);
  interpole(5,"data/func_Q2.dat", 3001, 0);
  interpole(6,"data/func_Q5.dat", 3001, 0);
  interpole(7,"data/func_Q8.dat", 3001, 0);
  interpole(24,"data/func_Qs.dat", 3001, 0);

  /*
  q_function(); // Compute q functions and Xi_L
  interpole(8,"data/Xi_L_func.dat", 3002, 0); 
  interp_qfunc();
  exit(1);
  */

  //write_qfuncs();
  
  write_qfuncs_fast();
  interpole_qfuncs(1);
  
  write_jfuncs_fast();
  interpole_jfuncs();

  write_cleft(pk_filename);
  
  //write_Xi_R();
  //write_V12();
  //write_Sigma();
  
  // Finalize
  
  for (int i=0; i<34; i++){
    gsl_spline_free (spline[i]);
    gsl_interp_accel_free (acc[i]);
  }
}

/************************************************************************************************************************************************************************/
/**********\\MAIN FUNCTION\\***************************************************************************************************************************************/

int main (int argc, char *argv[])
{	
  initialize_CLEFT(argv[1]);
  return 0;
}
