/* ============================================================ *
 * CONTRIBUTORS:    Antoine Rocher, Michel-Andrès Breton,       *
 *                  Sylvain de la Torre, Martin Kärcher         *
 * 							                                    *
 * Code for computing Gaussian Streaming for ZA/CLPT/CLEFT      *
 * ============================================================ */

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <gsl/gsl_sf_legendre.h>
#include <gsl/gsl_sf_bessel.h>
#include <gsl/gsl_integration.h>
#include <gsl/gsl_errno.h>
#include <gsl/gsl_interp.h>
#include <gsl/gsl_spline.h>
#include <time.h>

/************************************************************************************************************************************************************************/

gsl_interp_accel *acc[4];
gsl_spline *spline[4];
const double y_spanning = 50;
double rmin, rmax;
double **data;
unsigned int nrs;

/*******************\\ Gauss-Legendre integral quadrature\\**************************************************************************************************************/

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
struct my_f_params { double a; double b; double c; double d; double e;};
struct my_f_params_small {double a; double b; double c; double d;};

double Gauss_Legendre_Integration2_100pts(double a, double b, double (*f)(double, double[]), void *prms) {
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

void Gauss_Legendre_Integration2_100pts_array(double a, double b, void (*f)(double, double[], double[]), double prms[], double result[], int n) {
    double integral[n];
    for (unsigned int i = 0; i < n; i++)
        integral[i] = 0;
    const double c = 0.5 * (b - a);
    const double d = 0.5 * (b + a);
    double dum;
    const double *px = &x[NUM_OF_POSITIVE_ZEROS - 1];
    const double *pA = &A[NUM_OF_POSITIVE_ZEROS - 1];
    double outi[n], outs[n];

    for (; px >= x; pA--, px--) {
        dum = c * *px;
        (*f)(d - dum, prms, outi);
        (*f)(d + dum, prms, outs);
        for (unsigned int i = 0; i < n; i++)
            integral[i] += *pA * (outi[i] + outs[i]);
    }
    for (unsigned int i = 0; i < n; i++)
        result[i] = c * integral[i];
}

/*************\\ Moments \\***********************************************************************************************************************************/

double gauss(double x, double moy, double var) {
    return 1. / (sqrt(2. * M_PI * var)) * exp(-0.5 * (x - moy) * (x - moy) / var);
}

double Xi(double r) {
    if (r < rmin)
        r = rmin;
    if (r > rmax)
        r = rmax;

    return gsl_spline_eval(spline[0], r, acc[0]);
}

void interpXi(double p[]) {
    // xi_L, xi_1, xi_b1, xi_b2, xi_b1^2, xi_b1.b2, xi_b2^2, xi_bs, xi_bs^2, xi_b1.bs, xi_b2.bs, xi_A, xi_bn2, xi_b1.bn2
    double xi[nrs];

    for (unsigned int i = 0; i < nrs; i++)
        xi[i] = data[2][i] + p[1] * data[3][i] + p[2] * data[4][i] + p[1] * p[1] * data[5][i] + p[1] * p[2] * data[6][i] + p[2] * p[2] * data[7][i] + p[3] * data[8][i] + p[3] * p[3] * data[9][i] + p[1] * p[3] * data[10][i] + p[2] * p[3] * data[11][i] + p[4] * data[13][i] + p[1] * p[4] * data[14][i] + p[8] * data[12][i];

    acc[0] = gsl_interp_accel_alloc();
    spline[0] = gsl_spline_alloc(gsl_interp_cspline, nrs);
    gsl_spline_init(spline[0], data[0], xi, nrs);
}

double V12(double r) {
    if (r < rmin)
        r = rmin;
    if (r > rmax)
        r = rmax;

    return gsl_spline_eval(spline[1], r, acc[1]);
}

void interpV12(double p[]) {
    // v_1, v_b1, v_b2, v_b1^2, v_b1.b2, v_A, v_Ap, v_bn2, v_bs, v_b1.bs
    double v12[nrs];

    for (unsigned int i = 0; i < nrs; i++)
        v12[i] = data[15][i] + p[1] * data[16][i] + p[2] * data[17][i] + p[1] * p[1] * data[18][i] + p[1] * p[2] * data[19][i] + p[3] * data[23][i] + p[1] * p[3] * data[24][i] + p[4] * data[22][i] + p[9] * data[20][i] + p[10] * data[21][i];

    acc[1] = gsl_interp_accel_alloc();
    spline[1] = gsl_spline_alloc(gsl_interp_cspline, nrs);
    gsl_spline_init(spline[1], data[0], v12, nrs);
}

double Sigma12(double xi, double mu, double r) {
    double spar, sper;

    if (r < rmin)
        r = rmin;
    if (r > rmax)
        r = rmax;

    spar = gsl_spline_eval(spline[2], r, acc[2]);
    sper = gsl_spline_eval(spline[3], r, acc[3]);

    return (mu * mu * spar + (1. - mu * mu) * sper) / (1.0 + xi);
}

void interpSigma12(double *p) {
    // spar_1, spar_b1, spar_b2, spar_b1^2, spar_bs, spar_A, spar_B, s_1, s_b1, s_b2, s_b1^2, s_bs, s_A, s_B
    double spa[nrs], spe[nrs];

    for (unsigned int i = 0; i < nrs; i++) {
        spa[i] = data[25][i] + p[1] * data[26][i] + p[2] * data[27][i] + p[1] * p[1] * data[28][i] + p[3] * data[29][i] + p[11] * data[30][i] + p[12] * data[31][i];
        spe[i] = data[32][i] + p[1] * data[33][i] + p[2] * data[34][i] + p[1] * p[1] * data[35][i] + p[3] * data[36][i] + p[11] * data[37][i] + p[12] * data[38][i];
    }

    acc[2] = gsl_interp_accel_alloc();
    spline[2] = gsl_spline_alloc(gsl_interp_cspline, nrs);
    gsl_spline_init(spline[2], data[0], spa, nrs);

    acc[3] = gsl_interp_accel_alloc();
    spline[3] = gsl_spline_alloc(gsl_interp_cspline, nrs);
    gsl_spline_init(spline[3], data[0], spe, nrs);
}

/****************\\ Correlation function in z-space for CLPT prediction \\**********************************************************************************************/

double fXis_cquad(double y, void * p) {

    struct my_f_params_small *params = (struct my_f_params_small*)p;

    const double rperp = (params->b);
    const double fg = (params->c);
    const double sigv = (params->d);


    const double r = sqrt(rperp * rperp + y * y);
    const double xi_r = Xi(r);
    const double mu_r = y / r;
    const double var = fg * fg * Sigma12(xi_r, mu_r, r) + sigv;
    if (var <= 0)
        return 0;
    const double spara = (params->a);
    const double v = fg * V12(r) / (1. + xi_r);
    const double x = spara - y;
    const double mean = mu_r * v;
    return (1. + xi_r) * gauss(x, mean, var);
}

double Xis_cquad(double sperp, double spara, double p[]) {
    double result;
    double error;

    gsl_function F;
    struct my_f_params_small params = {spara, sperp, p[0], p[1]};

    F.function = &fXis_cquad;

    F.params = &params;

    gsl_integration_cquad_workspace *w = gsl_integration_cquad_workspace_alloc(200);

    gsl_integration_cquad(&F, spara - y_spanning, spara + y_spanning, 0, 1e-6, w, &result, &error, NULL);

    gsl_integration_cquad_workspace_free(w);


    return result - 1.;
}

double fXis(double y, double p[]) {
    const double rperp = p[1];
    const double fg = p[2];
    const double sigv = p[3];

    const double r = sqrt(rperp * rperp + y * y);
    const double xi_r = Xi(r);
    const double mu_r = y / r;
    const double var = fg * fg * Sigma12(xi_r, mu_r, r) + sigv;
    if (var <= 0)
        return 0;
    const double spara = p[0];
    const double v = fg * V12(r) / (1. + xi_r);
    const double x = spara - y;
    const double mean = mu_r * v;
    return (1. + xi_r) * gauss(x, mean, var);
}

double Xis(double sperp, double spara, double p[]) {
    double params[4];

    params[0] = spara;
    params[1] = sperp;
    params[2] = p[0];
    params[3] = p[1];

    const double result = Gauss_Legendre_Integration2_100pts(spara - y_spanning, spara + y_spanning, &fXis, params);

    return result - 1.;
}


/****************\\Legendre Multipole with CLPT prediction\\*********************************************************************************************************/

// All multipoles with GL and with GL in streaming
void fmultipole(double mu, double p[], double result[]) {
    const double s = p[0];
    const double apar = p[3];
    const double aper = p[4];
    const double spara = apar * s * mu;
    const double rperp = aper * s * sqrt(1. - mu * mu);
    double par[2];
    par[0] = p[1];
    par[1] = p[2];
    const double xi_s = Xis(rperp, spara, par);

    result[0] = xi_s * gsl_sf_legendre_Pl(0, mu);
    result[1] = xi_s * gsl_sf_legendre_Pl(2, mu);
    result[2] = xi_s * gsl_sf_legendre_Pl(4, mu);
}

void multipole(double s, double p[], double result[]) {
    double par[5] = {s, p[0], p[5], p[6], p[7]};
    Gauss_Legendre_Integration2_100pts_array(0, 1, &fmultipole, par, result, 3);
}

// All multipoles in GL but with cquad in streaming
void fmultipole_cquad(double mu, double p[], double result[]) {
    const double s = p[0];
    const double apar = p[3];
    const double aper = p[4];
    const double spara = apar * s * mu;
    const double rperp = aper * s * sqrt(1. - mu * mu);
    double par[2];
    par[0] = p[1];
    par[1] = p[2];
    const double xi_s = Xis_cquad(rperp, spara, par);

    result[0] = xi_s * gsl_sf_legendre_Pl(0, mu);
    result[1] = xi_s * gsl_sf_legendre_Pl(2, mu);
    result[2] = xi_s * gsl_sf_legendre_Pl(4, mu);
}

void multipole_cquad(double s, double p[], double result[]) {
    double par[5] = {s, p[0], p[5], p[6], p[7]};
    Gauss_Legendre_Integration2_100pts_array(0, 1, &fmultipole_cquad, par, result, 3);
}

// Only the first two multipoles with GL but with cquad in streaming
void fmultipole_cquad_first_two(double mu, double p[], double result[]) {
    const double s = p[0];
    const double apar = p[3];
    const double aper = p[4];
    const double spara = apar * s * mu;
    const double rperp = aper * s * sqrt(1. - mu * mu);
    double par[2];
    par[0] = p[1];
    par[1] = p[2];
    const double xi_s = Xis_cquad(rperp, spara, par);

    result[0] = xi_s * gsl_sf_legendre_Pl(0, mu);
    result[1] = xi_s * gsl_sf_legendre_Pl(2, mu);
}

// The hexadecapole with cquad in multipole and in streaming
double fhexadecapole_cquad(double mu, void * p) {
    struct my_f_params * params = (struct my_f_params *)p;
    double par[2];
    
    double s = (params->a);
    par[0] = (params->b);
    par[1] = (params->c);
    double aper = (params->d);
    double apar = (params->e);

    double spara = apar*s*mu;
    double rperp = aper*s*sqrt(1.-mu*mu);
    double xi_s = Xis_cquad(rperp, spara, par);

    return xi_s*gsl_sf_legendre_Pl(4,mu);
}

// compute multipoles with GL for monopole and quadrupole but cquad for hexadecapole
// Use cquad for streaming thoughout
void multipole_only_hexa_cquad(double s, double p[], double result[]) {
    double par[5];

    par[0] = s;
    par[1] = p[0];
    par[2] = p[5];
    par[3] = p[6];
    par[4] = p[7];

    double error_hexa;

    // Do monopole and quadrupole with GL (but cquad in streaming)
    Gauss_Legendre_Integration2_100pts_array(0, 1, &fmultipole_cquad_first_two, par, result, 2);

    // Do hexadecapole with cquad in multipole and streaming
    gsl_function F_hexa;
    struct my_f_params params = {par[0], par[1], par[2], par[3], par[4]};

    F_hexa.function = &fhexadecapole_cquad;

    F_hexa.params = &params;

    gsl_integration_cquad_workspace *hexadecap = gsl_integration_cquad_workspace_alloc(200);

    gsl_integration_cquad(&F_hexa, 0, 1, 0, 1e-6, hexadecap, &result[2], &error_hexa, NULL);

    gsl_integration_cquad_workspace_free(hexadecap);
}

/***********\\ Reading and interpolation \\*****************************************************************************************************************************/

int GetNumLines(char file[]) {
    FILE *fic;
    int n;

    fic = fopen(file, "r");
    if (fic) {
        int car;
        fpos_t pos;

        car = getc(fic);
        while (car == '#') {
            fscanf(fic, "%*[^\n]\n");
            fgetpos(fic, &pos);
            car = getc(fic);
        }
        fsetpos(fic, &pos);

        n = 0;
        while (!feof(fic)) {
            fscanf(fic, "%*[^\n]\n");
            n++;
        }
        fclose(fic);
        return n;
    } else {
        fprintf(stderr, "Can't read %s file!\n", file);
        return 0;
    }
}

int GetNumCols(char file[]) {
    FILE *fic;
    int n;

    fic = fopen(file, "r");
    if (fic) {
        int car;
        fpos_t pos;

        car = getc(fic);
        while (car == '#') {
            fscanf(fic, "%*[^\n]\n");
            fgetpos(fic, &pos);
            car = getc(fic);
        }
        fsetpos(fic, &pos);

        n = 0;
        while (!feof(fic)) {
            fscanf(fic, "%*s");
            n++;
        }
        fclose(fic);
        return n / GetNumLines(file);
    } else {
        fprintf(stderr, "Can't read %s file!\n", file);
        return 0;
    }
}

void load_CLEFT(char *file) {
    const unsigned int nc = 39;
    char filename[strlen(file) + 1 + 7];
    FILE *fi;

    strcpy(filename, file);
    strcat(filename, ".cleft");

    const unsigned int n = GetNumLines(filename);
    if (GetNumCols(filename) != nc) {
        fprintf(stderr, "Error: unknown file format\n");
        exit(1);
    }

    data = (double **)malloc(sizeof(double *) * nc);
    for (unsigned int i = 0; i < nc; i++)
        data[i] = (double *)malloc(sizeof(double) * n);

    fi = fopen(filename, "r");
    if (fi) {
        int car;
        fpos_t pos;

        car = getc(fi);
        while (car == '#') {
            fscanf(fi, "%*[^\n]\n");
            fgetpos(fi, &pos);
            car = getc(fi);
        }
        fsetpos(fi, &pos);

        for (unsigned int i = 0; i < n; i++)
            for (unsigned int j = 0; j < nc; j++)
                fscanf(fi, "%lf", &data[j][i]);
    }
    fclose(fi);

    rmin = data[0][0];
    rmax = data[0][n - 1];

    nrs = n;
}

void load_CLPT(char *file) {
    const unsigned int nc = 39;
    char filename[strlen(file) + 1 + 7];
    FILE *fi;

    strcpy(filename, file);
    strcat(filename, ".clpt");

    const unsigned int n = GetNumLines(filename);
    if (GetNumCols(filename) != nc) {
        fprintf(stderr, "Error: unknown file format\n");
        exit(1);
    }

    data = (double **)malloc(sizeof(double *) * nc);
    for (unsigned int i = 0; i < nc; i++)
        data[i] = (double *)malloc(sizeof(double) * n);

    fi = fopen(filename, "r");
    if (fi) {
        int car;
        fpos_t pos;

        car = getc(fi);
        while (car == '#') {
            fscanf(fi, "%*[^\n]\n");
            fgetpos(fi, &pos);
            car = getc(fi);
        }
        fsetpos(fi, &pos);

        for (unsigned int i = 0; i < n; i++)
            for (unsigned int j = 0; j < nc; j++)
                fscanf(fi, "%lf", &data[j][i]);
    }
    fclose(fi);

    rmin = data[0][0];
    rmax = data[0][n - 1];

    nrs = n;
}

void load_ZA(char *file) {
    const unsigned int nc = 39;
    char filename[strlen(file) + 1 + 7];
    FILE *fi;

    strcpy(filename, file);
    strcat(filename, ".za");

    const unsigned int n = GetNumLines(filename);
    if (GetNumCols(filename) != nc) {
        fprintf(stderr, "Error: unknown file format\n");
        exit(1);
    }

    data = (double **)malloc(sizeof(double *) * nc);
    for (unsigned int i = 0; i < nc; i++)
        data[i] = (double *)malloc(sizeof(double) * n);

    fi = fopen(filename, "r");
    if (fi) {
        int car;
        fpos_t pos;

        car = getc(fi);
        while (car == '#') {
            fscanf(fi, "%*[^\n]\n");
            fgetpos(fi, &pos);
            car = getc(fi);
        }
        fsetpos(fi, &pos);

        for (unsigned int i = 0; i < n; i++)
            for (unsigned int j = 0; j < nc; j++)
                fscanf(fi, "%lf", &data[j][i]);
    }
    fclose(fi);

    rmin = data[0][0];
    rmax = data[0][n - 1];

    nrs = n;
}



// This is a wrappable loading function where we can give a pointer to a numpy ingredients file directly, rather than specifying a filename
// I am assuming here that the ingredinets_file is stored by python as a C-contiguous array in memory (this can be checked once we write the wrapper
// Also I am assuming that the first dimension of the array is the number of bins in r and then the second dimension is the number of ingredients (in this case 39)
void load_CLEFT_wrappable(double *data_in, const unsigned int nrows) {
    // Ok so we just hardcode the number of ingredients to be 39 as we won't change that
    const unsigned int ncols = 39;

    // Allocate memory for ingredients file
    data = (double **)malloc(sizeof(double *) * ncols);
    for (unsigned int j = 0; j < ncols; j++)
        data[j] = (double *)malloc(sizeof(double) * nrows);

    // Write the ingredients into the global data array
    for (unsigned int i = 0; i < nrows; i++)
        for (unsigned int j = 0; j < ncols; j++)
            // Standard 'accessing a 2D array flattened into 1D'
            data[j][i] = data_in[i * ncols + j];

    rmin = data[0][0];
    rmax = data[0][nrows - 1];

    nrs = nrows;
}

void free_CLEFT(void) {
    for (int i = 0; i < 39; i++) {
        free(data[i]);
    }
    free(data);
}

/************************************************************************************************************************************************************************/
/***********\\ Compute multipoles CLPT \\*****************************************************************************************************************************/

void get_prediction_ZA(double *s_array, int nbins, double out[],
                       double in_f, double in_b1, double in_sigv, double in_alpha_par, double in_alpha_per) {
    double out_tmp[3], par[13];

    par[0] = in_f;
    par[1] = in_b1;
    par[2] = 0;
    par[3] = 0;
    par[4] = 0;
    par[5] = in_sigv;
    par[6] = in_alpha_par;
    par[7] = in_alpha_per;
    par[8] = par[9] = par[10] = par[11] = par[12] = 0;

    interpXi(par);
    interpV12(par);
    interpSigma12(par);

    for (unsigned int i = 0; i < nbins; i++) {
        if (s_array[i] < 25){ // If below 25Mpc/h then use cquad for the hexadecapole but GL for the monopole/quadrupole and cquad for the streaming throughout
            multipole_only_hexa_cquad(s_array[i], par, out_tmp);
        }
        else if (s_array[i] >= 25 && s_array[i] < 50){ // If in this range use GL for all multipoles but cquad for the streaming
            multipole_cquad(s_array[i], par, out_tmp);
        }
        else{ // Use GL for streaming and multipoles
            multipole(s_array[i], par, out_tmp);
        }
        out[i] = out_tmp[0];
        out[nbins + i] = 5 * out_tmp[1];
        out[2 * nbins + i] = 9 * out_tmp[2];
    }

    for (unsigned int i = 0; i < 4; i++) {
        gsl_spline_free(spline[i]);
        gsl_interp_accel_free(acc[i]);
    }
    free_CLEFT();
}

void get_prediction_ZA_tmp_fitting(double *s_array, int nbins, double out[],
                       double in_f, double in_b1, double in_sigv, double in_alpha_par, double in_alpha_per) {
    double out_tmp[3], par[13];

    par[0] = in_f;
    par[1] = in_b1;
    par[2] = 0;
    par[3] = 0;
    par[4] = 0;
    par[5] = in_sigv;
    par[6] = in_alpha_par;
    par[7] = in_alpha_per;
    par[8] = par[9] = par[10] = par[11] = par[12] = 0;

    interpXi(par);
    interpV12(par);
    interpSigma12(par);

    for (unsigned int i = 0; i < nbins; i++) {
        if (s_array[i] < 25){ // If below 25Mpc/h then use cquad for the hexadecapole but GL for the monopole/quadrupole and cquad for the streaming throughout
            multipole_only_hexa_cquad(s_array[i], par, out_tmp);
        }
        else if (s_array[i] >= 25 && s_array[i] < 50){ // If in this range use GL for all multipoles but cquad for the streaming
            multipole_cquad(s_array[i], par, out_tmp);
        }
        else{ // Use GL for streaming and multipoles
            multipole(s_array[i], par, out_tmp);
        }
        out[i] = out_tmp[0];
        out[nbins + i] = 5 * out_tmp[1];
        out[2 * nbins + i] = 9 * out_tmp[2];

    }

    for (unsigned int i = 0; i < 4; i++) {
        gsl_spline_free(spline[i]);
        gsl_interp_accel_free(acc[i]);
    }
}



void get_prediction_CLPT(double *s_array, int nbins, double out[],
                         double in_f, double in_b1, double in_b2, double in_sigv, double in_alpha_par, double in_alpha_per) {
    double out_tmp[3], par[13];

    par[0] = in_f;
    par[1] = in_b1;
    par[2] = in_b2;
    par[3] = 0;
    par[4] = 0;
    par[5] = in_sigv;
    par[6] = in_alpha_par;
    par[7] = in_alpha_per;
    par[8] = par[9] = par[10] = par[11] = par[12] = 0;

    interpXi(par);
    interpV12(par);
    interpSigma12(par);

    for (unsigned int i = 0; i < nbins; i++) {
        if (s_array[i] < 25){ // If below 25Mpc/h then use cquad for the hexadecapole but GL for the monopole/quadrupole and cquad for the streaming throughout
            multipole_only_hexa_cquad(s_array[i], par, out_tmp);
        }
        else if (s_array[i] >= 25 && s_array[i] < 50){ // If in this range use GL for all multipoles but cquad for the streaming
            multipole_cquad(s_array[i], par, out_tmp);
        }
        else{ // Use GL for streaming and multipoles
            multipole(s_array[i], par, out_tmp);
        }
        out[i] = out_tmp[0];
        out[nbins + i] = 5 * out_tmp[1];
        out[2 * nbins + i] = 9 * out_tmp[2];
    }

    for (unsigned int i = 0; i < 4; i++) {
        gsl_spline_free(spline[i]);
        gsl_interp_accel_free(acc[i]);
    }
    free_CLEFT();
}

void get_prediction_CLPT_tmp_fitting(double *s_array, int nbins, double out[],
                         double in_f, double in_b1, double in_b2, double in_sigv, double in_alpha_par, double in_alpha_per) {
    double out_tmp[3], par[13];

    par[0] = in_f;
    par[1] = in_b1;
    par[2] = in_b2;
    par[3] = 0;
    par[4] = 0;
    par[5] = in_sigv;
    par[6] = in_alpha_par;
    par[7] = in_alpha_per;
    par[8] = par[9] = par[10] = par[11] = par[12] = 0;

    interpXi(par);
    interpV12(par);
    interpSigma12(par);

    for (unsigned int i = 0; i < nbins; i++) {
        if (s_array[i] < 25){ // If below 25Mpc/h then use cquad for the hexadecapole but GL for the monopole/quadrupole and cquad for the streaming throughout
            multipole_only_hexa_cquad(s_array[i], par, out_tmp);
        }
        else if (s_array[i] >= 25 && s_array[i] < 50){ // If in this range use GL for all multipoles but cquad for the streaming
            multipole_cquad(s_array[i], par, out_tmp);
        }
        else{ // Use GL for streaming and multipoles
            multipole(s_array[i], par, out_tmp);
        }
        out[i] = out_tmp[0];
        out[nbins + i] = 5 * out_tmp[1];
        out[2 * nbins + i] = 9 * out_tmp[2];
    }

    for (unsigned int i = 0; i < 4; i++) {
        gsl_spline_free(spline[i]);
        gsl_interp_accel_free(acc[i]);
    }
}

void get_prediction_CLPT_only_xi_realspace(double *s_array, int nbins, double out[], double in_b1, double in_b2, double in_bs2, double in_bn2) {
    double par[13];

    par[0] = 0;
    par[1] = in_b1;
    par[2] = in_b2;
    par[3] = in_bs2; // This is most likely b_s2
    par[4] = in_bn2; // This is most likely b_nabla2
    par[5] = 0;
    par[6] = 0;
    par[7] = 0;
    par[8] = par[9] = par[10] = par[11] = par[12] = 0;

    interpXi(par);

    for (unsigned int i = 0; i < nbins; i++) {
        out[i] = Xi(s_array[i]);
    }

    // Now we only have to free the spline from xi
    gsl_spline_free(spline[0]);
    gsl_interp_accel_free(acc[0]);


    free_CLEFT();
}


void get_prediction_CLEFT(double *s_array, int nbins, double out[],
                          double in_f, double in_b1, double in_b2, double in_bs, double in_ax, double in_av, double in_as, double in_alpha_par, double in_alpha_per) {
    double par[13], out_tmp[3];

    par[0] = in_f;
    par[1] = in_b1;
    par[2] = in_b2;
    par[3] = in_bs;
    par[4] = 0;
    par[5] = 0;
    par[6] = in_alpha_par;
    par[7] = in_alpha_per;
    par[8] = in_ax;
    par[9] = in_av;
    par[10] = 0;
    par[11] = in_as;
    par[12] = 0;

    // clock_t begin = clock();

    interpXi(par);
    interpV12(par);
    interpSigma12(par);

    for (unsigned int i = 0; i < nbins; i++) {
        if (s_array[i] < 25){ // If below 25Mpc/h then use cquad for the hexadecapole but GL for the monopole/quadrupole and cquad for the streaming throughout
            multipole_only_hexa_cquad(s_array[i], par, out_tmp);
        }
        else if (s_array[i] >= 25 && s_array[i] < 50){ // If in this range use GL for all multipoles but cquad for the streaming
            multipole_cquad(s_array[i], par, out_tmp);
        }
        else{ // Use GL for streaming and multipoles
            multipole(s_array[i], par, out_tmp);
        }
        out[i] = out_tmp[0];
        out[nbins + i] = 5 * out_tmp[1];
        out[2 * nbins + i] = 9 * out_tmp[2];
    }

    for (unsigned int i = 0; i < 4; i++) {
        gsl_spline_free(spline[i]);
        gsl_interp_accel_free(acc[i]);
    }

    free_CLEFT();
}

void get_prediction_CLEFT_tmp_fitting(double *s_array, int nbins, double out[],
                          double in_f, double in_b1, double in_b2, double in_bs, double in_ax, double in_av, double in_as, double in_alpha_par, double in_alpha_per) {
    double par[13], out_tmp[3];

    par[0] = in_f;
    par[1] = in_b1;
    par[2] = in_b2;
    par[3] = in_bs;
    par[4] = 0;
    par[5] = 0;
    par[6] = in_alpha_par;
    par[7] = in_alpha_per;
    par[8] = in_ax;
    par[9] = in_av;
    par[10] = 0;
    par[11] = in_as;
    par[12] = 0;

    interpXi(par);
    interpV12(par);
    interpSigma12(par);

    for (unsigned int i = 0; i < nbins; i++) {
        if (s_array[i] < 25){ // If below 25Mpc/h then use cquad for the hexadecapole but GL for the monopole/quadrupole and cquad for the streaming throughout
            multipole_only_hexa_cquad(s_array[i], par, out_tmp);
        }
        else if (s_array[i] >= 25 && s_array[i] < 50){ // If in this range use GL for all multipoles but cquad for the streaming
            multipole_cquad(s_array[i], par, out_tmp);
        }
        else{ // Use GL for streaming and multipoles
            multipole(s_array[i], par, out_tmp);
        }
        out[i] = out_tmp[0];
        out[nbins + i] = 5 * out_tmp[1];
        out[2 * nbins + i] = 9 * out_tmp[2];
    }

    for (unsigned int i = 0; i < 4; i++) {
        gsl_spline_free(spline[i]);
        gsl_interp_accel_free(acc[i]);
    }
}
/******************************************************************************************************************************************************************/
/***************\\MAIN FUNCTION\\**********************************************************************************************************************************/

int main(int argc, char *argv[]) {
    double out[120];

    load_CLEFT(argv[1]);
    printf("> CLEFT predictions read\n");

    int ns = 40;
    double smin = 0;
    double smax = 200;
    double ds = (smax - smin) / ns;
    double s_array[ns];
    for (int i = 0; i < ns; i++)
        s_array[i] = smin + i * ds + 0.5 * ds;
    get_prediction_CLPT(s_array, ns, out, 1.0, 0.1, 0.5, 5, 1, 1);

    for (unsigned int i = 0; i < ns; i++)
        printf("%le %le %le %le\n", s_array[i], out[i], out[ns + i], out[2 * ns + i]);

    return 0;
}
